"""
Reusable upscaling engine.

This module contains everything needed to upscale images, folders and archives
without being tied to a specific frontend (CLI, GUI, or worker).  The
:class:`UpscaleEngine` keeps models, the PyTorch context and ICC transforms
loaded between jobs so the GPU stays warm across many jobs, exactly like a bulk
chapter run through the GUI or the CLI.

Only the *postprocess* step runs in a separate process (because of pyvips); the
preprocess and upscale steps run in threads of the calling process, so the model
cache in :attr:`UpscaleEngine.loaded_models` persists across jobs.
"""

from __future__ import annotations

import ctypes
import io
import os
import platform
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Literal, Protocol, TypedDict

import cv2
import numpy as np
import pyvips
import rarfile
from chainner_ext import ResizeFilter, resize
from cv2.typing import MatLike
from PIL import Image, ImageCms, ImageFilter
from PIL.ImageCms import ImageCmsProfile
from rarfile import RarFile
from spandrel import ImageModelDescriptor, ModelDescriptor
from zipfile import ZipFile, ZIP_DEFLATED

sys_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

import spandrel_custom
from nodes.impl.image_utils import normalize, to_uint8
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.utils.utils import get_h_w_c
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node,
)
from progress_controller import ProgressController, ProgressToken

from api import (
    Aborted,
    NodeContext,
    SettingsJson,
    SettingsParser,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

UPSCALE_SENTINEL = (None, None, None, None, None, None, None, None, None)
POSTPROCESS_SENTINEL = (None, None, None, None, None, None, None)

CV2_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_EXTENSIONS = (*CV2_IMAGE_EXTENSIONS, ".avif")
ZIP_EXTENSIONS = (".zip", ".cbz")
RAR_EXTENSIONS = (".rar", ".cbr")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS

_QUEUE_TIMEOUT = 0.5


# --------------------------------------------------------------------------- #
# Progress / result abstractions
# --------------------------------------------------------------------------- #


class ProgressReporter(Protocol):
    def log(self, message: str) -> None: ...

    def archive_total(self, total: int) -> None: ...

    def file_completed(self, kind: str) -> None: ...

    def phase(self, name: str) -> None: ...


class _NoopReporter(ProgressReporter):
    def log(self, message: str) -> None:
        pass

    def archive_total(self, total: int) -> None:
        pass

    def file_completed(self, kind: str) -> None:
        pass

    def phase(self, name: str) -> None:
        pass


class FileResult(TypedDict, total=False):
    input: str
    output: str
    status: str  # "upscaled" | "skipped" | "copied" | "error"
    error: str


class JobResult:
    """Collects per-file results of a single upscale job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files: list[FileResult] = []
        self.error: str | None = None
        self.cancelled: bool = False

    def add(self, entry: FileResult) -> None:
        with self._lock:
            self._files.append(entry)

    @property
    def files(self) -> list[FileResult]:
        with self._lock:
            return list(self._files)


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self, progress: ProgressToken, settings: SettingsParser, storage_dir: Path
    ) -> None:
        super().__init__()

        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir

        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0.001)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

        # TODO: send progress event

    @property
    def settings(self) -> SettingsParser:
        return self.__settings

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


@dataclass
class _JobExecution:
    """Per-job state shared by the threads of a single job."""

    reporter: ProgressReporter
    controller: ProgressController
    context: _ExecutorNodeContext
    result: JobResult = field(default_factory=JobResult)


# --------------------------------------------------------------------------- #
# Stateless helpers
# --------------------------------------------------------------------------- #


def get_tile_size(tile_size_str: str) -> TileSize:
    if tile_size_str == "Auto (Estimate)":
        return ESTIMATE
    elif tile_size_str == "Maximum":
        return MAX_TILE_SIZE
    elif tile_size_str == "No Tiling":
        return NO_TILING
    elif tile_size_str.isdecimal():
        return TileSize(int(tile_size_str))

    return ESTIMATE


def standard_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Lanczos downscale without color conversion, for pre-upscale downscale."""
    new_image = image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.Lanczos, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    _, _, c = get_h_w_c(image)

    if c == 1 and new_image.ndim == 3:
        new_image = np.squeeze(new_image, axis=-1)

    return new_image


def dotgain20_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Final downscale for grayscale images only."""
    h, _, c = get_h_w_c(image)
    size_ratio = h / new_size[1]
    blur_size = (1 / size_ratio - 1) / 3.5
    if blur_size >= 0.1:
        blur_size = min(blur_size, 250)

    pil_image = Image.fromarray(image, mode="L")
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
    pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)

    new_image = np.array(pil_image)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.CubicCatrom, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(new_image[:, :, 0], mode="L")
    pil_image = ImageCms.applyTransform(pil_image, gamma1todotgain20transform, False)
    return np.array(pil_image)


def image_resize(
    image: np.ndarray, new_size: tuple[int, int], is_grayscale: bool
) -> np.ndarray:
    if is_grayscale:
        return dotgain20_resize(image, new_size)

    return standard_resize(image, new_size)


def enhance_contrast(
    image: np.ndarray, log: Callable[[str], None] | None = None
) -> MatLike:
    image_p = Image.fromarray(image).convert("L")

    hist = image_p.histogram()

    new_black_level = 0
    global_max_black = hist[0]

    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i

    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    new_white_level = 255
    global_max_white = hist[255]

    for i in range(254, 224, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i

    continuous_count = 0
    for i in range(223, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    if log is not None:
        log(
            f"Auto adjusted levels: new black level = {new_black_level}; "
            f"new white level = {new_white_level}"
        )

    image_array = np.array(image_p).astype("float32")
    image_array = np.maximum(image_array - new_black_level, 0) / (
        new_white_level - new_black_level
    )
    return np.clip(image_array, 0, 1)


def _read_image(img_stream: bytes, filename: str) -> np.ndarray:
    return _read_vips(img_stream)


def _read_image_from_path(path: str) -> np.ndarray:
    return (
        pyvips.Image.new_from_file(path, access="sequential", fail=True)
        .icc_transform("srgb")
        .numpy()
    )


def _read_vips(img_stream: bytes) -> np.ndarray:
    return (
        pyvips.Image.new_from_buffer(img_stream, "", access="sequential")
        .icc_transform("srgb")
        .numpy()
    )


def cv_image_is_grayscale(image: np.ndarray, user_threshold: float) -> bool:
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = user_threshold

    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)  # type: ignore
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)  # type: ignore
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)  # type: ignore

    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    if size_without_black_and_white == 0:
        return False

    ratio = diff_sum / size_without_black_and_white

    return ratio <= user_threshold / 12


def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    channels = get_h_w_c(image)[2]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return image


def should_chain_activate_for_image(
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    target_scale: float,
    chain: dict[str, Any],
) -> bool:
    min_width, min_height = (int(x) for x in chain["MinResolution"].split("x"))
    max_width, max_height = (int(x) for x in chain["MaxResolution"].split("x"))

    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True


def get_chain_for_image(
    image: np.ndarray,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    grayscale_detection_threshold: int,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], bool, int, int] | tuple[None, None, int, int]:
    original_height, original_width, _ = get_h_w_c(image)

    if target_width != 0 and target_height != 0:
        target_scale = min(
            target_height / original_height, target_width / original_width
        )
    if target_height != 0:
        target_scale = target_height / original_height
    elif target_width != 0:
        target_scale = target_width / original_width

    assert target_scale is not None

    is_grayscale = cv_image_is_grayscale(image, grayscale_detection_threshold)

    for chain in chains:
        if should_chain_activate_for_image(
            original_width, original_height, is_grayscale, target_scale, chain
        ):
            if log is not None:
                log(f"Matched Chain: {chain}")
            return chain, is_grayscale, original_width, original_height

    return None, None, original_width, original_height


def postprocess_image(image: np.ndarray) -> np.ndarray:
    return to_uint8(image, normalized=True)


def final_target_resize(
    image: np.ndarray,
    target_scale: float,
    target_width: int,
    target_height: int,
    original_width: int,
    original_height: int,
    is_grayscale: bool,
) -> np.ndarray:
    if target_height != 0 and target_width != 0:
        h, w, _ = get_h_w_c(image)
        if target_height / original_height < target_width / original_width:
            target_width = 0
        else:
            target_height = 0

    if target_height != 0:
        h, w, _ = get_h_w_c(image)
        if h != target_height:
            return image_resize(
                image, (round(w * target_height / h), target_height), is_grayscale
            )
    elif target_width != 0:
        h, w, _ = get_h_w_c(image)
        if w != target_width:
            return image_resize(
                image, (target_width, round(h * target_width / w)), is_grayscale
            )
    else:
        h, w, _ = get_h_w_c(image)
        new_target_height = round(original_height * target_scale)
        if h != new_target_height:
            return image_resize(
                image,
                (round(w * new_target_height / h), new_target_height),
                is_grayscale,
            )

    return image


def _vips_from_array(image: np.ndarray) -> pyvips.Image:
    vips_img = pyvips.Image.new_from_array(image)
    if vips_img.interpretation == "multiband":
        if vips_img.bands == 1:
            vips_img = vips_img.copy(interpretation="b-w")
        elif vips_img.bands in (3, 4):
            vips_img = vips_img.copy(interpretation="srgb")
    return vips_img


def save_image_zip(
    image: np.ndarray,
    file_name: str,
    output_zip: ZipFile,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
) -> None:
    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
    )

    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
    buf_img = _vips_from_array(image).write_to_buffer(f".{image_format}", **args)
    output_buffer = io.BytesIO(buf_img)  # type: ignore

    upscaled_image_data = output_buffer.getvalue()

    output_zip.writestr(file_name, upscaled_image_data)


def save_image(
    image: np.ndarray,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
) -> None:
    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
    )

    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
    _vips_from_array(image).write_to_file(output_file_path, **args)


# --------------------------------------------------------------------------- #
# Postprocess workers (run in a separate process)
# --------------------------------------------------------------------------- #


def _postprocess_worker_zip(
    postprocess_queue: Queue,
    progress_queue: Queue,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    with ZipFile(output_zip_path, "w", ZIP_DEFLATED) as output_zip:
        while True:
            (
                image,
                file_name,
                is_image,
                is_grayscale,
                original_width,
                original_height,
                input_name,
            ) = postprocess_queue.get()
            if image is None:
                break
            if is_image:
                entry_name = str(Path(file_name).with_suffix(f".{image_format}"))
                progress_queue.put(("log", f"save image to zip: {entry_name}"))
                save_image_zip(
                    image,
                    entry_name,
                    output_zip,
                    image_format,
                    lossy_compression_quality,
                    use_lossless_compression,
                    original_width,
                    original_height,
                    target_scale,
                    target_width,
                    target_height,
                    is_grayscale,
                )
                progress_queue.put(
                    ("result", {"input": input_name, "output": entry_name, "status": "upscaled"})
                )
            else:
                output_zip.writestr(file_name, image)
                progress_queue.put(
                    ("result", {"input": input_name, "output": file_name, "status": "copied"})
                )
            progress_queue.put(("progress", "postprocess_worker_zip_image"))
        progress_queue.put(("progress", "postprocess_worker_zip_archive"))


def _postprocess_worker_folder(
    postprocess_queue: Queue,
    progress_queue: Queue,
    output_folder_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    while True:
        (
            image,
            file_name,
            _is_image,
            is_grayscale,
            original_width,
            original_height,
            input_name,
        ) = postprocess_queue.get()
        if image is None:
            break
        image = postprocess_image(image)
        output_file_path = os.path.join(
            output_folder_path, str(Path(f"{file_name}.{image_format}"))
        )
        progress_queue.put(("log", f"save image: {output_file_path}"))
        save_image(
            image,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        progress_queue.put(
            ("result", {"input": input_name, "output": output_file_path, "status": "upscaled"})
        )
        progress_queue.put(("progress", "postprocess_worker_folder"))


def _postprocess_worker_image(
    postprocess_queue: Queue,
    progress_queue: Queue,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    while True:
        (
            image,
            _file_name,
            _is_image,
            is_grayscale,
            original_width,
            original_height,
            input_name,
        ) = postprocess_queue.get()
        if image is None:
            break

        progress_queue.put(("log", f"save image: {output_file_path}"))
        save_image(
            image,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        progress_queue.put(
            ("result", {"input": input_name, "output": output_file_path, "status": "upscaled"})
        )
        progress_queue.put(("progress", "postprocess_worker_image"))


# --------------------------------------------------------------------------- #
# ICC profiles (module-level so postprocess subprocesses inherit/re-create them)
# --------------------------------------------------------------------------- #

current_file_directory = os.path.dirname(os.path.abspath(__file__))

is_windows = platform.system() == "win32"


def get_system_codepage() -> Any:
    return None if not is_windows else ctypes.windll.kernel32.GetConsoleOutputCP()


def get_gamma_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Custom Gray Gamma 1.0.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def get_dot20_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Dot Gain 20%.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


gamma1icc = get_gamma_icc_profile()
dotgain20icc = get_dot20_icc_profile()

dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
    dotgain20icc, gamma1icc, "L", "L"
)
gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
    gamma1icc, dotgain20icc, "L", "L"
)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class UpscaleEngine:
    """
    Long-lived upscaling engine.

    The constructor loads PyTorch, registers the custom spandrel architectures
    and builds the ICC transforms.  Models are loaded lazily on first use and
    cached in :attr:`loaded_models`, so the GPU stays warm across jobs.
    """

    def __init__(
        self,
        settings: SettingsJson,
        reporter: ProgressReporter | None = None,
        models_directory: str | None = None,
    ) -> None:
        spandrel_custom.install(ignore_duplicates=True)

        self.settings_parser = SettingsParser(
            {
                "use_cpu": settings["SelectedDeviceIndex"] == 0,
                "use_fp16": settings["UseFp16"],
                "accelerator_device_index": settings["SelectedDeviceIndex"],
                "budget_limit": 0,
            }
        )
        self.models_directory = models_directory or settings["ModelsDirectory"]
        self.system_codepage = get_system_codepage()
        self.loaded_models: dict[str, ModelDescriptor] = {}
        self.reporter: ProgressReporter = reporter or _NoopReporter()

    # -- helpers ------------------------------------------------------------ #

    def _log(self, exec: _JobExecution, message: str) -> None:
        exec.reporter.log(message)

    @staticmethod
    def _check_abort(controller: ProgressController) -> None:
        if controller.aborted:
            raise Aborted()

    @staticmethod
    def _put_up(
        upscale_queue: Queue, item: tuple, controller: ProgressController
    ) -> None:
        while True:
            if controller.aborted:
                raise Aborted()
            try:
                upscale_queue.put(item, timeout=_QUEUE_TIMEOUT)
                return
            except queue.Full:
                continue

    def get_model_abs_path(self, chain_model_file_path: str) -> str:
        return os.path.abspath(os.path.join(self.models_directory, chain_model_file_path))

    def _forward_progress(
        self, progress_queue: Queue, reporter: ProgressReporter, result: JobResult
    ) -> None:
        while True:
            item = progress_queue.get()
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                reporter.file_completed(payload)
            elif kind == "result":
                result.add(payload)
            elif kind == "log":
                reporter.log(payload)

    def warmup(self, chains: list[dict[str, Any]]) -> int:
        """Preload all models referenced by ``chains`` into the model cache.

        Returns the number of models actually loaded.  This is optional; the
        model cache is already shared across jobs, so models are loaded lazily
        on first use and stay warm afterwards.
        """
        loaded = 0
        controller = ProgressController()
        context = _ExecutorNodeContext(controller, self.settings_parser, Path())
        for chain in chains:
            model_file_path = chain.get("ModelFilePath")
            if not model_file_path or model_file_path == "No Model":
                continue
            abs_path = self.get_model_abs_path(model_file_path)
            if abs_path in self.loaded_models:
                continue
            if os.path.exists(abs_path):
                model, _, _ = load_model_node(context, Path(abs_path))
                self.loaded_models[abs_path] = model
                loaded += 1
        return loaded

    # -- upscale step ------------------------------------------------------- #

    def ai_upscale_image(
        self,
        context: _ExecutorNodeContext,
        image: np.ndarray,
        model_tile_size: TileSize,
        model: ImageModelDescriptor | None,
    ) -> np.ndarray:
        if model is not None:
            result = upscale_image_node(
                context,
                image,
                model,
                False,
                0,
                model_tile_size,
                256,
                False,
            )

            _, _, c = get_h_w_c(image)

            if c == 1 and result.ndim == 3:
                result = np.squeeze(result, axis=-1)

            return result

        return image

    def _upscale_worker(
        self, exec: _JobExecution, upscale_queue: Queue, postprocess_queue: Queue
    ) -> None:
        try:
            while True:
                if exec.controller.aborted:
                    break
                try:
                    item = upscale_queue.get(timeout=_QUEUE_TIMEOUT)
                except queue.Empty:
                    continue

                (
                    image,
                    file_name,
                    is_image,
                    is_grayscale,
                    original_width,
                    original_height,
                    model_tile_size,
                    model,
                    input_name,
                ) = item
                if image is None:
                    break

                if is_image:
                    image = self.ai_upscale_image(
                        exec.context, image, model_tile_size, model
                    )

                    if is_grayscale:
                        image = convert_image_to_grayscale(image)

                postprocess_queue.put(
                    (
                        image,
                        file_name,
                        is_image,
                        is_grayscale,
                        original_width,
                        original_height,
                        input_name,
                    )
                )
        except Aborted:
            pass
        finally:
            try:
                postprocess_queue.put(POSTPROCESS_SENTINEL, timeout=5)
            except Exception:
                pass

    # -- preprocess step ---------------------------------------------------- #

    def _preprocess_worker_archive(
        self,
        exec: _JobExecution,
        upscale_queue: Queue,
        input_archive_path: str,
        output_archive_path: str,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        if input_archive_path.endswith(ZIP_EXTENSIONS):
            with ZipFile(input_archive_path, "r") as input_zip:
                self._preprocess_worker_archive_file(
                    exec,
                    upscale_queue,
                    input_zip,
                    output_archive_path,
                    target_scale,
                    target_width,
                    target_height,
                    chains,
                    grayscale_detection_threshold,
                )
        elif input_archive_path.endswith(RAR_EXTENSIONS):
            with rarfile.RarFile(input_archive_path, "r") as input_rar:
                self._preprocess_worker_archive_file(
                    exec,
                    upscale_queue,
                    input_rar,
                    output_archive_path,
                    target_scale,
                    target_width,
                    target_height,
                    chains,
                    grayscale_detection_threshold,
                )

    def _preprocess_worker_archive_file(
        self,
        exec: _JobExecution,
        upscale_queue: Queue,
        input_archive: RarFile | ZipFile,
        output_archive_path: str,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
        namelist = input_archive.namelist()
        exec.reporter.archive_total(len(namelist))
        try:
            for filename in namelist:
                if exec.controller.aborted:
                    break
                decoded_filename = filename
                image_data = None
                try:
                    decoded_filename = decoded_filename.encode("cp437").decode(
                        f"cp{self.system_codepage}"
                    )
                except:  # noqa: E722
                    pass

                try:
                    with input_archive.open(filename) as file_in_archive:
                        image_data = file_in_archive.read()

                        image = _read_image(image_data, filename)
                        self._log(exec, f"read image {filename}")
                        chain, is_grayscale, original_width, original_height = (
                            get_chain_for_image(
                                image,
                                target_scale,
                                target_width,
                                target_height,
                                chains,
                                grayscale_detection_threshold,
                                log=exec.reporter.log,
                            )
                        )

                        if is_grayscale:
                            image = convert_image_to_grayscale(image)

                        model = None
                        tile_size_str = ""
                        if chain is not None:
                            resize_width_before_upscale = chain[
                                "ResizeWidthBeforeUpscale"
                            ]
                            resize_height_before_upscale = chain[
                                "ResizeHeightBeforeUpscale"
                            ]
                            resize_factor_before_upscale = chain[
                                "ResizeFactorBeforeUpscale"
                            ]

                            if (
                                resize_height_before_upscale != 0
                                and resize_width_before_upscale != 0
                            ):
                                h, w, _ = get_h_w_c(image)
                                image = standard_resize(
                                    image,
                                    (
                                        resize_width_before_upscale,
                                        resize_height_before_upscale,
                                    ),
                                )
                            elif resize_height_before_upscale != 0:
                                h, w, _ = get_h_w_c(image)
                                image = standard_resize(
                                    image,
                                    (
                                        round(
                                            w * resize_height_before_upscale / h
                                        ),
                                        resize_height_before_upscale,
                                    ),
                                )
                            elif resize_width_before_upscale != 0:
                                h, w, _ = get_h_w_c(image)
                                image = standard_resize(
                                    image,
                                    (
                                        resize_width_before_upscale,
                                        round(
                                            h * resize_width_before_upscale / w
                                        ),
                                    ),
                                )
                            elif resize_factor_before_upscale != 100:
                                h, w, _ = get_h_w_c(image)
                                image = standard_resize(
                                    image,
                                    (
                                        round(
                                            w * resize_factor_before_upscale / 100
                                        ),
                                        round(
                                            h * resize_factor_before_upscale / 100
                                        ),
                                    ),
                                )

                            if is_grayscale and chain["AutoAdjustLevels"]:
                                image = enhance_contrast(image, log=exec.reporter.log)
                            else:
                                image = normalize(image)

                            model_abs_path = self.get_model_abs_path(
                                chain["ModelFilePath"]
                            )

                            if model_abs_path in self.loaded_models:
                                model = self.loaded_models[model_abs_path]

                            elif os.path.exists(model_abs_path):
                                model, _, _ = load_model_node(
                                    exec.context, Path(model_abs_path)
                                )
                                self.loaded_models[model_abs_path] = model

                            tile_size_str = chain["ModelTileSize"]
                        else:
                            image = normalize(image)

                        self._put_up(
                            upscale_queue,
                            (
                                image,
                                decoded_filename,
                                True,
                                is_grayscale,
                                original_width,
                                original_height,
                                get_tile_size(tile_size_str),
                                model,
                                decoded_filename,
                            ),
                            exec.controller,
                        )
                except Aborted:
                    raise
                except Exception as e:
                    self._log(
                        exec,
                        f"could not read as image, copying file to zip instead of "
                        f"upscaling: {decoded_filename}, {e}",
                    )
                    self._put_up(
                        upscale_queue,
                        (
                            image_data,
                            decoded_filename,
                            False,
                            False,
                            None,
                            None,
                            None,
                            None,
                            decoded_filename,
                        ),
                        exec.controller,
                    )
        except Aborted:
            pass
        finally:
            try:
                upscale_queue.put(UPSCALE_SENTINEL, timeout=1)
            except Exception:
                pass

    def _preprocess_worker_folder(
        self,
        exec: _JobExecution,
        upscale_queue: Queue,
        input_folder_path: str,
        output_folder_path: str,
        output_filename: str,
        upscale_images: bool,
        upscale_archives: bool,
        overwrite_existing_files: bool,
        image_format: str,
        lossy_compression_quality: int,
        use_lossless_compression: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        self._log(
            exec,
            f"preprocess_worker_folder entering {input_folder_path} "
            f"{output_folder_path} {output_filename}",
        )
        try:
            for root, _dirs, files in os.walk(input_folder_path):
                for filename in files:
                    if exec.controller.aborted:
                        return
                    input_file_base = Path(filename).stem
                    filename_rel = os.path.relpath(
                        os.path.join(root, filename), input_folder_path
                    )
                    output_filename_rel = os.path.join(
                        os.path.dirname(filename_rel),
                        output_filename.replace("%filename%", input_file_base),
                    )
                    output_file_path = Path(
                        os.path.join(output_folder_path, output_filename_rel)
                    )

                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        if upscale_images:
                            output_file_path = str(
                                Path(f"{output_file_path}.{image_format}")
                            ).replace("%filename%", input_file_base)

                            if not overwrite_existing_files and os.path.isfile(
                                output_file_path
                            ):
                                self._log(exec, f"file exists, skip: {output_file_path}")
                                exec.result.add(
                                    {
                                        "input": os.path.join(
                                            input_folder_path, filename_rel
                                        ),
                                        "output": output_file_path,
                                        "status": "skipped",
                                    }
                                )
                                continue

                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                            image = _read_image_from_path(os.path.join(root, filename))

                            chain, is_grayscale, original_width, original_height = (
                                get_chain_for_image(
                                    image,
                                    target_scale,
                                    target_width,
                                    target_height,
                                    chains,
                                    grayscale_detection_threshold,
                                    log=exec.reporter.log,
                                )
                            )

                            if is_grayscale:
                                image = convert_image_to_grayscale(image)

                            model = None
                            tile_size_str = ""
                            if chain is not None:
                                resize_width_before_upscale = chain[
                                    "ResizeWidthBeforeUpscale"
                                ]
                                resize_height_before_upscale = chain[
                                    "ResizeHeightBeforeUpscale"
                                ]
                                resize_factor_before_upscale = chain[
                                    "ResizeFactorBeforeUpscale"
                                ]

                                if (
                                    resize_height_before_upscale != 0
                                    and resize_width_before_upscale != 0
                                ):
                                    h, w, _ = get_h_w_c(image)
                                    image = standard_resize(
                                        image,
                                        (
                                            resize_width_before_upscale,
                                            resize_height_before_upscale,
                                        ),
                                    )
                                elif resize_height_before_upscale != 0:
                                    h, w, _ = get_h_w_c(image)
                                    image = standard_resize(
                                        image,
                                        (
                                            round(
                                                w * resize_height_before_upscale / h
                                            ),
                                            resize_height_before_upscale,
                                        ),
                                    )
                                elif resize_width_before_upscale != 0:
                                    h, w, _ = get_h_w_c(image)
                                    image = standard_resize(
                                        image,
                                        (
                                            resize_width_before_upscale,
                                            round(
                                                h * resize_width_before_upscale / w
                                            ),
                                        ),
                                    )
                                elif resize_factor_before_upscale != 100:
                                    h, w, _ = get_h_w_c(image)
                                    image = standard_resize(
                                        image,
                                        (
                                            round(
                                                w
                                                * resize_factor_before_upscale
                                                / 100
                                            ),
                                            round(
                                                h
                                                * resize_factor_before_upscale
                                                / 100
                                            ),
                                        ),
                                    )

                                if is_grayscale and chain["AutoAdjustLevels"]:
                                    image = enhance_contrast(
                                        image, log=exec.reporter.log
                                    )
                                else:
                                    image = normalize(image)

                                model_abs_path = self.get_model_abs_path(
                                    chain["ModelFilePath"]
                                )

                                if model_abs_path in self.loaded_models:
                                    model = self.loaded_models[model_abs_path]

                                elif os.path.exists(model_abs_path):
                                    model, _, _ = load_model_node(
                                        exec.context, Path(model_abs_path)
                                    )
                                    self.loaded_models[model_abs_path] = model
                                tile_size_str = chain["ModelTileSize"]
                            else:
                                image = normalize(image)

                            self._put_up(
                                upscale_queue,
                                (
                                    image,
                                    output_filename_rel,
                                    True,
                                    is_grayscale,
                                    original_width,
                                    original_height,
                                    get_tile_size(tile_size_str),
                                    model,
                                    os.path.join(input_folder_path, filename_rel),
                                ),
                                exec.controller,
                            )
                    elif filename.lower().endswith(ARCHIVE_EXTENSIONS):
                        if upscale_archives:
                            output_file_path = f"{output_file_path}.cbz"
                            if not overwrite_existing_files and os.path.isfile(
                                output_file_path
                            ):
                                self._log(exec, f"file exists, skip: {output_file_path}")
                                exec.result.add(
                                    {
                                        "input": os.path.join(
                                            input_folder_path, filename_rel
                                        ),
                                        "output": output_file_path,
                                        "status": "skipped",
                                    }
                                )
                                continue
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                            self.upscale_archive_file(
                                exec,
                                os.path.join(root, filename),
                                output_file_path,
                                image_format,
                                lossy_compression_quality,
                                use_lossless_compression,
                                target_scale,
                                target_width,
                                target_height,
                                chains,
                                grayscale_detection_threshold,
                            )
        except Aborted:
            pass
        finally:
            try:
                upscale_queue.put(UPSCALE_SENTINEL, timeout=1)
            except Exception:
                pass

    def _preprocess_worker_image(
        self,
        exec: _JobExecution,
        upscale_queue: Queue,
        input_image_path: str,
        output_image_path: str,
        overwrite_existing_files: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        try:
            if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
                if not overwrite_existing_files and os.path.isfile(output_image_path):
                    self._log(exec, f"file exists, skip: {output_image_path}")
                    exec.result.add(
                        {
                            "input": input_image_path,
                            "output": output_image_path,
                            "status": "skipped",
                        }
                    )
                    return

                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                image = _read_image_from_path(input_image_path)

                chain, is_grayscale, original_width, original_height = (
                    get_chain_for_image(
                        image,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        grayscale_detection_threshold,
                        log=exec.reporter.log,
                    )
                )

                if is_grayscale:
                    image = convert_image_to_grayscale(image)

                model = None
                tile_size_str = ""
                if chain is not None:
                    resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                    resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
                    resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

                    if (
                        resize_height_before_upscale != 0
                        and resize_width_before_upscale != 0
                    ):
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image, (resize_width_before_upscale, resize_height_before_upscale)
                        )
                    elif resize_height_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_height_before_upscale / h),
                                resize_height_before_upscale,
                            ),
                        )
                    elif resize_width_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                resize_width_before_upscale,
                                round(h * resize_width_before_upscale / w),
                            ),
                        )
                    elif resize_factor_before_upscale != 100:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_factor_before_upscale / 100),
                                round(h * resize_factor_before_upscale / 100),
                            ),
                        )

                    if is_grayscale and chain["AutoAdjustLevels"]:
                        image = enhance_contrast(image, log=exec.reporter.log)
                    else:
                        image = normalize(image)

                    if chain["ModelFilePath"] == "No Model":
                        pass
                    else:
                        model_abs_path = self.get_model_abs_path(chain["ModelFilePath"])

                        if not os.path.exists(model_abs_path):
                            raise FileNotFoundError(model_abs_path)

                        if model_abs_path in self.loaded_models:
                            model = self.loaded_models[model_abs_path]

                        elif os.path.exists(model_abs_path):
                            model, _, _ = load_model_node(
                                exec.context, Path(model_abs_path)
                            )
                            self.loaded_models[model_abs_path] = model
                        tile_size_str = chain["ModelTileSize"]
                else:
                    self._log(exec, "No chain!!!!!!!")
                    image = normalize(image)

                self._put_up(
                    upscale_queue,
                    (
                        image,
                        None,
                        True,
                        is_grayscale,
                        original_width,
                        original_height,
                        get_tile_size(tile_size_str),
                        model,
                        input_image_path,
                    ),
                    exec.controller,
                )
        except Aborted:
            pass
        finally:
            try:
                upscale_queue.put(UPSCALE_SENTINEL, timeout=1)
            except Exception:
                pass

    # -- job runners -------------------------------------------------------- #

    def upscale_archive_file(
        self,
        exec: _JobExecution,
        input_zip_path: str,
        output_zip_path: str,
        image_format: str,
        lossy_compression_quality: int,
        use_lossless_compression: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

        upscale_queue: Queue = Queue(maxsize=1)
        postprocess_queue: Queue = Queue(maxsize=1)
        progress_queue: Queue = Queue()

        forwarder = Thread(
            target=self._forward_progress,
            args=(progress_queue, exec.reporter, exec.result),
            daemon=True,
        )
        forwarder.start()

        preprocess_process = Thread(
            target=self._preprocess_worker_archive,
            args=(
                exec,
                upscale_queue,
                input_zip_path,
                output_zip_path,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            ),
        )
        preprocess_process.start()

        upscale_process = Thread(
            target=self._upscale_worker, args=(exec, upscale_queue, postprocess_queue)
        )
        upscale_process.start()

        postprocess_thread = Thread(
            target=_postprocess_worker_zip,
            args=(
                postprocess_queue,
                progress_queue,
                output_zip_path,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
            ),
        )
        postprocess_thread.start()

        preprocess_process.join()
        upscale_process.join()
        # All pages are upscaled and written to the zip; the remaining work is closing the
        # archive (central directory + flush), which is pure I/O. Report it as a distinct
        # phase so the driver/UI can show "finalizing" instead of a stuck 100%.
        exec.reporter.phase("finalizing")
        postprocess_thread.join()

        try:
            progress_queue.put(None, timeout=1)
        except Exception:
            pass
        forwarder.join(timeout=5)

    def upscale_image_file(
        self,
        exec: _JobExecution,
        input_image_path: str,
        output_image_path: str,
        overwrite_existing_files: bool,
        image_format: str,
        lossy_compression_quality: int,
        use_lossless_compression: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
    ) -> None:
        upscale_queue: Queue = Queue(maxsize=1)
        postprocess_queue: Queue = Queue(maxsize=1)
        progress_queue: Queue = Queue()

        forwarder = Thread(
            target=self._forward_progress,
            args=(progress_queue, exec.reporter, exec.result),
            daemon=True,
        )
        forwarder.start()

        preprocess_process = Thread(
            target=self._preprocess_worker_image,
            args=(
                exec,
                upscale_queue,
                input_image_path,
                output_image_path,
                overwrite_existing_files,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            ),
        )
        preprocess_process.start()

        upscale_process = Thread(
            target=self._upscale_worker, args=(exec, upscale_queue, postprocess_queue)
        )
        upscale_process.start()

        postprocess_thread = Thread(
            target=_postprocess_worker_image,
            args=(
                postprocess_queue,
                progress_queue,
                output_image_path,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
            ),
        )
        postprocess_thread.start()

        preprocess_process.join()
        upscale_process.join()
        postprocess_thread.join()

        try:
            progress_queue.put(None, timeout=1)
        except Exception:
            pass
        forwarder.join(timeout=5)

    # -- public API --------------------------------------------------------- #

    def _make_exec(
        self,
        reporter: ProgressReporter | None,
        controller: ProgressController | None,
    ) -> _JobExecution:
        ctrl = controller or ProgressController()
        return _JobExecution(
            reporter=reporter or self.reporter,
            controller=ctrl,
            context=_ExecutorNodeContext(ctrl, self.settings_parser, Path()),
        )

    def upscale_file(
        self,
        input_file_path: str,
        output_folder_path: str,
        output_filename: str,
        overwrite_existing_files: bool,
        image_format: str,
        lossy_compression_quality: int,
        use_lossless_compression: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
        reporter: ProgressReporter | None = None,
        controller: ProgressController | None = None,
    ) -> JobResult:
        exec = self._make_exec(reporter, controller)
        input_file_base = Path(input_file_path).stem

        if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):
            output_file_path = str(
                Path(
                    f"{os.path.join(output_folder_path, output_filename.replace('%filename%', input_file_base))}.cbz"
                )
            )
            exec.reporter.log(f"output_file_path {output_file_path}")
            if not overwrite_existing_files and os.path.isfile(output_file_path):
                exec.reporter.log(f"file exists, skip: {output_file_path}")
                exec.result.add(
                    {
                        "input": input_file_path,
                        "output": output_file_path,
                        "status": "skipped",
                    }
                )
                return exec.result

            self.upscale_archive_file(
                exec,
                input_file_path,
                output_file_path,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            )

        elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):
            output_file_path = str(
                Path(
                    f"{os.path.join(output_folder_path, output_filename.replace('%filename%', input_file_base))}.{image_format}"
                )
            )
            if not overwrite_existing_files and os.path.isfile(output_file_path):
                exec.reporter.log(f"file exists, skip: {output_file_path}")
                exec.result.add(
                    {
                        "input": input_file_path,
                        "output": output_file_path,
                        "status": "skipped",
                    }
                )
                return exec.result

            self.upscale_image_file(
                exec,
                input_file_path,
                output_file_path,
                overwrite_existing_files,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            )

        return exec.result

    def upscale_folder(
        self,
        input_folder_path: str,
        output_folder_path: str,
        output_filename: str,
        upscale_images: bool,
        upscale_archives: bool,
        overwrite_existing_files: bool,
        image_format: str,
        lossy_compression_quality: int,
        use_lossless_compression: bool,
        target_scale: float | None,
        target_width: int,
        target_height: int,
        chains: list[dict[str, Any]],
        grayscale_detection_threshold: int,
        reporter: ProgressReporter | None = None,
        controller: ProgressController | None = None,
    ) -> JobResult:
        exec = self._make_exec(reporter, controller)

        upscale_queue: Queue = Queue(maxsize=1)
        postprocess_queue: Queue = Queue(maxsize=1)
        progress_queue: Queue = Queue()

        forwarder = Thread(
            target=self._forward_progress,
            args=(progress_queue, exec.reporter, exec.result),
            daemon=True,
        )
        forwarder.start()

        preprocess_process = Thread(
            target=self._preprocess_worker_folder,
            args=(
                exec,
                upscale_queue,
                input_folder_path,
                output_folder_path,
                output_filename,
                upscale_images,
                upscale_archives,
                overwrite_existing_files,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            ),
        )
        preprocess_process.start()

        upscale_process = Thread(
            target=self._upscale_worker, args=(exec, upscale_queue, postprocess_queue)
        )
        upscale_process.start()

        postprocess_thread = Thread(
            target=_postprocess_worker_folder,
            args=(
                postprocess_queue,
                progress_queue,
                output_folder_path,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
            ),
        )
        postprocess_thread.start()

        preprocess_process.join()
        upscale_process.join()
        postprocess_thread.join()

        try:
            progress_queue.put(None, timeout=1)
        except Exception:
            pass
        forwarder.join(timeout=5)

        return exec.result


def resolve_workflow_params(
    workflow: dict[str, Any],
) -> tuple[str, float | None, int, int, int]:
    """Derive the image format and target size from a workflow dict."""
    if workflow["WebpSelected"]:
        image_format = "webp"
    elif workflow["PngSelected"]:
        image_format = "png"
    elif workflow["AvifSelected"]:
        image_format = "avif"
    else:
        image_format = "jpeg"

    target_scale: float | None = None
    target_width = 0
    target_height = 0

    grayscale_detection_threshold = workflow["GrayscaleDetectionThreshold"]

    if workflow["ModeScaleSelected"]:
        target_scale = workflow["UpscaleScaleFactor"]
    elif workflow["ModeWidthSelected"]:
        target_width = workflow["ResizeWidthAfterUpscale"]
    elif workflow["ModeHeightSelected"]:
        target_height = workflow["ResizeHeightAfterUpscale"]
    else:
        target_width = workflow["DisplayDeviceWidth"]
        target_height = workflow["DisplayDeviceHeight"]

    return (
        image_format,
        target_scale,
        target_width,
        target_height,
        grayscale_detection_threshold,
    )
