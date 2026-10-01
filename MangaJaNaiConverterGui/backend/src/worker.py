"""
Long-running, single-client upscale worker.

The worker speaks newline-delimited JSON (NDJSON) over stdin/stdout:

* requests (one JSON object per line) come in on **stdin**;
* events (one JSON object per line) go out on **stdout**;
* human-readable logs go to **stderr**.

This makes it trivial to drive from any language: the parent process spawns the
worker, writes ``{"type": "job", ...}`` lines to its stdin and reads ``ready`` /
``progress`` / ``done`` events from its stdout.  The worker keeps the
``UpscaleEngine`` (and therefore the model cache and GPU) alive across jobs, so
consecutive jobs stay warm exactly like a bulk chapter run.

Chapters can also be streamed page by page instead of as one archive: an
``open_chapter`` request is followed by ``page`` messages and a
``close_chapter``, and the worker emits a ``page_done`` event for every page as
soon as it is written.  See ``worker_protocol.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
from typing import Any

import torch

# Let the CUDA caching allocator return freed blocks to the driver more readily,
# so releasing the cache while idle actually lowers the resident VRAM footprint.
# Must be set before torch initializes its allocator; an explicit user override
# is respected.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from accelerator_detection import AcceleratorType, get_accelerator_detector  # noqa: E402
from nodes.impl.pytorch.utils import safe_accelerator_cache_empty  # noqa: E402
from progress_controller import ProgressController  # noqa: E402
from upscale_engine import (  # noqa: E402
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PAGE_SENTINEL,
    ProgressReporter,
    UpscaleEngine,
    resolve_workflow_params,
)


def _detect_kind(path: str) -> str:
    lower = path.lower()
    if lower.endswith(ARCHIVE_EXTENSIONS):
        return "archive"
    if lower.endswith(IMAGE_EXTENSIONS):
        return "file"
    if os.path.isdir(path):
        return "folder"
    return "file"


def resolve_job(job: dict[str, Any], base_workflow: dict[str, Any]) -> dict[str, Any]:
    """Normalize a job request into the engine call parameters.

    ``base_workflow`` is the default workflow (from the settings file) that
    provides defaults for anything the job does not override.
    """
    if "workflow" in job:
        wf = job["workflow"]
        if wf.get("SelectedTabIndex") == 1:
            kind, path = "folder", wf["InputFolderPath"]
        else:
            kind, path = "file", wf.get("InputFilePath")
        image_format, ts, tw, th, gdt = resolve_workflow_params(wf)
        return {
            "kind": kind,
            "path": path,
            "output_folder": wf["OutputFolderPath"],
            "output_filename": wf.get("OutputFilename", "%filename%"),
            "overwrite": wf.get("OverwriteExistingFiles", False),
            "image_format": image_format,
            "quality": wf.get("LossyCompressionQuality", 80),
            "lossless": wf.get("UseLosslessCompression", False),
            "target_scale": ts,
            "target_width": tw,
            "target_height": th,
            "chains": wf["Chains"]["$values"],
            "grayscale_threshold": wf.get("GrayscaleDetectionThreshold", 12),
            "upscale_images": wf.get("UpscaleImages", True),
            "upscale_archives": wf.get("UpscaleArchives", True),
        }

    inp = job.get("input") or {}
    path = inp.get("path") or inp.get("input")
    if not path:
        raise ValueError("job.input.path is required")
    kind = inp.get("kind") or _detect_kind(path)

    params = _resolve_output_params(job, base_workflow)
    params["kind"] = kind
    params["path"] = path
    params["upscale_images"] = base_workflow.get("UpscaleImages", True)
    params["upscale_archives"] = base_workflow.get("UpscaleArchives", True)
    return params


def _resolve_output_params(
    job: dict[str, Any], base_workflow: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the output/options half of a job, shared by jobs and chapters."""
    out = job.get("output") or {}
    opts = job.get("options") or {}

    base_format, base_scale, base_width, base_height, _base_gdt = (
        resolve_workflow_params(base_workflow)
    )

    target_scale = base_scale
    target_width = base_width
    target_height = base_height
    if opts.get("scale") is not None:
        target_scale = float(opts["scale"])
        target_width = 0
        target_height = 0
    else:
        if opts.get("width") is not None:
            target_scale = None
            target_width = int(opts["width"])
        if opts.get("height") is not None:
            target_scale = None
            target_height = int(opts["height"])

    return {
        "output_folder": out.get("folder")
        or base_workflow.get("OutputFolderPath", "."),
        "output_filename": out.get(
            "filename", base_workflow.get("OutputFilename", "%filename%")
        ),
        "overwrite": out.get(
            "overwrite", base_workflow.get("OverwriteExistingFiles", False)
        ),
        "image_format": out.get("format") or base_format,
        "quality": out.get("quality", base_workflow.get("LossyCompressionQuality", 80)),
        "lossless": out.get(
            "lossless", base_workflow.get("UseLosslessCompression", False)
        ),
        "target_scale": target_scale,
        "target_width": target_width,
        "target_height": target_height,
        "chains": job.get("chains") or base_workflow["Chains"]["$values"],
        "grayscale_threshold": job.get(
            "grayscale_detection_threshold",
            base_workflow.get("GrayscaleDetectionThreshold", 12),
        ),
    }


def resolve_chapter_job(
    job: dict[str, Any], base_workflow: dict[str, Any]
) -> dict[str, Any]:
    """Resolve an ``open_chapter`` request into engine parameters.

    A chapter has no single input path: pages arrive later through ``page``
    messages, so only the output/options and the expected page count are resolved.
    """
    params = _resolve_output_params(job, base_workflow)
    params["total_pages"] = int(job.get("total_pages") or 0)
    return params


class _JobReporter(ProgressReporter):
    """Routes engine progress for one job into NDJSON events."""

    def __init__(self, emit, job_id: str) -> None:
        self._emit = emit
        self._id = job_id
        self._lock = threading.Lock()
        self._completed = 0
        self._archive_total: int | None = None
        self._archive_completed = 0

    def log(self, message: str) -> None:
        sys.stderr.write(f"[{self._id}] {message}\n")
        sys.stderr.flush()

    def archive_total(self, total: int) -> None:
        with self._lock:
            self._archive_total = total
            self._archive_completed = 0
            completed = self._completed
        self._emit(
            {
                "type": "progress",
                "id": self._id,
                "completed": completed,
                "archive_total": total,
                "archive_completed": 0,
            }
        )

    def file_completed(self, kind: str) -> None:
        with self._lock:
            self._completed += 1
            if kind in ("postprocess_worker_zip_image", "postprocess_worker_page"):
                self._archive_completed += 1
            elif kind == "postprocess_worker_zip_archive":
                if self._archive_total is not None:
                    self._archive_completed = self._archive_total
            completed = self._completed
            archive_total = self._archive_total
            archive_completed = self._archive_completed
        event: dict[str, Any] = {
            "type": "progress",
            "id": self._id,
            "completed": completed,
        }
        if archive_total is not None:
            event["archive_total"] = archive_total
            event["archive_completed"] = archive_completed
        self._emit(event)

    def phase(self, name: str) -> None:
        """Emit a named phase transition (e.g. 'finalizing') without touching counters."""
        with self._lock:
            completed = self._completed
            archive_total = self._archive_total
            archive_completed = self._archive_completed
        event: dict[str, Any] = {
            "type": "progress",
            "id": self._id,
            "completed": completed,
            "phase": name,
        }
        if archive_total is not None:
            event["archive_total"] = archive_total
            event["archive_completed"] = archive_completed
        self._emit(event)


class Worker:
    def __init__(
        self,
        settings: dict[str, Any],
        queue_capacity: int,
        warmup: bool,
        cache_release_idle: float = 0.0,
    ) -> None:
        self.settings = settings
        self.queue_capacity = max(1, int(queue_capacity))
        # Seconds of idleness after which cached VRAM is returned to the driver
        # (0 disables). See _release_accelerator_cache.
        self.cache_release_idle = max(0.0, float(cache_release_idle))
        self._lock = threading.Lock()
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._outstanding = 0
        self._cancelled: set[str] = set()
        self._current_id: str | None = None
        self._current_controller: ProgressController | None = None
        self._shutdown = False
        # Active chapter streams keyed by job id. The reader thread routes
        # ``page``/``close_chapter`` messages into these queues while the chapter
        # job runs on the job loop.
        self._chapter_inputs: dict[str, queue.Queue[Any]] = {}
        self._chapter_indices: dict[str, dict[str, int]] = {}

        self.engine = UpscaleEngine(settings)

        base = settings["Workflows"]["$values"][
            settings.get("SelectedWorkflowIndex", 0)
        ]
        self.base_workflow = base

        if warmup:
            self.engine.warmup(base["Chains"]["$values"])

        self.device_info: dict[str, Any] = {
            "selected_device_index": settings.get("SelectedDeviceIndex"),
            "use_cpu": settings.get("SelectedDeviceIndex") == 0,
            "use_fp16": settings.get("UseFp16", False),
            "models_directory": settings.get("ModelsDirectory"),
        }

    # -- stdout ------------------------------------------------------------- #

    def _write(self, event: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._write(event)

    def _ready(self) -> None:
        with self._lock:
            capacity = self.queue_capacity - self._outstanding
            self._write(
                {"type": "ready", "capacity": capacity, "device": self.device_info}
            )

    # -- accelerator cache release ------------------------------------------ #

    def _accelerator_torch_device(self) -> torch.device | None:
        """Resolve the torch device the engine runs inference on.

        Mirrors ``PyTorchSettings.device`` in
        ``packages/chaiNNer_pytorch/settings.py`` so we release the cache of the
        same device the engine actually uses. Returns ``None`` for CPU mode.
        """
        index = self.settings.get("SelectedDeviceIndex", 0)
        index = int(index) if isinstance(index, (int, float)) else 0
        if index <= 0:
            return None  # CPU mode: nothing cached on an accelerator

        gpu_devices = [
            device
            for device in get_accelerator_detector().available_devices
            if device.type != AcceleratorType.CPU
        ]
        if gpu_devices and 0 <= index < len(gpu_devices):
            return gpu_devices[index].torch_device

        best = get_accelerator_detector().get_best_device(prefer_gpu=True)
        if best is not None and best.type != AcceleratorType.CPU:
            return best.torch_device
        return None

    def _release_accelerator_cache(self) -> None:
        """Return cached allocator blocks to the driver while idle.

        The PyTorch caching allocator retains freed blocks up to the run's
        high-water mark, which can starve co-tenant GPU processes (e.g. the
        manga-vert-split-nn detection model in MangaIngestWithUpscaling).
        Releasing the cache keeps the engine, models and CUDA context warm while
        shrinking the idle VRAM footprint to roughly weights + context.
        """
        try:
            device = self._accelerator_torch_device()
            if device is None:
                gc.collect()
                return
            safe_accelerator_cache_empty(device)
        except Exception as e:  # noqa: BLE001 -- best-effort, never fail a job over this
            sys.stderr.write(f"accelerator cache release failed: {e}\n")

    def _handle_release_cache(self) -> None:
        with self._lock:
            busy = self._current_id is not None
        if busy:
            # Freeing cached blocks mid-job would only slow down the running
            # inference; the host needs this while we are idle.
            self.emit({"type": "cache_released", "status": "busy"})
            return
        self._release_accelerator_cache()
        self.emit({"type": "cache_released", "status": "ok"})

    # -- request handling --------------------------------------------------- #

    def _accept(self, msg: dict[str, Any]) -> None:
        job_id = msg.get("id")
        with self._lock:
            if self._outstanding >= self.queue_capacity:
                self._write({"type": "rejected", "id": job_id, "reason": "queue_full"})
                return
            self._outstanding += 1
            capacity = self.queue_capacity - self._outstanding
        self._job_queue.put(msg)
        self.emit({"type": "accepted", "id": job_id, "capacity": capacity})

    def _accept_chapter(self, msg: dict[str, Any]) -> None:
        """Accept an ``open_chapter`` job and create its page input queue.

        The queue must exist before the first ``page`` message arrives, so it is
        registered here rather than when the job loop starts the chapter.
        """
        job_id = msg.get("id")
        with self._lock:
            if self._outstanding >= self.queue_capacity:
                self._write({"type": "rejected", "id": job_id, "reason": "queue_full"})
                return
            self._outstanding += 1
            capacity = self.queue_capacity - self._outstanding
            self._chapter_inputs[job_id] = queue.Queue()
            self._chapter_indices[job_id] = {}
        self._job_queue.put(msg)
        self.emit({"type": "accepted", "id": job_id, "capacity": capacity})

    def _handle_page(self, msg: dict[str, Any]) -> None:
        job_id = msg.get("id")
        with self._lock:
            page_queue = self._chapter_inputs.get(job_id)
            if page_queue is not None:
                self._chapter_indices[job_id][msg.get("name")] = msg.get("index")
        if page_queue is None:
            self.emit(
                {
                    "type": "error",
                    "id": job_id,
                    "message": "no active chapter for this page",
                }
            )
            return
        page_queue.put((msg.get("name"), msg.get("path")))

    def _handle_close_chapter(self, msg: dict[str, Any]) -> None:
        job_id = msg.get("id")
        with self._lock:
            page_queue = self._chapter_inputs.get(job_id)
        if page_queue is None:
            self.emit(
                {"type": "error", "id": job_id, "message": "no active chapter to close"}
            )
            return
        page_queue.put(PAGE_SENTINEL)

    def _signal_active_chapters(self) -> None:
        """Unblock any in-flight chapter stream when the reader stops.

        On shutdown/EOF no further ``close_chapter`` can arrive, so without this the
        job loop would block forever on the chapter's page queue. The host's kill
        masks it, but a bare worker (or a crashed host) would leak the process.
        """
        with self._lock:
            queues = list(self._chapter_inputs.values())
            controller = self._current_controller
        if controller is not None:
            controller.abort()
        for page_queue in queues:
            page_queue.put(PAGE_SENTINEL)

    def _cancel(self, job_id: str | None) -> None:
        if job_id is None:
            self.emit({"type": "error", "id": None, "message": "cancel requires an id"})
            return
        with self._lock:
            self._cancelled.add(job_id)
            controller = (
                self._current_controller if self._current_id == job_id else None
            )
        if controller is not None:
            controller.abort()
        self.emit({"type": "cancelled", "id": job_id})

    # -- job execution ------------------------------------------------------ #

    def _resolve_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return resolve_job(job, self.base_workflow)

    def _execute(
        self,
        params: dict[str, Any],
        reporter: ProgressReporter,
        controller: ProgressController,
    ):
        common = (
            params["output_folder"],
            params["output_filename"],
            params["overwrite"],
            params["image_format"],
            params["quality"],
            params["lossless"],
            params["target_scale"],
            params["target_width"],
            params["target_height"],
            params["chains"],
            params["grayscale_threshold"],
        )
        if params["kind"] == "folder":
            return self.engine.upscale_folder(
                params["path"],
                common[0],
                common[1],
                params["upscale_images"],
                params["upscale_archives"],
                common[2],
                common[3],
                common[4],
                common[5],
                common[6],
                common[7],
                common[8],
                common[9],
                common[10],
                reporter=reporter,
                controller=controller,
            )
        return self.engine.upscale_file(
            params["path"],
            common[0],
            common[1],
            common[2],
            common[3],
            common[4],
            common[5],
            common[6],
            common[7],
            common[8],
            common[9],
            common[10],
            reporter=reporter,
            controller=controller,
        )

    def _run_job(self, job: dict[str, Any]) -> None:
        if job.get("type") == "open_chapter":
            self._run_chapter_job(job)
            return

        job_id = job.get("id")

        try:
            params = self._resolve_job(job)
        except Exception as e:
            self.emit({"type": "error", "id": job_id, "message": f"invalid job: {e}"})
            return

        with self._lock:
            cancelled = job_id in self._cancelled
        if cancelled:
            self._cancelled.discard(job_id)
            self.emit(
                {"type": "done", "id": job_id, "status": "cancelled", "files": []}
            )
            return

        controller = ProgressController()
        reporter = _JobReporter(self.emit, job_id)

        with self._lock:
            self._current_id = job_id
            self._current_controller = controller

        self.emit({"type": "started", "id": job_id})
        start = time.monotonic()
        try:
            result = self._execute(params, reporter, controller)
            status = "cancelled" if controller.aborted else "ok"
            self.emit(
                {
                    "type": "done",
                    "id": job_id,
                    "status": status,
                    "files": result.files,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                }
            )
        except Exception as e:
            self.emit({"type": "error", "id": job_id, "message": str(e)})
        finally:
            with self._lock:
                if self._current_id == job_id:
                    self._current_id = None
                    self._current_controller = None
            self._cancelled.discard(job_id)

    def _run_chapter_job(self, job: dict[str, Any]) -> None:
        """Run a streaming chapter: pages arrive through the job's input queue.

        Pages are emitted as ``page_done`` events as soon as each one is on disk
        (so the driver can stream it back immediately); the final ``done`` event
        carries the complete ``files`` list, exactly like a normal job.
        """
        job_id = job.get("id")

        try:
            params = resolve_chapter_job(job, self.base_workflow)
        except Exception as e:
            self._cleanup_chapter(job_id)
            self.emit(
                {"type": "error", "id": job_id, "message": f"invalid chapter: {e}"}
            )
            return

        with self._lock:
            cancelled = job_id in self._cancelled
            page_queue = self._chapter_inputs.get(job_id)
        if cancelled:
            self._cancelled.discard(job_id)
            self._cleanup_chapter(job_id)
            self.emit(
                {"type": "done", "id": job_id, "status": "cancelled", "files": []}
            )
            return
        if page_queue is None:
            self.emit(
                {
                    "type": "error",
                    "id": job_id,
                    "message": "chapter input queue missing",
                }
            )
            return

        controller = ProgressController()
        reporter = _JobReporter(self.emit, job_id)

        with self._lock:
            self._current_id = job_id
            self._current_controller = controller

        self.emit({"type": "started", "id": job_id})
        start = time.monotonic()

        def on_page_done(result: dict[str, Any]) -> None:
            with self._lock:
                index = self._chapter_indices.get(job_id, {}).get(result.get("input"))
            self.emit({"type": "page_done", "id": job_id, "index": index, **result})

        try:
            if params["total_pages"]:
                reporter.archive_total(params["total_pages"])
            result = self.engine.upscale_pages(
                page_queue,
                params["output_folder"],
                params["image_format"],
                params["quality"],
                params["lossless"],
                params["target_scale"],
                params["target_width"],
                params["target_height"],
                params["chains"],
                params["grayscale_threshold"],
                reporter=reporter,
                controller=controller,
                on_page_done=on_page_done,
            )
            status = "cancelled" if controller.aborted else "ok"
            self.emit(
                {
                    "type": "done",
                    "id": job_id,
                    "status": status,
                    "files": result.files,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                }
            )
        except Exception as e:
            self.emit({"type": "error", "id": job_id, "message": str(e)})
        finally:
            with self._lock:
                if self._current_id == job_id:
                    self._current_id = None
                    self._current_controller = None
            self._cancelled.discard(job_id)
            self._cleanup_chapter(job_id)

    def _cleanup_chapter(self, job_id: str | None) -> None:
        with self._lock:
            self._chapter_inputs.pop(job_id, None)
            self._chapter_indices.pop(job_id, None)

    def _job_loop(self) -> None:
        poll = 0.5
        idle_since: float | None = None
        while True:
            try:
                job = self._job_queue.get(timeout=poll)
                idle_since = None
            except queue.Empty:
                if self._shutdown:
                    break
                # While idle, give cached VRAM back to the driver so co-tenant
                # processes can use the GPU; the next job re-populates the cache.
                if self.cache_release_idle > 0 and self._current_id is None:
                    now = time.monotonic()
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= self.cache_release_idle:
                        self._release_accelerator_cache()
                        idle_since = None
                continue
            if job is None:
                break
            self._run_job(job)
            with self._lock:
                self._outstanding -= 1
            self._ready()

    # -- main loop ---------------------------------------------------------- #

    def _stdin_loop(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                self.emit(
                    {"type": "error", "id": None, "message": f"invalid JSON: {e}"}
                )
                continue
            if not isinstance(msg, dict):
                self.emit(
                    {
                        "type": "error",
                        "id": None,
                        "message": "request must be an object",
                    }
                )
                continue

            mtype = msg.get("type")
            if mtype == "job":
                self._accept(msg)
            elif mtype == "open_chapter":
                self._accept_chapter(msg)
            elif mtype == "page":
                self._handle_page(msg)
            elif mtype == "close_chapter":
                self._handle_close_chapter(msg)
            elif mtype == "preload":
                chains = msg.get("chains") or self.base_workflow["Chains"]["$values"]
                loaded = self.engine.warmup(chains)
                self.emit({"type": "preloaded", "loaded": loaded})
            elif mtype == "cancel":
                self._cancel(msg.get("id"))
            elif mtype == "release_cache":
                self._handle_release_cache()
            elif mtype == "shutdown":
                self._shutdown = True
                break
            elif mtype == "ping":
                self.emit({"type": "pong"})
            else:
                self.emit(
                    {
                        "type": "error",
                        "id": msg.get("id"),
                        "message": f"unknown request type: {mtype}",
                    }
                )

        self._shutdown = True
        self._signal_active_chapters()
        self._job_queue.put(None)

    def run(self) -> None:
        # The job loop runs on the *main* thread; each job's preprocess, upscale
        # and postprocess stages run on their own threads. The reader thread only
        # dispatches requests, so a chapter can receive pages while it runs.
        self._ready()
        reader = threading.Thread(target=self._stdin_loop, daemon=True)
        reader.start()

        self._job_loop()

        reader.join(timeout=5)
        self.emit({"type": "exited"})


def _load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.settings:
        with open(args.settings, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        default_file = os.path.join("..", "resources", "default_cli_configuration.json")
        with open(default_file) as f:
            settings = json.load(f)
        settings["SelectedDeviceIndex"] = (
            int(args.device_index) if args.device_index is not None else 0
        )
        settings["ModelsDirectory"] = args.models_directory_path or os.path.join(
            "..", "models"
        )
        wf = settings["Workflows"]["$values"][0]
        wf["OutputFolderPath"] = args.output_folder_path
        wf["UpscaleScaleFactor"] = args.upscale_factor

    if args.models_directory_path:
        settings["ModelsDirectory"] = args.models_directory_path
    if args.use_cpu:
        settings["SelectedDeviceIndex"] = 0
        settings["UseFp16"] = False
    elif args.device_index is not None:
        settings["SelectedDeviceIndex"] = int(args.device_index)
    if args.use_fp16 is not None:
        settings["UseFp16"] = args.use_fp16

    return settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python worker.py",
        description="Long-running, single-client upscale worker speaking NDJSON over stdin/stdout.",
    )
    parser.add_argument(
        "--settings", help="Path to an appstate2.json-style settings file."
    )
    parser.add_argument(
        "-m",
        "--models-directory-path",
        default=None,
        help="Directory with models used for upscaling.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Device used to run upscaling jobs (0 = CPU, 1 = first GPU).",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode.")
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable FP16 mode.",
    )
    parser.add_argument(
        "-u",
        "--upscale-factor",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Used when no settings file is given. Default: 2",
    )
    parser.add_argument(
        "-o",
        "--output-folder-path",
        default=os.path.join(".", "out"),
        help="Default output directory (only used when no settings file is given).",
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=1,
        help="Max number of in-flight + queued jobs. Default: 1",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Preload all chain models before emitting the first 'ready' event.",
    )
    parser.add_argument(
        "--cache-release-idle",
        type=float,
        default=0.0,
        help=(
            "Seconds of idleness after which cached VRAM is returned to the driver "
            "(0 disables). Keeps the engine warm while shrinking the idle VRAM "
            "footprint so co-tenant GPU processes can run. Default: 0"
        ),
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore

    settings = _load_settings(args)
    Worker(settings, args.queue_capacity, args.warmup, args.cache_release_idle).run()


if __name__ == "__main__":
    main()
