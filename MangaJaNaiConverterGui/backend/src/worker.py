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
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from typing import Any

sys_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from progress_controller import ProgressController  # noqa: E402
from upscale_engine import (  # noqa: E402
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
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
    out = job.get("output") or {}
    opts = job.get("options") or {}

    path = inp.get("path") or inp.get("input")
    if not path:
        raise ValueError("job.input.path is required")
    kind = inp.get("kind") or _detect_kind(path)

    base_format, base_scale, base_width, base_height, base_gdt = (
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
        "kind": kind,
        "path": path,
        "output_folder": out.get("folder") or base_workflow.get("OutputFolderPath", "."),
        "output_filename": out.get("filename", base_workflow.get("OutputFilename", "%filename%")),
        "overwrite": out.get("overwrite", base_workflow.get("OverwriteExistingFiles", False)),
        "image_format": out.get("format") or base_format,
        "quality": out.get("quality", base_workflow.get("LossyCompressionQuality", 80)),
        "lossless": out.get("lossless", base_workflow.get("UseLosslessCompression", False)),
        "target_scale": target_scale,
        "target_width": target_width,
        "target_height": target_height,
        "chains": job.get("chains") or base_workflow["Chains"]["$values"],
        "grayscale_threshold": job.get(
            "grayscale_detection_threshold",
            base_workflow.get("GrayscaleDetectionThreshold", 12),
        ),
        "upscale_images": base_workflow.get("UpscaleImages", True),
        "upscale_archives": base_workflow.get("UpscaleArchives", True),
    }


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
            if kind == "postprocess_worker_zip_image":
                self._archive_completed += 1
            elif kind == "postprocess_worker_zip_archive":
                if self._archive_total is not None:
                    self._archive_completed = self._archive_total
            completed = self._completed
            archive_total = self._archive_total
            archive_completed = self._archive_completed
        event: dict[str, Any] = {"type": "progress", "id": self._id, "completed": completed}
        if archive_total is not None:
            event["archive_total"] = archive_total
            event["archive_completed"] = archive_completed
        self._emit(event)


class Worker:
    def __init__(self, settings: dict[str, Any], queue_capacity: int, warmup: bool) -> None:
        self.queue_capacity = max(1, int(queue_capacity))
        self._lock = threading.Lock()
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._outstanding = 0
        self._cancelled: set[str] = set()
        self._current_id: str | None = None
        self._current_controller: ProgressController | None = None
        self._shutdown = False

        self.engine = UpscaleEngine(settings)

        base = settings["Workflows"]["$values"][settings.get("SelectedWorkflowIndex", 0)]
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
            self._write({"type": "ready", "capacity": capacity, "device": self.device_info})

    # -- request handling --------------------------------------------------- #

    def _accept(self, msg: dict[str, Any]) -> None:
        job_id = msg.get("id")
        with self._lock:
            if self._outstanding >= self.queue_capacity:
                self._write(
                    {"type": "rejected", "id": job_id, "reason": "queue_full"}
                )
                return
            self._outstanding += 1
            capacity = self.queue_capacity - self._outstanding
        self._job_queue.put(msg)
        self.emit({"type": "accepted", "id": job_id, "capacity": capacity})

    def _cancel(self, job_id: str | None) -> None:
        if job_id is None:
            self.emit({"type": "error", "id": None, "message": "cancel requires an id"})
            return
        with self._lock:
            self._cancelled.add(job_id)
            controller = self._current_controller if self._current_id == job_id else None
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
            self.emit({"type": "done", "id": job_id, "status": "cancelled", "files": []})
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

    def _job_loop(self) -> None:
        while True:
            job = self._job_queue.get()
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
                self.emit({"type": "error", "id": None, "message": f"invalid JSON: {e}"})
                continue
            if not isinstance(msg, dict):
                self.emit({"type": "error", "id": None, "message": "request must be an object"})
                continue

            mtype = msg.get("type")
            if mtype == "job":
                self._accept(msg)
            elif mtype == "cancel":
                self._cancel(msg.get("id"))
            elif mtype == "shutdown":
                self._shutdown = True
                break
            elif mtype == "ping":
                self.emit({"type": "pong"})
            else:
                self.emit(
                    {"type": "error", "id": msg.get("id"), "message": f"unknown request type: {mtype}"}
                )

        self._shutdown = True
        self._job_queue.put(None)

    def run(self) -> None:
        # The job loop runs on the *main* thread so that the postprocess
        # subprocess (multiprocessing fork) is always started from the main
        # thread, matching the original CLI/GUI behaviour and avoiding
        # fork-from-a-worker-thread deadlocks.
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
    parser.add_argument("--settings", help="Path to an appstate2.json-style settings file.")
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
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore

    settings = _load_settings(args)
    Worker(settings, args.queue_capacity, args.warmup).run()


if __name__ == "__main__":
    main()
