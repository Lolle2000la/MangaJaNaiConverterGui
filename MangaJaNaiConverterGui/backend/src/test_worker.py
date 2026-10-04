import json
import os
import queue
import subprocess
import sys
import threading
import zipfile
from io import BytesIO

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None
from PIL import Image

SRC = os.path.dirname(os.path.abspath(__file__))

import worker as worker_mod  # noqa: E402

NO_MODEL_CHAIN = {
    "MinResolution": "0x0",
    "MaxResolution": "0x0",
    "IsGrayscale": False,
    "IsColor": True,
    "MinScaleFactor": 0,
    "MaxScaleFactor": 0,
    "ModelFilePath": "No Model",
    "ModelTileSize": "Auto (Estimate)",
    "AutoAdjustLevels": False,
    "ResizeWidthBeforeUpscale": 0,
    "ResizeHeightBeforeUpscale": 0,
    "ResizeFactorBeforeUpscale": 100.0,
}


def make_workflow(out_folder: str, chains=None) -> dict:
    return {
        "SelectedTabIndex": 0,
        "GrayscaleDetectionThreshold": 12,
        "InputFilePath": "",
        "InputFolderPath": "",
        "OutputFilename": "%filename%",
        "OutputFolderPath": out_folder,
        "OverwriteExistingFiles": False,
        "UpscaleImages": True,
        "UpscaleArchives": True,
        "LossyCompressionQuality": 80,
        "UseLosslessCompression": False,
        "WebpSelected": False,
        "PngSelected": True,
        "AvifSelected": False,
        "JpegSelected": False,
        "ModeScaleSelected": True,
        "UpscaleScaleFactor": 2,
        "ModeWidthSelected": False,
        "ModeHeightSelected": False,
        "ModeFitToDisplaySelected": False,
        "DisplayDeviceWidth": 0,
        "DisplayDeviceHeight": 0,
        "ResizeWidthAfterUpscale": 0,
        "ResizeHeightAfterUpscale": 0,
        "Chains": {"$values": chains or [NO_MODEL_CHAIN]},
    }


def make_settings(out_folder: str, models_dir: str) -> dict:
    return {
        "SelectedDeviceIndex": 0,
        "UseFp16": False,
        "ModelsDirectory": models_dir,
        "SelectedWorkflowIndex": 0,
        "Workflows": {"$values": [make_workflow(out_folder)]},
    }


def write_image(path: str, size=(24, 24)) -> None:
    Image.fromarray(
        (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8), "RGB"
    ).save(path)


class WorkerClient:
    def __init__(self, settings_path: str, capacity: str = "1") -> None:
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "worker.py",
                "--settings",
                settings_path,
                "--queue-capacity",
                capacity,
            ],
            cwd=SRC,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            if line:
                self._q.put(line)

    def read(self, timeout: float = 30):
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return json.loads(line)

    def send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def read_until(self, event_type: str, timeout: float = 30):
        while True:
            ev = self.read(timeout)
            if ev is None:
                return None
            if ev.get("type") == event_type:
                return ev

    def shutdown(self) -> None:
        self.send({"type": "shutdown"})
        self.read_until("exited", timeout=30)
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# --------------------------------------------------------------------------- #
# Unit tests (pure, no GPU/models)
# --------------------------------------------------------------------------- #


def test_detect_kind(tmp_path):
    assert worker_mod._detect_kind("/a/b.png") == "file"
    assert worker_mod._detect_kind("/a/b.cbz") == "archive"
    assert worker_mod._detect_kind("/a/b.cbr") == "archive"
    assert worker_mod._detect_kind(str(tmp_path)) == "folder"


def test_resolve_job_simple_merges_defaults():
    base = make_workflow("/base/out")
    params = worker_mod.resolve_job(
        {
            "id": "j",
            "input": {"path": "/in/a.png"},
            "output": {"folder": "/custom", "format": "webp"},
        },
        base,
    )
    assert params["kind"] == "file"
    assert params["path"] == "/in/a.png"
    assert params["output_folder"] == "/custom"
    assert params["image_format"] == "webp"
    assert params["target_scale"] == 2  # inherited from base workflow scale
    assert params["grayscale_threshold"] == 12


def test_resolve_job_workflow_form():
    wf = make_workflow("/wf/out")
    wf["SelectedTabIndex"] = 1
    wf["InputFolderPath"] = "/in/folder"
    params = worker_mod.resolve_job(
        {"id": "j", "workflow": wf}, make_workflow("/base/out")
    )
    assert params["kind"] == "folder"
    assert params["path"] == "/in/folder"
    assert params["output_folder"] == "/wf/out"


def test_resolve_job_missing_path_raises():
    if pytest is not None:
        with pytest.raises(ValueError):
            worker_mod.resolve_job({"id": "j", "input": {}}, make_workflow("/out"))
    else:
        try:
            worker_mod.resolve_job({"id": "j", "input": {}}, make_workflow("/out"))
            assert False, "Expected ValueError"
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# End-to-end subprocess smoke tests (No-Model chain, CPU)
# --------------------------------------------------------------------------- #


def test_worker_end_to_end(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(make_settings(str(out_dir), str(models_dir))))

    inp = tmp_path / "img.png"
    write_image(str(inp))
    folder = tmp_path / "folder_in"
    folder.mkdir()
    write_image(str(folder / "a.png"))
    write_image(str(folder / "b.png"))
    cbz = tmp_path / "chap.cbz"
    with zipfile.ZipFile(str(cbz), "w") as z:
        # Include directory entry to verify directory skipping
        z.writestr("chap/", b"")
        for name in ["chap/p1.png", "chap/p2.png"]:
            buf = BytesIO()
            Image.fromarray(
                (np.random.rand(16, 16, 3) * 255).astype(np.uint8), "RGB"
            ).save(buf, "PNG")
            z.writestr(name, buf.getvalue())
        z.writestr("notes.txt", b"hello")

    cbz_upper = tmp_path / "chap_upper.CBZ"
    with zipfile.ZipFile(str(cbz_upper), "w") as z:
        z.writestr("folder/", b"")
        buf = BytesIO()
        Image.fromarray((np.random.rand(16, 16, 3) * 255).astype(np.uint8), "RGB").save(
            buf, "PNG"
        )
        z.writestr("folder/page.png", buf.getvalue())

    w = WorkerClient(str(settings_path), capacity="3")
    assert w.read()["type"] == "ready"

    # single file
    w.send(
        {
            "type": "job",
            "id": "f1",
            "input": {"path": str(inp)},
            "output": {"folder": str(tmp_path / "o1"), "format": "png"},
        }
    )
    ev = w.read_until("done")
    assert ev["id"] == "f1" and ev["status"] == "ok", ev
    assert (tmp_path / "o1" / "img.png").is_file()

    # folder
    w.send(
        {
            "type": "job",
            "id": "d1",
            "input": {"path": str(folder), "kind": "folder"},
            "output": {"folder": str(tmp_path / "o2"), "format": "png"},
        }
    )
    ev = w.read_until("done")
    assert ev["id"] == "d1" and ev["status"] == "ok", ev
    assert len(ev["files"]) == 2
    assert (tmp_path / "o2" / "a.png").is_file()
    assert (tmp_path / "o2" / "b.png").is_file()

    # archive with dir entry (images upscaled, non-image copied, dir ignored)
    w.send(
        {
            "type": "job",
            "id": "z1",
            "input": {"path": str(cbz), "kind": "archive"},
            "output": {"folder": str(tmp_path / "o3"), "format": "png"},
        }
    )
    ev = w.read_until("done")
    assert ev["id"] == "z1" and ev["status"] == "ok", ev
    statuses = {f["output"]: f["status"] for f in ev["files"]}
    assert statuses == {
        "chap/p1.png": "upscaled",
        "chap/p2.png": "upscaled",
        "notes.txt": "copied",
    }
    assert (tmp_path / "o3" / "chap.cbz").is_file()

    # uppercase archive extension (.CBZ)
    w.send(
        {
            "type": "job",
            "id": "z2",
            "input": {"path": str(cbz_upper), "kind": "archive"},
            "output": {"folder": str(tmp_path / "o4"), "format": "png"},
        }
    )
    ev = w.read_until("done")
    assert ev["id"] == "z2" and ev["status"] == "ok", ev
    assert (tmp_path / "o4" / "chap_upper.cbz").is_file()

    w.shutdown()


def test_worker_release_cache_while_idle(tmp_path):
    """release_cache is acknowledged while the worker is idle."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(make_settings(str(out_dir), str(models_dir))))

    w = WorkerClient(str(settings_path), capacity="1")
    assert w.read()["type"] == "ready"

    # Idle: the release runs and is acknowledged.
    w.send({"type": "release_cache"})
    ev = w.read_until("cache_released")
    assert ev["status"] == "ok", ev

    w.shutdown()


def test_worker_flow_control_and_cancel(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(make_settings(str(out_dir), str(models_dir))))

    # A folder with many images gives a deterministic window for the queue to be
    # full while the first job is still running.
    slow = tmp_path / "slow"
    slow.mkdir()
    for i in range(150):
        write_image(str(slow / f"i{i:03d}.png"))

    w = WorkerClient(str(settings_path), capacity="1")
    assert w.read()["type"] == "ready"

    w.send(
        {
            "type": "job",
            "id": "s1",
            "input": {"path": str(slow), "kind": "folder"},
            "output": {"folder": str(tmp_path / "os"), "format": "png"},
        }
    )
    assert w.read_until("accepted")["id"] == "s1"

    # Second job while s1 is in-flight must be rejected (capacity == 1).
    w.send(
        {
            "type": "job",
            "id": "s2",
            "input": {"path": str(slow), "kind": "folder"},
            "output": {"folder": str(tmp_path / "os2"), "format": "png"},
        }
    )
    rej = w.read_until("rejected")
    assert rej["id"] == "s2" and rej["reason"] == "queue_full"

    # Cancel the in-flight job.
    w.send({"type": "cancel", "id": "s1"})
    done = w.read_until("done")
    assert done["id"] == "s1"
    assert done["status"] == "cancelled"

    w.shutdown()


# --------------------------------------------------------------------------- #
# Streaming chapter jobs
# --------------------------------------------------------------------------- #


def collect_until(client: "WorkerClient", event_type: str, timeout: float = 60):
    events = []
    while True:
        event = client.read(timeout)
        if event is None:
            return events
        events.append(event)
        if event.get("type") == event_type:
            return events


def make_chapter_settings(tmp_path) -> tuple[str, str]:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "chapter_out"
    out_dir.mkdir()
    settings_path = tmp_path / "chapter_settings.json"
    settings_path.write_text(json.dumps(make_settings(str(out_dir), str(models_dir))))
    return str(settings_path), str(out_dir)


def test_resolve_chapter_job_merges_defaults():
    base = make_workflow("/base/out")
    params = worker_mod.resolve_chapter_job(
        {
            "id": "ch",
            "output": {"folder": "/custom", "format": "webp"},
            "total_pages": 7,
        },
        base,
    )
    assert params["output_folder"] == "/custom"
    assert params["image_format"] == "webp"
    assert params["target_scale"] == 2
    assert params["total_pages"] == 7
    assert params["chains"] == base["Chains"]["$values"]


def test_worker_chapter_streams_pages(tmp_path):
    settings_path, out_dir = make_chapter_settings(tmp_path)

    pages = []
    for i in range(3):
        page = tmp_path / f"page_{i:03d}.png"
        write_image(str(page))
        pages.append(page)

    w = WorkerClient(settings_path, capacity="1")
    assert w.read()["type"] == "ready"

    w.send(
        {
            "type": "open_chapter",
            "id": "ch1",
            "output": {"folder": out_dir, "format": "png"},
            "total_pages": len(pages),
        }
    )
    assert w.read_until("accepted")["id"] == "ch1"

    for index, page in enumerate(pages):
        w.send(
            {
                "type": "page",
                "id": "ch1",
                "index": index,
                "name": page.name,
                "path": str(page),
            }
        )
    w.send({"type": "close_chapter", "id": "ch1"})

    events = collect_until(w, "done")
    page_done = [e for e in events if e["type"] == "page_done"]
    assert len(page_done) == 3, events
    assert {e["index"] for e in page_done} == {0, 1, 2}
    assert all(e["status"] == "upscaled" for e in page_done)

    done = events[-1]
    assert done["type"] == "done"
    assert done["id"] == "ch1"
    assert done["status"] == "ok"
    assert len(done["files"]) == 3
    for page in pages:
        assert (tmp_path / "chapter_out" / page.name).is_file()

    w.shutdown()


def test_worker_shutdown_mid_chapter_emits_exited(tmp_path):
    settings_path, out_dir = make_chapter_settings(tmp_path)

    page = tmp_path / "page_000.png"
    write_image(str(page))

    w = WorkerClient(settings_path, capacity="1")
    assert w.read()["type"] == "ready"

    w.send(
        {
            "type": "open_chapter",
            "id": "ch1",
            "output": {"folder": out_dir, "format": "png"},
            "total_pages": 3,
        }
    )
    assert w.read_until("accepted")["id"] == "ch1"

    # Send one page but never close the chapter, then ask the worker to shut down. The
    # reader thread stops, so no close_chapter can arrive; before the fix the job loop
    # blocked on the chapter's page queue forever and no "exited" event was emitted.
    w.send(
        {
            "type": "page",
            "id": "ch1",
            "index": 0,
            "name": page.name,
            "path": str(page),
        }
    )
    w.send({"type": "shutdown"})

    assert w.read_until("exited", timeout=30) is not None
    w.proc.stdin.close()
    try:
        w.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        w.proc.kill()
        raise AssertionError("worker did not exit after shutdown mid-chapter") from None


def test_worker_chapter_reports_bad_page_without_failing(tmp_path):
    settings_path, out_dir = make_chapter_settings(tmp_path)
    good = tmp_path / "good.png"
    write_image(str(good))

    w = WorkerClient(settings_path, capacity="1")
    assert w.read()["type"] == "ready"

    w.send(
        {
            "type": "open_chapter",
            "id": "ch2",
            "output": {"folder": out_dir, "format": "png"},
            "total_pages": 2,
        }
    )
    assert w.read_until("accepted")["id"] == "ch2"
    w.send(
        {"type": "page", "id": "ch2", "index": 0, "name": good.name, "path": str(good)}
    )
    w.send(
        {
            "type": "page",
            "id": "ch2",
            "index": 1,
            "name": "missing.png",
            "path": str(tmp_path / "missing.png"),
        }
    )
    w.send({"type": "close_chapter", "id": "ch2"})

    events = collect_until(w, "done")
    page_done = [e for e in events if e["type"] == "page_done"]
    assert len(page_done) == 2, events
    statuses = {e["index"]: e["status"] for e in page_done}
    assert statuses == {0: "upscaled", 1: "error"}

    done = events[-1]
    assert done["status"] == "ok"
    assert (tmp_path / "chapter_out" / good.name).is_file()

    w.shutdown()


def test_worker_chapter_cancel_before_start(tmp_path):
    settings_path, out_dir = make_chapter_settings(tmp_path)

    w = WorkerClient(settings_path, capacity="1")
    assert w.read()["type"] == "ready"

    w.send(
        {
            "type": "open_chapter",
            "id": "ch3",
            "output": {"folder": out_dir, "format": "png"},
        }
    )
    assert w.read_until("accepted")["id"] == "ch3"
    w.send({"type": "cancel", "id": "ch3"})
    done = w.read_until("done")
    assert done["id"] == "ch3"
    assert done["status"] == "cancelled"

    w.shutdown()


def test_worker_preload_command(tmp_path):
    settings_path, _out_dir = make_chapter_settings(tmp_path)

    w = WorkerClient(settings_path, capacity="1")
    assert w.read()["type"] == "ready"

    # The No-Model chain has nothing to preload, but the command must answer.
    w.send({"type": "preload"})
    preloaded = w.read_until("preloaded")
    assert preloaded["loaded"] == 0

    w.shutdown()


def test_engine_upscale_image_bytes(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    engine = worker_mod.UpscaleEngine(
        make_settings(str(tmp_path / "out"), str(models_dir))
    )

    buffer = BytesIO()
    Image.fromarray((np.random.rand(24, 24, 3) * 255).astype(np.uint8), "RGB").save(
        buffer, "PNG"
    )

    encoded = engine.upscale_image_bytes(
        buffer.getvalue(),
        "png",
        80,
        False,
        2,
        0,
        0,
        [NO_MODEL_CHAIN],
        12,
    )
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"

    with Image.open(BytesIO(encoded)) as result:
        # A 2x scale with no model still resizes the output to 48x48.
        assert result.size == (48, 48)


# --------------------------------------------------------------------------- #
# SelectedDeviceIndex is CPU-inclusive (0 = CPU, 1 = first accelerator) but the
# PyTorchSettings / accelerator list indexes over non-CPU devices only.
# --------------------------------------------------------------------------- #


class _FakeDevice:
    def __init__(self, device_type, torch_device: str) -> None:
        self.type = device_type
        self.torch_device = torch_device


class _FakeDetector:
    def __init__(self, devices: list) -> None:
        self.available_devices = devices

    def get_best_device(self, prefer_gpu: bool = False):
        return self.available_devices[-1]


def test_engine_maps_cpu_inclusive_device_index(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    settings = make_settings(str(tmp_path / "out"), str(models_dir))

    # 1 = first accelerator -> 0-based index 0 over gpu_devices.
    settings["SelectedDeviceIndex"] = 1
    engine = worker_mod.UpscaleEngine(settings)
    assert engine.settings_parser.get_bool("use_cpu", False) is False
    assert engine.settings_parser.get_int("accelerator_device_index", 0) == 0

    # A JSON string must be coerced, not treated as index 0.
    settings["SelectedDeviceIndex"] = "2"
    engine = worker_mod.UpscaleEngine(settings)
    assert engine.settings_parser.get_int("accelerator_device_index", 0) == 1

    # 0 = CPU.
    settings["SelectedDeviceIndex"] = 0
    engine = worker_mod.UpscaleEngine(settings)
    assert engine.settings_parser.get_bool("use_cpu", False) is True


def test_accelerator_torch_device_maps_cpu_inclusive_index(monkeypatch):
    cpu = _FakeDevice(worker_mod.AcceleratorType.CPU, "cpu")
    gpu0 = _FakeDevice(worker_mod.AcceleratorType.CUDA, "cuda:0")
    gpu1 = _FakeDevice(worker_mod.AcceleratorType.CUDA, "cuda:1")
    detector = _FakeDetector([cpu, gpu0, gpu1])
    monkeypatch.setattr(worker_mod, "get_accelerator_detector", lambda: detector)

    class _Stub:
        pass

    stub = _Stub()

    # 0 = CPU mode: no accelerator cache to release.
    stub.settings = {"SelectedDeviceIndex": 0}
    assert worker_mod.Worker._accelerator_torch_device(stub) is None

    # 1 = first non-CPU device, not the second.
    stub.settings = {"SelectedDeviceIndex": 1}
    assert worker_mod.Worker._accelerator_torch_device(stub) == "cuda:0"

    # 2 = second non-CPU device.
    stub.settings = {"SelectedDeviceIndex": 2}
    assert worker_mod.Worker._accelerator_torch_device(stub) == "cuda:1"
