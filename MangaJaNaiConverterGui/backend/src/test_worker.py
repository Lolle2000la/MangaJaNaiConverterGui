import json
import os
import queue
import subprocess
import sys
import threading
import zipfile
from io import BytesIO

import numpy as np
import pytest
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
        {"id": "j", "input": {"path": "/in/a.png"}, "output": {"folder": "/custom", "format": "webp"}},
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
    params = worker_mod.resolve_job({"id": "j", "workflow": wf}, make_workflow("/base/out"))
    assert params["kind"] == "folder"
    assert params["path"] == "/in/folder"
    assert params["output_folder"] == "/wf/out"


def test_resolve_job_missing_path_raises():
    with pytest.raises(ValueError):
        worker_mod.resolve_job({"id": "j", "input": {}}, make_workflow("/out"))


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
        for name in ["p1.png", "p2.png"]:
            buf = BytesIO()
            Image.fromarray(
                (np.random.rand(16, 16, 3) * 255).astype(np.uint8), "RGB"
            ).save(buf, "PNG")
            z.writestr(name, buf.getvalue())
        z.writestr("notes.txt", b"hello")

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

    # archive (images upscaled, non-image copied)
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
    assert statuses == {"p1.png": "upscaled", "p2.png": "upscaled", "notes.txt": "copied"}
    assert (tmp_path / "o3" / "chap.cbz").is_file()

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
