# Worker protocol

`worker.py` is a long-running, single-client upscale worker. It is designed to
be spawned by another process (in any language) and then fed upscaling jobs
while it keeps its PyTorch models loaded and the GPU warm — exactly like a bulk
chapter run through the GUI or the CLI.

The protocol is **newline-delimited JSON (NDJSON)**:

* **stdin**  — one request per line (JSON object).
* **stdout** — one event per line (JSON object), always UTF-8 and flushed.
* **stderr** — human-readable logs only.

There is exactly one client: the process that spawned the worker. The worker
never opens a network port.

---

## Running the worker

```bash
# from MangaJaNaiConverterGui/backend/src/
python worker.py --settings /path/to/appstate2.json --queue-capacity 2

# or, without a settings file (uses resources/default_cli_configuration.json):
python worker.py -m ../models --device-index 1 -u 2 -o ./out
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--settings PATH` | — | `appstate2.json`-style settings file (device, FP16, models dir, default workflow/chains). |
| `-m/--models-directory-path` | `../models` | Models directory. |
| `--device-index N` | `0` | Device (0 = CPU, 1 = first GPU). |
| `--use-cpu` | off | Force CPU. |
| `--use-fp16` / `--no-fp16` | — | Override FP16. |
| `-u/--upscale-factor` | `2` | Default scale (only when no settings file). |
| `-o/--output-folder-path` | `./out` | Default output dir (only when no settings file). |
| `--queue-capacity N` | `1` | Max number of in-flight + queued jobs. |
| `--warmup` | off | Preload all chain models before the first `ready`. |

Even without `--warmup`, models are loaded lazily on first use and cached in
the engine, so consecutive jobs that use the same models stay warm.

---

## Lifecycle

1. The worker initializes PyTorch/device detection and (optionally) warms up
   models.
2. It writes a `ready` event and starts accepting jobs.
3. For each accepted job it writes `accepted`, `started`, zero or more
   `progress` events, then a `done` or `error` event.
4. After every job it writes a fresh `ready` event carrying the updated free
   capacity.
5. On `shutdown` or end-of-stdin it finishes pending jobs, writes `exited`, and
   terminates.

---

## Requests (stdin → worker)

### `job`

```json
{"type": "job", "id": "chap-01", "input": {"path": "/data/ch1.cbz"}, "output": {"folder": "/out", "format": "webp"}}
```

A job may be described two ways.

**Simple form** (merged with the default workflow from the settings file):

| Field | Meaning |
| --- | --- |
| `id` | Required, unique, arbitrary string echoed on every related event. |
| `input.path` | Required. File / folder / archive path. |
| `input.kind` | Optional. `file`, `folder`, or `archive`. Auto-detected from the extension / `isdir()` when omitted. |
| `output.folder` | Output directory. Defaults to the workflow's `OutputFolderPath`. |
| `output.filename` | Output name template (`%filename%` keeps the source name). |
| `output.format` | `webp` \| `png` \| `jpeg` \| `avif`. Defaults to the workflow selection. |
| `output.quality` | Lossy quality (0-100). |
| `output.lossless` | WebP lossless flag. |
| `output.overwrite` | Overwrite existing outputs. |
| `options.scale` | Target scale factor (1/2/3/4). Overrides width/height. |
| `options.width` / `options.height` | Target dimension (fit). |
| `chains` | Optional override of the workflow's chain list. |
| `grayscale_detection_threshold` | Optional override. |

**Workflow form** — pass a full `appstate2.json` workflow object verbatim:

```json
{"type": "job", "id": "chap-01", "workflow": {"SelectedTabIndex": 0, "InputFilePath": "/data/ch1.cbz", "OutputFolderPath": "/out", "...": "..."}}
```

### `cancel`

```json
{"type": "cancel", "id": "chap-01"}
```

Cancels the queued or in-flight job `id`. A queued job is dropped; an in-flight
job is aborted as soon as the current tile/file finishes. The worker replies
with a `cancelled` event immediately and a `done` event with
`"status": "cancelled"` when the job slot is released.

### `shutdown`

```json
{"type": "shutdown"}
```

Stops accepting new jobs, finishes already-queued and in-flight jobs, then
exits. Closing stdin has the same effect.

### `ping`

```json
{"type": "ping"}
```

Replies with `{"type": "pong"}` (useful as a health/liveness check).

---

## Events (worker → stdout)

Every event is a single-line JSON object.

### `ready`

```json
{"type": "ready", "capacity": 2, "device": {"selected_device_index": 1, "use_cpu": false, "use_fp16": true, "models_directory": "/models"}}
```

Emitted at startup and after every job completes. `capacity` is the number of
jobs the worker can still accept right now. The parent may send up to that many
`job` requests without waiting for another `ready`.

### `accepted` / `rejected`

```json
{"type": "accepted", "id": "chap-01", "capacity": 1}
{"type": "rejected", "id": "chap-02", "reason": "queue_full"}
```

`accepted` means the job is queued; `capacity` is the remaining free slots.
`rejected` means the worker is full — retry after the next `ready`.

### `started`

```json
{"type": "started", "id": "chap-01"}
```

The worker began executing this job.

### `progress`

```json
{"type": "progress", "id": "chap-01", "completed": 7}
{"type": "progress", "id": "chap-01", "completed": 7, "archive_total": 42, "archive_completed": 3}
```

`completed` counts finished files for the job. When processing an archive,
`archive_total` is the number of entries in the current archive and
`archive_completed` the number finished so far.

### `done`

```json
{"type": "done", "id": "chap-01", "status": "ok", "elapsed_seconds": 12.34,
 "files": [
   {"input": "/data/ch1.cbz#p1.png", "output": "p1.webp", "status": "upscaled"},
   {"input": "/data/ch1.cbz#notes.txt", "output": "notes.txt", "status": "copied"}
 ]}
```

`status` is `ok` or `cancelled`. `files` is the list of per-file results:

* `status: "upscaled"` — image upscaled and written to `output`.
* `status: "copied"` — non-image entry copied unchanged (archives).
* `status: "skipped"` — output already existed and overwrite was disabled.
* `status: "error"` — this file failed (see `error`).

For archive jobs, `output` is the entry name inside the output `.cbz`; the
archive path itself is known from the request.

### `error`

```json
{"type": "error", "id": "chap-01", "message": "FileNotFoundError: ..."}
```

The job failed (or a request was malformed, in which case `id` may be `null`).

### `cancelled`

```json
{"type": "cancelled", "id": "chap-01"}
```

Acknowledges a `cancel` request.

### `pong`

```json
{"type": "pong"}
```

### `exited`

```json
{"type": "exited"}
```

The worker is about to terminate.

---

## Flow control

The worker enforces a bounded queue. `--queue-capacity N` means at most `N`
jobs may be outstanding (in-flight plus queued) at once.

* The initial `ready` advertises `capacity == N`.
* Every `accepted` advertises the remaining capacity.
* Sending more than `capacity` jobs yields `rejected` for the overflow.
* After each job finishes, a new `ready` advertises the freed capacity.

The simplest, stall-free parent loop is: send one job per `ready`/`accepted`
capacity unit, and never send more jobs than the last advertised capacity.

---

## Non-Python parent example (Go, outline)

```go
cmd := exec.Command("python3", "worker.py", "--settings", "appstate2.json", "--queue-capacity", "2")
stdin, _ := cmd.StdinPipe()
stdout, _ := cmd.StdoutPipe()
cmd.Stderr = os.Stderr
cmd.Start()

scanner := bufio.NewScanner(stdout)          // one JSON event per line
scanner.Scan()                                // first "ready"
var ready struct {
    Type     string `json:"type"`
    Capacity int    `json:"capacity"`
}
json.Unmarshal(scanner.Bytes(), &ready)

job := map[string]any{"type": "job", "id": "j1",
    "input": map[string]any{"path": "/data/ch1.cbz"},
    "output": map[string]any{"folder": "/out", "format": "webp"}}
enc, _ := json.Marshal(job)
fmt.Fprintln(stdin, string(enc))              // send the job

for scanner.Scan() {                          // read events until "done"
    var ev map[string]any
    json.Unmarshal(scanner.Bytes(), &ev)
    if ev["type"] == "done" {
        break
    }
}
fmt.Fprintln(stdin, `{"type":"shutdown"}`)
```

The same pattern works in Node (`child_process.spawn`, readline on stdout),
Python (`subprocess.Popen` + `readline`), or any language with process pipes and
a JSON parser.
