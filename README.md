# cv.people external runtime example

This directory is an independently installable Python package implementing the
current `cv_demo.runtimes` and `cv_demo.http_apis` contracts. It adapts the existing
people detection and tracking workers without importing the application's
`internal`, `app`, or `pkg` namespaces. Copy this directory into a separate
repository to develop it as an external package. Model weights are not included.

## Setup

From the cv-demo repository root, in the application's Python environment:

```sh
python3 -m pip install -e internal/ai_runtime/runtime_package_example/cv_people
export CV_DEMO_RUNTIME_REGISTRY=internal/ai_runtime/runtime_package_example/cv_people/registry.yaml
```

Start cv-demo normally. This registry enables only the example; the default
`new_registry.yaml` requires the other listed packages too. Install exactly one
distribution exporting `cv.people`. The local registry is for an already installed
package, not the Git deployment installer. For deployment, put the extracted
package's repository URL in the `cv.people` entry of `new_registry.yaml`.

Configure existing weights in the application's configuration:

```json
{
  "model": {
    "person": "/absolute/path/to/person.pt",
    "reid": "/absolute/path/to/osnet.engine"
  }
}
```

Detection uses the installed `person-inference` package (`person_inference.Detector`).
Install its backend extra for your model: `ultralytics` for `.pt`, `onnx` for
`.onnx`, or `tensorrt` for `.trt`. Put a same-stem `.txt` label file beside the
model, with one class per line including `person`.

Optional ReID uses the installed `osnet-reid-trt` package
(`osnet_reid_trt.ReIDTRT`) and a serialized OSNet TensorRT engine, including its
backbone and BN neck. Install that package from your local clone or repository
before installing this runtime with its `reid` extra; it is not published on
PyPI. Without `model.reid`, tracking uses motion/IoU association. PyTorch
checkpoints are no longer accepted for ReID.

See [detection package reference](docs/detection_external.md) and
[OSNet package reference](docs/osnet_external.md) for installation and engine
requirements. These packages must be installed in the application's environment;
the runtime does not bundle their implementations. Models load on first session
creation and are shared by that runtime instance. For TensorRT, create sessions,
process frames, and close the runtime on the same CUDA context-owning thread,
with serialized calls.

## Requests and lifecycle

Send `session.example.json` to `POST /api/v1/runtime` after replacing the source
with an actual configured camera. Change `inference_type` to `detection` for
counting. Explicit selection is also supported:

```json
{"runtime": {"id": "cv.people", "capability": "person.tracking"}}
```

If both selectors are supplied they must agree. The core resolves the selector
and calls `PeopleRuntime.create_session()` with a plain dictionary:

```python
{
    "session_id": "people-example-1",
    "camera_id": "camera-1",
    "capability": "person.tracking",
    "config": {"fps": 5},
    "zones": [],
    "output": {"mode": "periodic", "interval": 5},
}
```

The core calls `process_frame(frame, session_ids)` with a read-only BGR image.
One detector pass serves all submitted sessions; compatible ReID features are
also shared. Each tracking session owns its tracker and analytics. Detection
sessions filter the shared result with their own confidence threshold.
Each submitted ID receives `{"snapshot": ..., "events": [...]}`; annotations
use an image copy. The core owns ingestion, FPS scheduling, metadata, uploads,
and backend delivery. This package owns inference, event conditions and timers.

`close_session()` discards session/tracker state. `close()` removes all sessions
and releases model references. Calls are synchronous and serialized by the host.
Non-increasing timestamps/sequences, resolution changes, and source gaps reset
tracking and analytics. Missing tracks are pruned without synthesized exit events,
matching the original behavior.

## Configuration and output

Settings go directly in `inference.config`, without the old `params` wrapper:

| Setting | Default | Purpose |
| --- | --- | --- |
| `fps` | `5` | Host frame rate and tracker timing |
| `confidence_threshold` | `0.4` | Detection session confidence filter |
| `count_stabilization_time` | `0` | Seconds a changed count must persist |
| `zone_min_overlap` | `0.3` | Fraction of a box inside a polygon |
| `tracking_config` | unset | YAML path containing `BoTSORTConfig` fields |
| `missing_track_ttl_seconds` | `10` | Analytics retention for missing tracks |
| `direction_threshold_pixels` | `3` | Direction classification threshold |
| `line_crossing_tolerance_pixels` | `1` | Crossing jitter band |
| `source_gap_seconds` | `10` | Source interruption reset threshold |

Tracking confidence thresholds come from BoT-SORT settings. Tracker `frame_rate`
and `person_class_id` are derived from session FPS and model class names. ReID is
disabled if weights are absent. Relative model/YAML paths resolve in the app process.

Zones accept `type`/`points` or legacy `zone_type`/`polygon`. Coordinates between
zero and one scale to image dimensions; others are pixels. Polygon membership
uses box overlap; line crossings use box centers. Snapshot boxes are pixels:
`box` is xyxy for detection, `bbox` is xywh for tracking.

Detection snapshots contain people and total/per-zone counts. Event mode emits
the initial count and stabilized total/zone count changes, including when no
zones exist. Tracking snapshots include direction, speed, occupancy and dwell
time; event mode emits `person_zone.enter`, `person_zone.exit`, and
`person_line.cross`. Periodic mode emits immediately on the first frame and then
at `output.interval` seconds (zero means every processed frame). Events include
an annotated image. The core adds IDs/timestamps and handles S3/backend delivery.

## HTTP API

`api: true` loads `cv_people.http:PeopleDetectionAPI` and registers
`POST /api/v1/detect_people`. Send a raw image body or multipart `image`/`file`:

```json
{"status": 200, "message": "succeeded", "data": {"people": [], "people_count": 0}}
```

The HTTP confidence threshold is 0.4. Its model instance is independent of
runtime models because HTTP APIs run in a separate process. Inference and cleanup run on one dedicated worker thread under a lock, preserving
TensorRT CUDA context ownership. The HTTP API only loads the detector.
This is an image-upload example: JSON/base64 and on-demand camera capture from
the old HTTP endpoint are not implemented. Runtime camera ingestion remains
the core application's responsibility.

## Source mapping

- `runtime.py`: lifecycle, shared inference, and output scheduling.
- `backend.py`: detection conversion and BoT-SORT/ReID helpers from the workers.
- `analytics.py`: count stabilization, dwell, direction, and crossing logic from
  `person_detection_session.py` and `people_tracking_session.py`.
- `zones.py`: geometry and drawing adapted from `generic_runtime.py`.
- `botsort_reid/`: local tracking helpers using the installed detection and ReID packages.
- `http.py`: standalone image-upload adaptation of `PeopleDetectionAPI`.

No tests, model inference, package build, or dependency installation were run
when creating this example.
