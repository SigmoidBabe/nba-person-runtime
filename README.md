# cv.people CUDA runtime

Standalone `cv_demo.runtimes` package for `person.detection` and
`person.tracking`. Detection, TensorRT execution, and OSNet ReID are implemented
inside this package. There is no dependency on `person-inference`,
`osnet-reid-trt`, PyCUDA, or host application modules.

## Installation

Use CUDA 12.x, CuPy 13.x and TensorRT 8.6–10.x compatible with your engines:

```sh
python3 -m pip install -e ./packages/inference-person
CV_DEMO_RUNTIME_REGISTRY=packages/inference-person/registry.yaml python3 -m app.cv_demo.main
```

Set engine paths in this package's `registry.yaml` before starting the host.
Only one installed distribution may export `cv.people`; replace the earlier
example runtime when installing this one. When combining registries, enable
only one owner for each person capability. Models are loaded on first session
creation and shared across sessions. Engines and weights are supplied separately.

The registry's `config` contains package settings (not the host's app config).
Only `model.person` is required; omitted settings use the defaults below:

```yaml
config:
  model:
    person: /opt/models/person_sprint_11.engine
```

| Optional setting | Default | Purpose |
| --- | --- | --- |
| `device` | `0` | CUDA GPU index; the host ingestion GPU must match |
| `model.reid` | unset | ReID engine path; omit for IoU/motion-only tracking |
| `model.tracking_config` | unset | BoT-SORT YAML path; otherwise use built-in tracker defaults |
| `detector_size` | `[640, 640]` | Detector input size |
| `detector_layout` | `BCN` | Raw detector output layout |
| `detector_resize` | `stretch` | Detector resize mode |
| `iou_threshold` | `0.4` | Detection NMS IoU threshold |
| `max_people` | `300` | Maximum detections per frame |
| `reid_batch_size` | `32` | Maximum ReID crop batch size |
| `zone_min_overlap` | `0.3` | Default session zone overlap threshold |

A session's `inference.config.tracking_config` overrides `model.tracking_config`.

## TOPS calculation API

Set `config.model.person_detection.operations_per_inference` and, for tracking,
`config.model.person_reidentification.operations_per_inference` in `registry.yaml`
to each model's operation count for one input item. These values are raw operation
counts, not GOPs or TOPs. Missing or `null` counts for a required model cause an
error. Detection requests do not require or validate ReID operation counts.

The API follows the example package's host-managed gRPC pattern. It is exported
under `cv_demo.grpc_apis` as `person.performance` and enabled by `grpc_apis` in
`registry.yaml`. Reinstall the package after this entry point change and restart
the host to load it. Calculation requires no GPU or loaded engines.

Call `cv_demo.api.v1.PackageAPI/Invoke` with:

```json
{
  "api": "person.performance",
  "method": "calculate_tops",
  "payload": {"cv_capability": "person.tracking", "fps": 30, "batch_size": 4}
}
```

`cv_capability` is required: use `person.detection` or `person.tracking`.
The strings `person detection` and `person tracking` are also accepted and
normalized to the dotted capability IDs in the response.

`fps` must be a finite positive number and `batch_size` a positive integer
no greater than `2**53 - 1`, the safe integer limit for protobuf Struct numbers.
For this estimate, `fps` means batches per second: every batch contains
`batch_size` frames. Detection uses one detector inference per frame. Tracking
combines one detector inference and 20 ReID inferences per frame, assuming
20 people in every frame:

```text
detection TOPS = detector_operations * fps * batch_size / 1e12
tracking TOPS = (detector_operations + 20 * reid_operations) * fps * batch_size / 1e12
```

The result estimates required arithmetic throughput, not measured hardware
performance. Actual ReID work depends on the number of person crops.

For example, with detection set to `1000000000` operations and ReID to
`500000000` operations, the tracking response is:

```json
{
  "cv_capability": "person.tracking",
  "fps": 30,
  "batch_size": 4,
  "people_per_frame": 20,
  "models": {
    "person_detection": {"operations_per_inference": 1000000000, "tops": 0.12},
    "person_reidentification": {"operations_per_inference": 500000000, "tops": 1.2}
  },
  "total_tops": 1.32
}
```

Invalid parameters or operation counts raise `ValueError` for the host to map
to a gRPC error. Unknown methods raise `NotImplementedError`. Python callers can
also use `cv_people.tops.calculate_tops(config, fps, batch_size, cv_capability)`
directly.

## GPU data path

Runtime input must expose `__cuda_array_interface__` as a non-empty HWC uint8
BGR frame on the configured GPU. CPU frames are rejected. Strided CUDA IPC
frames are wrapped directly, without uploading or copying their pixel storage.

- A fused CUDA kernel resizes, converts BGR to RGB, normalizes and writes NCHW.
  `detector_resize: stretch` preserves the old TensorRT detector's resizing;
  select `letterbox` for an engine expecting aspect-preserving padding.
- TensorRT input and raw outputs use CuPy device allocations. YOLO decoding,
  best-class filtering, confidence filtering, sorting, box restoration/clipping,
  and greedy NMS run on CUDA. NMS considers the top 3000 candidates and returns
  at most `max_people` detections.
- OSNet crops are generated directly from the original CUDA frame. A fused
  kernel performs bilinear sampling, RGB conversion, ImageNet normalization and
  NCHW layout at 256×128. TensorRT ReID inference and L2 normalization run on CUDA.
  Crops are bounded by `reid_batch_size`; engine batch limits are also enforced.
- Only final boxes and normalized embeddings are downloaded for the existing
  BoT-SORT association and zone/session bookkeeping. These metadata operations
  and JSON serialization remain on CPU. Image pixels, crops, raw predictions,
  and image preprocessing/postprocessing never run on CPU.

CuPy and TensorRT share the ingestion CUDA primary context. Model calls are
serialized and select the configured device on every thread. CUDA work is
synchronized even on failures before borrowed frame storage can be recycled.

Detector engines must expose one linear FP16/FP32 NCHW input and one raw YOLOv8
output `[B,4+C,N]` (`BCN`) or `[B,N,4+C]` (`BNC`). Supply a same-stem `.txt` file
with one unique non-empty label per line, including `person`. This runtime uses
detection outputs; segmentation and end-to-end NMS engines are unsupported.
OSNet engines must include the backbone and BN neck, accept `[B,3,256,128]`, and
return `[B,embedding_dim]`. Profile zero controls dynamic batches; fixed batches
and profile minima are padded on GPU and their extra predictions discarded.

## gRPC sessions and events

Send `session.example.json` as a protobuf Struct to
`cv_demo.runtime.v1.RuntimeControl/StartRuntime`. Use
`inference.inference_primitive_usecase: "person tracking"` or `"person detection"`.
Explicit selection is also supported:

```json
{"runtime": {"id": "cv.people", "capability": "person.tracking"}}
```

The host supplies ingestion frames and schedules eligible session IDs. One
CUDA detector pass serves all sessions on the frame, with shared ReID features
for tracking sessions. Each session owns independent tracker and analytics state.

Each frame returns `{session_id: {"snapshot": ..., "events": [{"data": ...}]}}`.
The host sends snapshots to the separate metadata subscription and wraps each
event as `type: "inference_event"`, adding session/camera IDs and frame timestamps.
Events arrive on that session's open `StartRuntime` gRPC stream as
`event_payload`. The package performs no backend POST, S3 upload, JPEG encoding,
or CPU image annotation. Events contain data only, with no image attachment.
The legacy HTTP upload API is no longer exported; starting a gRPC session does
not require a `person.detection` package API entry point or `api: true` flag.

Detection snapshots contain `people_count`, `people`, `zone_counts`, and `zones`.
Event mode emits initial counts and stabilized count changes. Tracking retains
BoT-SORT IDs, direction, speed, occupancy and dwell-time analytics, with
`person_zone.enter`, `person_zone.exit`, and `person_line.cross` events.
Periodic mode emits immediately, then at `output.interval` seconds (zero means
every processed frame).

Session options go directly under `inference.config`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `fps` | 5 | Host frame scheduling and tracker timing |
| `confidence_threshold` | 0.4 | Detection session confidence filter |
| `count_stabilization_time` | 0 | Seconds a changed count must persist |
| `zone_min_overlap` | 0.3 | Fraction of a box inside a polygon |
| `tracking_config` | unset | Session override for tracker YAML |
| `missing_track_ttl_seconds` | 10 | Retention for missing tracks |
| `direction_threshold_pixels` | 3 | Direction classification threshold |
| `line_crossing_tolerance_pixels` | 1 | Crossing jitter band |
| `source_gap_seconds` | 10 | Reset after source interruption |

Zones accept `type`/`points` or `zone_type`/`polygon`; normalized coordinates
scale to frame dimensions. Detection uses xyxy boxes; tracking uses xywh boxes.
Non-increasing timestamps/sequences, resolution changes and source gaps reset
session analytics and tracking. `StopRuntime` closes the session; disconnecting
the event stream only unsubscribes, following the host's runtime contract.

Implementation details: [detector](docs/detection_external.md),
[OSNet](docs/osnet_external.md).
