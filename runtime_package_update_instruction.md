# Runtime package update instructions

The core owns event image saving, a shared GPU frame buffer and video recordings per camera.
Runtime packages continue to own inference and event detection. Existing packages
using the documented `process_frame` result shape need no interface changes to
receive media paths when enabled in the capability registry. Remove any package-side
evidence buffer, image saver or video writer that
duplicates this feature; retain temporal buffers required by the model itself.

## Enable media per capability

In `registry.yaml`, set the switches on each capability, alongside `id` and
`output_modes`, not inside package `config`:

```yaml
capabilities:
  - id: person.detection
    api: true
    recording: true
    image_save: true
    output_modes: [periodic, event]
  - id: person.tracking
    recording: false
    image_save: true
    output_modes: [periodic, event]
```

Both switches default to false and must be YAML booleans. They are independent:
image-only events are delivered without waiting for video; video-only events do
not save JPEG files. These settings come from the loaded deployment registry;
session request fields and package results cannot override them. Restart after
editing the registry. GPU video history is allocated only for cameras with an
active recording-enabled capability. The application-level
`event_recording.enabled` remains a global video off switch.

## Emit events on their trigger frame

Return an event during the `process_frame` call in which it occurs. The core uses
that call's camera ID, sequence and ingestion timestamp as the event anchor.
Events for older frames or asynchronous event callbacks are not supported by this
interface. Do not wait for the five future frames inside the package.

```python
def process_frame(self, frame, session_ids):
    results = {}
    for session_id in session_ids:
        snapshot, transitions = self.analyze(session_id, frame["image"])
        results[session_id] = {
            "snapshot": snapshot,
            "events": [{"data": transition} for transition in transitions],
        }
    return results
```

`analyze` above represents the package's existing analysis. Return one result for
every submitted session ID. Event `data` and `snapshot` must be JSON-compatible.
An optional event `image` supplies an annotated BGR image or crop. With
`image_save: true`, the core saves that image, or the original trigger frame when
the package omits it. The image is encoded and saved during event publication,
before the borrowed trigger frame is released; saving does not wait for video.
Package-provided images also retain the existing inline `image_base64` behavior,
independently of the image file switch.
The video contains the original unannotated camera frames. Do not place arrays,
video bytes, `video_path`, `image_path`, or core envelope fields inside event `data`.

Each item in `events` triggers the capability's enabled media outputs, including
periodic events. Use `snapshot`
for continuous metadata and emit `events` only at the package's intended event
frequency. The core does not infer event conditions from metadata.

## Preserve GPU frame ownership

RTSP frames provide a read-only BGR uint8 CuPy image and `memory: "cuda"`.
The image, slices, tensors and DLPack views are borrowed until `process_frame`
returns. Never retain them afterward. Copy into package-owned memory only when
the model itself needs temporal state. GPU work is synchronized before ingestion
can reuse the borrowed allocation. The core makes its own recording copy before
calling any runtime, so packages do not need to copy frames for recording.

CPU sources are uploaded into GPU-owned recording storage. Recording requires
CuPy and a working CUDA device even if the runtime itself performs CPU inference.
The existing `requirements-gpu.txt` supplies CuPy for CUDA 12 environments.

## Consume the gRPC event

`RuntimeControl.StartRuntime` continues to return `google.protobuf.Struct` messages;
no protobuf regeneration is needed. Metadata snapshots are published immediately
through `MetadataService.Subscribe`. When recording is enabled, inference events
are published after the recording finishes, retaining the original event ID and
trigger timestamp. Other events are published after any image saving completes.
Overlapping events collected for the same camera share a saved file, including
events from different runtime packages and sessions.

```json
{
  "type": "inference_event",
  "event_id": "example-event-id",
  "session_id": "session-01",
  "camera_id": "camera-01",
  "frame_sequence": 100,
  "frame_timestamp_ns": "1794800000000000000",
  "frame_timestamp": "2026-11-16T03:33:20.000Z",
  "timestamp": "2026-11-16T03:33:20.000Z",
  "event_payload": {"event_type": "person_line.cross"},
  "image_path": "/srv/cv-demo/recordings/images/example-event-id.jpg",
  "video_path": "/srv/cv-demo/recordings/events/example.mp4",
  "video_start_timestamp_ns": "1794799999800000000",
  "video_end_timestamp_ns": "1794800000200000000",
  "video_frame_count": 11,
  "video_fps": 25,
  "video_event_offset_seconds": 0.2,
  "video_pre_frames": 5,
  "video_post_frames": 5
}
```

`video_path` is an absolute path on the server, available only after the completed
MP4 has been renamed into place. Use a mounted shared directory if a remote client
needs file access; this feature does not expose a video download endpoint or upload
videos to object storage. Multiple events referencing the same path remain separate
events. Use `event_id` for event identity, not the video filename.

`image_path` is likewise an absolute server-side path to a completed JPEG. Each
event gets its own image filename, even when several events share a video clip.
When image saving is disabled, `image_path` is null. On image encoding or saving
failure, it is null with `image_error: "image_save_failed"`; event and video
processing continue independently. Optional inline image encoding failures use
`image_error: "image_encoding_failed"` when file saving was not requested.

`video_event_offset_seconds` is the trigger frame's playback offset. The video is
constant FPS, estimated from the first and last ingestion timestamps, so variable
source timing is approximated. Nanosecond timestamps remain strings to preserve
precision in protobuf Struct. Frame counts describe the entire shared clip relative
to this event and can exceed five when event windows merge.

If recording fails, the event is still emitted with `video_path: null` and a
`video_error`, such as `gpu_buffer_unavailable`, `recording_queue_full`,
`clip_event_limit`, or `encoding_failed`. A capability with recording disabled
gets `video_path: null` without a video error. A recording-enabled capability
blocked by the global video switch gets `video_error: "recording_disabled"`.
Treat the event as valid even without a video.
Source idle, stop, shutdown or discontinuity can finalize a partial recording with
`video_truncation_reason`; startup can also provide fewer than five preceding
frames. Inspect `video_pre_frames` and `video_post_frames` for the available context.
Streams and queues remain live and best effort; a disconnected client or a stream
already terminated by STOPPED/FAILED may not receive a pending recording event.

## Configure the core

Set capability switches in the registry as above. Configure storage directories
and resource limits in application configuration, not the runtime package config:

```yaml
event_jpeg_quality: 50
event_images:
  directory: recordings/images
event_recording:
  enabled: true
  directory: recordings/events
  pre_frames: 5
  post_frames: 5
  max_clip_frames: 150
  max_clip_bytes: 134217728
  max_events_per_clip: 256
  queue_size: 2
  idle_timeout_seconds: 5
  fallback_fps: 25
```

The defaults are five frames before, the trigger frame, and five frames after.
These are frames delivered by camera ingestion before inference FPS filtering,
not every frame originally produced by the camera: upstream ingestion may sample
or drop frames. Configure ingestion FPS accordingly.

The rolling history and active recordings hold owned GPU copies, shared across
all runtimes on the camera. Encoding uses one background OpenCV MP4 writer and
downloads one frame at a time to CPU RAM. This implementation does not use NVENC.
The deployed OpenCV build must support the `mp4v` encoder. Odd dimensions are
padded by one pixel on the right or bottom for encoding.

`max_clip_frames` must fit `pre_frames + post_frames + 1`. The GPU bytes required
by that window must fit `max_clip_bytes`. At a clip size limit, completed events
are queued for encoding and events still awaiting future frames continue in an
overlapping clip. A full encoding queue fails those queued events explicitly
rather than growing the queue. The event count limit rejects additional events
for the active clip while retaining their metadata in a failure event.

`max_clip_bytes` bounds frame pixels per clip, not total process VRAM. Each camera
has a history and active clip; up to `queue_size` clips can wait and one can encode.
Their frames share allocations wherever windows overlap. CuPy may retain freed
allocations in its memory pool. Account for camera count, resolution, models and
allocator overhead when sizing deployment VRAM. There is no automatic disk retention
policy; provision storage and delete or archive old recordings separately.

For a single-frame clip, `fallback_fps` supplies playback FPS. Idle sources finalize
pending clips after `idle_timeout_seconds` without delivered frames. Relative
recording directories resolve against the server working directory at startup.

## Load updated TensorRT models

`ModelService.UpdateModelPath` now receives `.tar.gz` archives containing
`manifest.json` and a self-contained ONNX model. The core builds engines while
the managed GPU pipeline is stopped, updates registry paths, and constructs fresh
package instances. See [model OTA](model-ota.md).

For nested `config.model` settings, load the generated TensorRT path from `engine`;
`onnx` and `manifest` point to staged release files. Packages using legacy string
model settings receive the generated engine path as a string. The package must
support TensorRT engines in its inference backend and release its resources in
`close()`. API instances now run in the supervised GPU worker, independently of
runtime instances, rather than inside the gRPC listener process.

Keep durable data outside model objects. Session trackers and counters restart
after OTA; enabled sessions are restored from the core database. Clients should
accept the nonterminal `PAUSED` state without closing their runtime stream, then
handle the normal startup states when inference resumes.
