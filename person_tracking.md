# Person tracking event payloads

Capability: `person.tracking` (use case: `person tracking`).

The host sends each event on the session's open `StartRuntime` gRPC stream.
The examples use the shared gRPC envelope below. Only the shape of
`event_payload` changes between event kinds; the outer `type` remains
`"inference_event"`. IDs, timestamps, and frame sequence values are illustrative
and vary for actual messages. `frame_timestamp_ns` is shown as a string to
preserve nanosecond precision.

The core adds `image_path` and `video_path` to this outer envelope according to
capability registry switches. Images use the trigger frame; enabled recording
delays event publication until the clip finishes, while metadata remains immediate.
Paths may be null; media failures do not invalidate events. Periodic events also
trigger enabled media. See
[core event images and recordings](README.md#core-event-images-and-recordings)
for configuration, timing, and failure fields.

| Output mode | `event_payload` variants |
| --- | --- |
| `periodic` | Tracking snapshot, immediately on the first processed frame and then every `output.interval` seconds; zero emits every processed frame. |
| `event` | Individual `person_zone.enter`, `person_zone.exit`, and `person_line.cross` events as transitions occur. Multiple events on one frame are separate messages. |

Tracking snapshots are available through the host's metadata subscription in
either mode. In `event` mode, the event stream emits transitions rather than
periodic snapshots or detection count-change events.

## Periodic tracking snapshot

```json
{
  "event_payload": {
    "people_count": 1,
    "people": [
      {
        "person_id": "11_0123456789abcdef0123456789abcdef_1",
        "bbox": [
          1321.0,
          489.0,
          204.0,
          187.0
        ],
        "direction": "right",
        "direction_vector": {
          "x": 5.0,
          "y": 0.0
        },
        "speed_pixels_per_second": 25.0,
        "zones": [
          {
            "zone_id": 9036,
            "zone_type": "polygon",
            "duration_seconds": 5.0,
            "entry_timestamp": "2026-11-13T08:27:14Z"
          }
        ]
      }
    ],
    "zone_counts": {
      "9036": 1
    },
    "zones": {
      "9036": {
        "people_count": 1,
        "people": [
          {
            "person_id": "11_0123456789abcdef0123456789abcdef_1",
            "duration_seconds": 5.0,
            "direction": "right"
          }
        ],
        "zone_id": 9036,
        "zone_type": "polygon"
      }
    }
  },
  "event_id": "7c110961-131d-4ea1-9107-21987aa78530",
  "frame_timestamp": "2026-11-13T08:27:19.503Z",
  "frame_timestamp_ns": "1794558439503367555",
  "timestamp": "2026-11-13T08:27:19.503Z",
  "frame_sequence": 1,
  "session_id": "11",
  "type": "inference_event",
  "camera_id": "CAM-002"
}
```

`people_count` is the number of active tracked people. Each `person_id` contains
a session-specific random prefix and the tracker ID; it persists across frames
within that tracker lifecycle and changes when tracking is reset.

Tracking uses `bbox: [x, y, width, height]` in pixels. Each person's `zones`
contains active polygon-zone objects with entry time and dwell duration.
`zone_counts` and the `zones` object are keyed by zone name, falling back to the
string form of its ID. Unlike detection, tracking's top-level `zones` is an
object, with occupancy and a list of people per polygon zone.

All tracking payloads include `direction`, `direction_vector`, and
`speed_pixels_per_second`. Direction is `unknown` on the first observation,
then `stationary`, `left`, `right`, `up`, or `down`. The direction vector is the
pixel displacement of the box center since the previous observation; speed is
that displacement's magnitude divided by elapsed seconds.

The following examples show replacement values for `event_payload` inside the
same gRPC envelope. They do not add another nested `event_payload` field.

## Polygon-zone entry: `person_zone.enter`

Emitted when a tracked person first satisfies the polygon overlap threshold,
including a person first observed inside the zone. Duration starts at zero.

```json
{
  "event_type": "person_zone.enter",
  "event_action": "enter",
  "person_id": "11_0123456789abcdef0123456789abcdef_1",
  "duration_seconds": 0.0,
  "entry_timestamp": "2026-11-13T08:27:19Z",
  "zone_id": 9036,
  "zone_type": "polygon",
  "bbox": [
    1321.0,
    489.0,
    204.0,
    187.0
  ],
  "direction": "right",
  "direction_vector": {
    "x": 5.0,
    "y": 0.0
  },
  "speed_pixels_per_second": 25.0
}
```

## Polygon-zone exit: `person_zone.exit`

Emitted when an observed tracked person moves outside the polygon overlap
threshold. `duration_seconds` measures the time since entry. A track disappearing
from the frame does not by itself emit an exit event.

```json
{
  "event_type": "person_zone.exit",
  "event_action": "exit",
  "person_id": "11_0123456789abcdef0123456789abcdef_1",
  "duration_seconds": 5.0,
  "entry_timestamp": "2026-11-13T08:27:14Z",
  "exit_timestamp": "2026-11-13T08:27:19Z",
  "zone_id": 9036,
  "zone_type": "polygon",
  "bbox": [
    1321.0,
    489.0,
    204.0,
    187.0
  ],
  "direction": "right",
  "direction_vector": {
    "x": 5.0,
    "y": 0.0
  },
  "speed_pixels_per_second": 25.0
}
```

## Line crossing: `person_line.cross`

Emitted when the tracked box center changes sides and its movement intersects
the configured line segment, subject to `line_crossing_tolerance_pixels`.
`crossing_direction` is `left_to_right` for a positive-to-negative side change,
or `right_to_left` for the reverse; these depend on the order of the configured
line endpoints. This payload uses `crossing_timestamp` instead of dwell fields.

```json
{
  "event_type": "person_line.cross",
  "event_action": "cross",
  "person_id": "11_0123456789abcdef0123456789abcdef_1",
  "crossing_direction": "left_to_right",
  "crossing_timestamp": "2026-11-13T08:27:19Z",
  "zone_id": 9040,
  "zone_type": "line",
  "bbox": [
    1321.0,
    489.0,
    204.0,
    187.0
  ],
  "direction": "right",
  "direction_vector": {
    "x": 5.0,
    "y": 0.0
  },
  "speed_pixels_per_second": 25.0
}
```

Zone-related objects and events include `zone_id` and configured zone metadata.
These examples assume zones configured with `polygon` and `zone_type`; additional
metadata such as `type`, `points`, or custom fields is included when configured.
Zone IDs retain their configured type. Analytics timestamps inside
`event_payload` use UTC with whole-second precision; outer frame timestamps
retain the host's precision.

See [person detection](person_detection.md) for count snapshots without tracking
IDs or motion analytics.
