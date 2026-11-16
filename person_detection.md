# Person detection event payloads

Capability: `person.detection` (use case: `person detection`).

The host sends each event on the session's open `StartRuntime` gRPC stream.
The examples use the shared gRPC envelope below. Only the shape of
`event_payload` changes between event kinds; the outer `type` remains
`"inference_event"`. IDs, timestamps, and frame sequence values are illustrative
and vary for actual messages. `frame_timestamp_ns` is shown as a string to
preserve nanosecond precision.

## Count snapshot

| Output mode | When this payload is emitted |
| --- | --- |
| `periodic` | Only when a polygon-zone count change persists for that zone's `config.min_count_change_time` seconds (default: 3); `output.interval` does not trigger detection events. |
| `event` | Only when a polygon-zone count change persists for that zone's `config.min_count_change_time` seconds (default: 3). |

Both modes use the same payload structure, with no `event_type` or
`event_action`. The first frame establishes a baseline without an event. Each
zone has an independent timer, restarted whenever its candidate count changes
and canceled if it returns to the confirmed count. Zero confirms a change
immediately. Total-person counts, boxes, and confidence alone do not trigger
events. Sessions without polygon zones emit no detection events.

Configure the delay inside each zone, for example
`"config": {"min_count_change_time": 5}`. Values must be finite nonnegative
numbers. The session-level `count_stabilization_time` option is no longer used.

Event `zone_counts` and `zones[].people_count` contain confirmed counts, so a
pending change in another zone is not published early. `people` and
`people_count` describe the current frame. Metadata snapshots continue to
contain the current frame's observed counts.

```json
{
  "event_payload": {
    "people": [
      {
        "person_index": 1,
        "zones": [
          "9036"
        ],
        "area": 38148,
        "confidence": 0.8116765022277832,
        "box": [
          1321,
          489,
          1525,
          676
        ]
      },
      {
        "person_index": 2,
        "zones": [],
        "area": 23940,
        "confidence": 0.7090195417404175,
        "box": [
          1044,
          220,
          1177,
          400
        ]
      },
      {
        "person_index": 3,
        "zones": [],
        "area": 8383,
        "confidence": 0.6397355198860168,
        "box": [
          993,
          128,
          1076,
          229
        ]
      },
      {
        "person_index": 4,
        "zones": [
          "9036"
        ],
        "area": 95904,
        "confidence": 0.6206216216087341,
        "box": [
          1272,
          714,
          1560,
          1047
        ]
      }
    ],
    "zone_counts": {
      "9032": 0,
      "9036": 2
    },
    "people_count": 4,
    "zones": [
      {
        "people_count": 0,
        "zone_id": 9032
      },
      {
        "people_count": 2,
        "zone_id": 9036
      }
    ]
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

| Field in `event_payload` | Meaning |
| --- | --- |
| `people_count` | Total detections after the session confidence filter, including people outside zones. |
| `people` | Detections for this frame. `person_index` is a per-frame index, not a persistent tracking ID. |
| `people[].box` | Pixel coordinates `[x1, y1, x2, y2]`. |
| `people[].area` | Box area in square pixels. |
| `people[].confidence` | Detector confidence score. |
| `people[].zones` | Polygon-zone keys containing the person according to `zone_min_overlap`; empty when outside all zones. |
| `zone_counts` | Number of people in each polygon zone, keyed by zone name, falling back to the string form of its ID. |
| `zones` | Array of polygon-zone summaries containing `zone_id` and `people_count`. IDs retain their configured type. |

Overlapping zones can count the same person more than once. With no detections,
`people` is empty and counts are zero. Detection does not emit individual
zone-entry, zone-exit, or line-crossing events; see
[person tracking](person_tracking.md) for those payloads.
