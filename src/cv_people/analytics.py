"""Per-session analytics adapted from the existing detection/tracking sessions."""
import time
from uuid import uuid4
import cv2
from .zones import ZoneAnalytics

class DetectionAnalytics(ZoneAnalytics):
    DEFAULT_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, spec):
        super().__init__(spec)
        self.confidence_threshold = self.resolve_confidence_threshold(spec["config"])
        self.count_stabilization_time = self.resolve_count_stabilization_time(spec["config"])
        self.confirmed_count_signature = None
        self.candidate_count_signature = None
        self.candidate_count_since = None

    def resolve_confidence_threshold(self, params):
        try:
            value = float(
                params.get(
                    "confidence_threshold",
                    self.DEFAULT_CONFIDENCE_THRESHOLD,
                )
            )
        except (TypeError, ValueError):
            return self.DEFAULT_CONFIDENCE_THRESHOLD

        if not 0.0 <= value <= 1.0:
            return self.DEFAULT_CONFIDENCE_THRESHOLD
        return value

    def filter_people_by_confidence(self, people):
        filtered_people = []
        for person in people:
            try:
                confidence = float(person.get("confidence"))
            except (AttributeError, TypeError, ValueError):
                continue

            if confidence >= self.confidence_threshold:
                filtered_people.append(person)

        return filtered_people

    def resolve_count_stabilization_time(self, params):
        try:
            value = float(params.get("count_stabilization_time", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    def count_signature(self, payload):
        event_payload = payload.get("event_payload") or {}
        zone_counts = event_payload.get("zone_counts") or {}
        return (int(event_payload.get("people_count", 0)), tuple(sorted(
            (str(zone_name), int(count or 0))
            for zone_name, count in zone_counts.items()
        )))

    def should_publish_count_event(self, payload, now=None):
        if now is None:
            now = time.monotonic()

        signature = self.count_signature(payload)
        if self.confirmed_count_signature is None:
            self.confirmed_count_signature = signature
            return True

        if signature == self.confirmed_count_signature:
            self.candidate_count_signature = None
            self.candidate_count_since = None
            return False

        if signature != self.candidate_count_signature:
            self.candidate_count_signature = signature
            self.candidate_count_since = now

        if now - self.candidate_count_since < self.count_stabilization_time:
            return False

        self.confirmed_count_signature = signature
        self.candidate_count_signature = None
        self.candidate_count_since = None
        return True

    def draw_people(self, frame, people):
        for idx, person in enumerate(people, start=1):
            box = person.get("box")
            if not box:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            confidence = person.get("confidence")
            label = f"person {idx}"
            if confidence is not None:
                label = f"{label} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def draw_zone_counts(self, frame, zone_counts):
        y = 30
        for zone_name, count in zone_counts.items():
            cv2.putText(
                frame,
                f"{zone_name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 30


class TrackingAnalytics(ZoneAnalytics):
    def __init__(self, spec):
        super().__init__(spec)
        self.track_id_prefix = f"{spec['session_id']}_{uuid4().hex}_"
        self.person_zone_state = {}
        self.person_line_state = {}
        self.person_motion_state = {}
        self.person_last_seen = {}
        params = spec["config"]
        self.missing_track_ttl_seconds = self.resolve_float_param(params, "missing_track_ttl_seconds", 10.0)
        self.direction_threshold_pixels = self.resolve_float_param(params, "direction_threshold_pixels", 3.0)
        self.line_crossing_tolerance_pixels = self.resolve_float_param(params, "line_crossing_tolerance_pixels", 1.0)

    def resolve_float_param(self, params, key, default):
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return default

    def draw_trackers(self, frame, trackers, analytics=None):
        analytics = analytics or {}
        for person_id, data in trackers.items():
            bbox = data.get("bbox")
            if not bbox:
                continue

            x, y, width, height = map(int, bbox)
            x2 = x + width
            y2 = y + height
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            person_analytics = analytics.get(str(person_id), {})
            formatted_person_id = person_analytics.get(
                "person_id",
                f"{self.track_id_prefix}{person_id}",
            )
            label = f"ID {formatted_person_id}"
            direction = person_analytics.get("direction")
            if direction:
                label = f"{label} {direction}"

            zones = person_analytics.get("zones") or []
            if zones:
                max_duration = max(
                    zone.get("duration_seconds", 0.0)
                    for zone in zones
                )
                label = f"{label} {int(max_duration)}s"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def tracker_center(self, bbox):
        x, y, width, height = [float(value) for value in bbox[:4]]
        return x + (width / 2.0), y + (height / 2.0)

    def classify_direction(self, dx, dy):
        if (
            abs(dx) < self.direction_threshold_pixels
            and abs(dy) < self.direction_threshold_pixels
        ):
            return "stationary"

        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"

        return "down" if dy > 0 else "up"

    def update_direction(self, person_id, bbox, now):
        center = self.tracker_center(bbox)
        state = self.person_motion_state.get(person_id)
        self.person_motion_state[person_id] = {
            "center": center,
            "timestamp": now,
        }

        if state is None:
            return {
                "direction": "unknown",
                "direction_vector": {"x": 0.0, "y": 0.0},
                "speed_pixels_per_second": 0.0,
            }

        previous_center = state["center"]
        elapsed = max(now - state.get("timestamp", now), 0.0)
        dx = center[0] - previous_center[0]
        dy = center[1] - previous_center[1]
        speed = 0.0
        if elapsed > 0:
            speed = ((dx ** 2 + dy ** 2) ** 0.5) / elapsed

        return {
            "direction": self.classify_direction(dx, dy),
            "direction_vector": {
                "x": round(dx, 3),
                "y": round(dy, 3),
            },
            "speed_pixels_per_second": round(speed, 3),
        }

    def iso_time(self, timestamp):
        return time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(timestamp),
        )

    def zone_payload_metadata(self, zone_name):
        metadata = {
            key: value
            for key, value in self.zone_meta.get(zone_name, {}).items()
            if key not in {"camera_zone_id", "zone_id", "zone_name"}
        }
        zone_metadata = self.zone_meta.get(zone_name, {})
        zone_id = zone_metadata.get("camera_zone_id")
        if zone_id is None:
            zone_id = zone_metadata.get("zone_id")
        return {"zone_id": zone_id, **metadata}

    def active_zone_payload(self, zone):
        zone_name = zone.get("zone_name")
        payload = {
            key: value
            for key, value in zone.items()
            if key not in {"camera_zone_id", "zone_id", "zone_name"}
        }
        return {**self.zone_payload_metadata(zone_name), **payload}

    def update_zone_durations(self, person_id, bbox, now):
        inside_zones = set(self.zones_for_box(bbox, box_format="tlwh"))
        person_state = self.person_zone_state.setdefault(person_id, {})
        active_zones = []
        zone_events = []

        for zone_name in self.zone_polygons:
            zone_state = person_state.setdefault(
                zone_name,
                {
                    "inside": False,
                    "entry_time": None,
                    "entry_timestamp": None,
                    "last_duration_seconds": 0.0,
                },
            )
            inside_now = zone_name in inside_zones
            was_inside = zone_state["inside"]

            if inside_now and not was_inside:
                entry_timestamp = self.iso_time(now)
                zone_state["entry_time"] = now
                zone_state["entry_timestamp"] = entry_timestamp
                zone_state["last_duration_seconds"] = 0.0
                zone_events.append({
                    "event_type": "person_zone.enter",
                    "event_action": "enter",
                    "person_id": person_id,
                    "duration_seconds": 0.0,
                    "entry_timestamp": entry_timestamp,
                    **self.zone_payload_metadata(zone_name),
                })
            elif not inside_now and was_inside:
                entry_time = zone_state.get("entry_time")
                entry_timestamp = zone_state.get("entry_timestamp")
                duration = 0.0
                if entry_time is not None:
                    duration = max(0.0, now - entry_time)
                    zone_state["last_duration_seconds"] = duration
                zone_events.append({
                    "event_type": "person_zone.exit",
                    "event_action": "exit",
                    "person_id": person_id,
                    "duration_seconds": round(duration, 3),
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": self.iso_time(now),
                    **self.zone_payload_metadata(zone_name),
                })
                zone_state["entry_time"] = None
                zone_state["entry_timestamp"] = None

            zone_state["inside"] = inside_now
            if not inside_now:
                continue

            entry_time = zone_state.get("entry_time") or now
            duration = max(0.0, now - entry_time)
            zone_state["last_duration_seconds"] = duration
            active_zones.append({
                "zone_name": zone_name,
                "duration_seconds": round(duration, 3),
                "entry_timestamp": zone_state.get("entry_timestamp"),
                **self.zone_meta.get(zone_name, {}),
            })

        return active_zones, zone_events

    def line_side(self, point, line):
        """Return the side of a directed line, allowing a small jitter band."""
        (x1, y1), (x2, y2) = line
        cross = (
            (x2 - x1) * (point[1] - y1)
            - (y2 - y1) * (point[0] - x1)
        )
        line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        tolerance = (
            getattr(self, "line_crossing_tolerance_pixels", 1.0)
            * line_length
        )
        if abs(cross) <= tolerance:
            return 0
        return 1 if cross > 0 else -1

    def segments_intersect(self, first_start, first_end, second_start, second_end):
        def orientation(start, end, point):
            return (
                (end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0])
            )

        def on_segment(start, point, end):
            epsilon = 1e-6
            return (
                min(start[0], end[0]) - epsilon
                <= point[0]
                <= max(start[0], end[0]) + epsilon
                and min(start[1], end[1]) - epsilon
                <= point[1]
                <= max(start[1], end[1]) + epsilon
            )

        o1 = orientation(first_start, first_end, second_start)
        o2 = orientation(first_start, first_end, second_end)
        o3 = orientation(second_start, second_end, first_start)
        o4 = orientation(second_start, second_end, first_end)
        epsilon = 1e-6

        if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
            (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
        ):
            return True

        return (
            (abs(o1) <= epsilon and on_segment(first_start, second_start, first_end))
            or (abs(o2) <= epsilon and on_segment(first_start, second_end, first_end))
            or (abs(o3) <= epsilon and on_segment(second_start, first_start, second_end))
            or (abs(o4) <= epsilon and on_segment(second_start, first_end, second_end))
        )

    def update_line_crossings(self, person_id, bbox, now):
        center = self.tracker_center(bbox)
        person_state = self.person_line_state.setdefault(person_id, {})
        line_events = []

        for zone_name, line in getattr(self, "zone_lines", {}).items():
            current_side = self.line_side(center, line)
            line_state = person_state.setdefault(
                zone_name,
                {"side": None, "center": center},
            )
            previous_side = line_state.get("side")
            previous_center = line_state.get("center", center)

            # Keep the last definite side while the center is within the jitter
            # band. This lets a track land on the line for one or more frames
            # without losing the eventual crossing.
            if current_side == 0:
                continue

            if (
                previous_side is not None
                and current_side != previous_side
                and self.segments_intersect(
                    previous_center,
                    center,
                    line[0],
                    line[1],
                )
            ):
                crossing_direction = (
                    "left_to_right" if previous_side > 0 else "right_to_left"
                )
                line_events.append({
                    "event_type": "person_line.cross",
                    "event_action": "cross",
                    "person_id": person_id,
                    "crossing_direction": crossing_direction,
                    "crossing_timestamp": self.iso_time(now),
                    **self.zone_payload_metadata(zone_name),
                })

            line_state["side"] = current_side
            line_state["center"] = center

        return line_events

    def analyze_trackers(self, trackers, now):
        active_ids = set()
        people = []
        analytics_by_person = {}
        zone_events = []
        zones = {
            zone_name: {
                "people_count": 0,
                "people": [],
                **self.zone_payload_metadata(zone_name),
            }
            for zone_name in self.zone_polygons
        }

        for raw_person_id, data in trackers.items():
            bbox = data.get("bbox")
            if not bbox:
                continue

            person_id = f"{self.track_id_prefix}{raw_person_id}"
            active_ids.add(person_id)
            self.person_last_seen[person_id] = now

            direction = self.update_direction(person_id, bbox, now)
            active_zones, person_zone_events = self.update_zone_durations(
                person_id,
                bbox,
                now,
            )
            person_line_events = self.update_line_crossings(
                person_id,
                bbox,
                now,
            )
            person_payload = {
                "person_id": person_id,
                "bbox": [round(float(value), 3) for value in bbox[:4]],
                **direction,
                "zones": [
                    self.active_zone_payload(zone)
                    for zone in active_zones
                ],
            }
            people.append(person_payload)
            analytics_by_person[str(raw_person_id)] = person_payload
            for zone_event in person_zone_events + person_line_events:
                zone_events.append({
                    **zone_event,
                    "bbox": person_payload["bbox"],
                    **direction,
                })

            for zone in active_zones:
                zone_name = zone["zone_name"]
                zones[zone_name]["people_count"] += 1
                zones[zone_name]["people"].append({
                    "person_id": person_id,
                    "duration_seconds": zone["duration_seconds"],
                    "direction": direction["direction"],
                })

        self.prune_missing_trackers(active_ids, now)
        zone_counts = {
            zone_name: zone_data["people_count"]
            for zone_name, zone_data in zones.items()
        }
        return analytics_by_person, people, zone_counts, zones, zone_events

    def prune_missing_trackers(self, active_ids, now):
        for person_id, last_seen in list(self.person_last_seen.items()):
            if person_id in active_ids:
                continue
            if now - last_seen < self.missing_track_ttl_seconds:
                continue

            self.person_last_seen.pop(person_id, None)
            self.person_zone_state.pop(person_id, None)
            self.person_line_state.pop(person_id, None)
            self.person_motion_state.pop(person_id, None)

