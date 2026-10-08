"""External runtime contract: shared models, isolated session analytics."""
import copy
import math
import time

from .analytics import DetectionAnalytics, TrackingAnalytics


class PeopleRuntime:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.sessions = {}
        self.backend = None
        self.closed = False

    def _backend(self):
        if self.closed:
            raise RuntimeError("Runtime is closed")
        if self.backend is None:
            from .backend import PeopleBackend
            self.backend = PeopleBackend(self.config)
        return self.backend

    def create_session(self, session):
        if self.closed:
            raise RuntimeError("Runtime is closed")
        spec = copy.deepcopy(session)
        sid = spec["session_id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError("session_id must be a non-empty string")
        if sid in self.sessions:
            raise ValueError(f"Session already exists: {sid}")
        if not isinstance(spec.get("camera_id"), str) or not spec["camera_id"]:
            raise ValueError("camera_id must be a non-empty string")
        if spec.get("capability") not in ("person.detection", "person.tracking"):
            raise ValueError("Unsupported person capability")
        config = spec.setdefault("config", {})
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if not isinstance(spec.get("zones", []), list) or any(
            not isinstance(zone, dict) for zone in spec.get("zones", [])
        ):
            raise ValueError("zones must be an array of objects")
        defaults = {
            "fps": 5, "confidence_threshold": 0.4,
            "zone_min_overlap": self.config.get("zone_min_overlap", 0.3),
            "count_stabilization_time": 0, "missing_track_ttl_seconds": 10,
            "direction_threshold_pixels": 3, "line_crossing_tolerance_pixels": 1,
            "source_gap_seconds": 10,
        }
        for key, default in defaults.items():
            value = config.setdefault(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            if key in ("fps", "source_gap_seconds") and value == 0:
                raise ValueError(f"{key} must be greater than zero")
            if key in ("confidence_threshold", "zone_min_overlap") and value > 1:
                raise ValueError(f"{key} must be between zero and one")
        output = spec.setdefault("output", {"mode": "periodic", "interval": 0})
        if not isinstance(output, dict) or output.get("mode", "periodic") not in ("periodic", "event"):
            raise ValueError("output.mode must be periodic or event")
        interval = output.setdefault("interval", 0)
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError("output.interval must be finite and nonnegative")
        output.setdefault("mode", "periodic")
        analytics_type = TrackingAnalytics if spec["capability"] == "person.tracking" else DetectionAnalytics
        analytics = analytics_type(spec)
        backend = self._backend()
        if spec["capability"] == "person.tracking":
            backend.get_tracker(self._item(spec))
        self.sessions[sid] = {"spec": spec, "analytics": analytics, "last_output": None,
                              "last_timestamp": None, "last_sequence": None, "last_shape": None}

    @staticmethod
    def _item(spec):
        return {"service_key": spec["session_id"], "camera_id": spec["camera_id"],
                "tracking_config": spec["config"].get("tracking_config"),
                "fps": spec["config"].get("fps", 5)}

    def _reset_on_discontinuity(self, state, frame):
        timestamp = frame["timestamp_ns"]
        previous = state["last_timestamp"]
        shape = frame["image"].shape[:2]
        spec = state["spec"]
        if previous is not None and (
            timestamp <= previous
            or timestamp - previous > spec["config"]["source_gap_seconds"] * 1_000_000_000
            or frame["sequence"] <= state["last_sequence"]
            or shape != state["last_shape"]
        ):
            state["analytics"] = type(state["analytics"])(spec)
            state["last_output"] = None
            if spec["capability"] == "person.tracking":
                self.backend.reset_tracker(self._item(spec))
        state.update(last_timestamp=timestamp, last_sequence=frame["sequence"], last_shape=shape)

    def process_frame(self, frame, session_ids):
        backend = self._backend()
        states = [self.sessions[sid] for sid in session_ids]
        if not states:
            return {}
        for state in states:
            if state["spec"]["camera_id"] != frame["camera_id"]:
                raise ValueError("Frame camera does not match the session")
            self._reset_on_discontinuity(state, frame)
        tracking = [state for state in states if state["spec"]["capability"] == "person.tracking"]
        trackers = [backend.get_tracker(self._item(state["spec"])) for state in tracking]
        thresholds = [state["spec"]["config"]["confidence_threshold"] for state in states
                      if state["spec"]["capability"] == "person.detection"]
        thresholds.extend(tracker.config.track_low_thresh for tracker in trackers)
        image = frame["image"]
        # Detect once for all submitted sessions, retaining low-confidence boxes for BoT-SORT.
        detections = backend.detect(image, min(thresholds))
        people = backend.convert_detections_to_people(detections)
        features = [None] * len(detections)
        reid_trackers = [tracker for tracker in trackers if tracker.config.with_reid]
        if reid_trackers:
            feature_tracker = min(reid_trackers, key=lambda tracker: tracker.config.track_high_thresh)
            features = backend.extract_reid_features_by_frame([image], [detections], [feature_tracker])[0]
        trackers_by_id = {state["spec"]["session_id"]: tracker for state, tracker in zip(tracking, trackers)}
        results = {}
        now = time.monotonic()
        for state in states:
            spec, analytics = state["spec"], state["analytics"]
            sid = spec["session_id"]
            analytics.ensure_zone_polygons(image)
            event_data = []
            tracks = None
            if spec["capability"] == "person.detection":
                filtered = analytics.filter_people_by_confidence(people)
                counts, filtered = analytics.count_people_in_zones(filtered)
                snapshot = {"people_count": len(filtered), "people": filtered, "zone_counts": counts,
                            "zones": [{"zone_id": analytics.zone_meta[name].get("camera_zone_id",
                                       analytics.zone_meta[name].get("zone_id")), "people_count": count}
                                      for name, count in counts.items()]}
                if spec["output"]["mode"] == "event" and analytics.should_publish_count_event(
                    {"event_payload": snapshot}, now=now
                ):
                    event_data.append(snapshot)
            else:
                tracker = trackers_by_id[sid]
                boxes, embeddings = backend.filter_tracker_roi(tracker, detections.copy(), list(features))
                tracked = tracker.update_from_detections(boxes, image, detection_features=embeddings)
                tracks = backend.convert_tracks(self._item(spec), tracker, tracked)
                annotations, tracked_people, counts, zones, crossings = analytics.analyze_trackers(
                    tracks, frame["timestamp_ns"] / 1_000_000_000
                )
                snapshot = {"people_count": len(tracked_people), "people": tracked_people,
                            "zone_counts": counts, "zones": zones}
                if spec["output"]["mode"] == "event":
                    event_data.extend(crossings)
            if spec["output"]["mode"] == "periodic" and (
                state["last_output"] is None or now - state["last_output"] >= spec["output"]["interval"]
            ):
                event_data.append(snapshot)
                state["last_output"] = now
            events = []
            if event_data:
                annotated = image.copy()
                if tracks is None:
                    analytics.draw_people(annotated, filtered)
                    analytics.draw_zone_counts(annotated, counts)
                else:
                    analytics.draw_trackers(annotated, tracks, annotations)
                analytics.draw_zones(annotated)
                events = [{"data": data, "image": annotated} for data in event_data]
            results[sid] = {"snapshot": snapshot, "events": events}
        return results

    def close_session(self, session_id):
        state = self.sessions.pop(session_id, None)
        if state is not None and self.backend is not None:
            self.backend.reset_tracker(self._item(state["spec"]))

    def close(self):
        for sid in list(self.sessions):
            self.close_session(sid)
        if self.backend is not None:
            self.backend.close()
        self.backend = None
        self.closed = True
