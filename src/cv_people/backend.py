"""CUDA detector/ReID with compact metadata for session tracking and analytics.

No source frame, crop, raw model output or unnormalized embedding is staged on
CPU. BoT-SORT and zone/event bookkeeping consume final boxes/features only.
"""
from contextlib import contextmanager
from dataclasses import fields
from threading import RLock

import cupy as cp
import numpy as np
import yaml

from .detection import PersonDetector
from .gpu import cuda_frame
from .reid import PersonReID
from .botsort_reid.botsort_tracker import BoTSORTConfig, YOLOv8BoTSORT, xyxy_polygon


class _PrecomputedFeatureEncoder:
    def inference(self, frame, bboxes):
        raise RuntimeError("The runtime must provide precomputed GPU ReID features")


class PeopleBackend:
    def __init__(self, config, *, enable_reid=True):
        self.config = config
        self.device = int(config.get("device", 0))
        self._model_lock = RLock()
        self.person_detector = self.person_reid = None
        self.trackers_by_service_key = {}
        self.tracker_data_by_service_key = {}
        with self._model_access():
            self.person_detector = PersonDetector(
                config["model"]["person"], self.device,
                size=config.get("detector_size", (640, 640)),
                layout=config.get("detector_layout", "BCN"),
                iou_threshold=config.get("iou_threshold", 0.4),
                max_people=config.get("max_people", 300),
                resize=config.get("detector_resize", "stretch"),
            )
            reid_path = config["model"].get("reid")
            if enable_reid and reid_path:
                self.person_reid = PersonReID(reid_path, self.device, config.get("reid_batch_size", 32))

    @contextmanager
    def _model_access(self):
        with self._model_lock, cp.cuda.Device(self.device):
            try:
                yield
            finally:
                # Complete work even on errors before ingestion recycles its
                # borrowed CUDA IPC frame or model bindings are released.
                cp.cuda.get_current_stream().synchronize()

    def validate_frame(self, image):
        with cp.cuda.Device(self.device):
            return cuda_frame(image, self.device)

    def infer(self, image, threshold, reid_threshold=None):
        """Shared CUDA detection/ReID; download only final tracker metadata."""
        with self._model_access():
            image = cuda_frame(image, self.device)
            boxes = self.person_detector.detect(image, threshold)
            features = [None] * len(boxes)
            if self.person_reid is not None and reid_threshold is not None:
                indices = cp.flatnonzero(boxes[:, 4] > reid_threshold)
                if len(indices):
                    embeddings = self.person_reid.extract(image, boxes[indices])
                    # Compact indices/normalized vectors only, never crops.
                    host_indices = cp.asnumpy(indices)
                    host_embeddings = cp.asnumpy(embeddings)
                    for index, vector in zip(host_indices, host_embeddings):
                        if np.any(vector):
                            features[int(index)] = vector
            return cp.asnumpy(boxes), features

    def detect(self, image, threshold):
        return self.infer(image, threshold)[0]

    @staticmethod
    def convert_detections_to_people(detections):
        people = []
        for detection in detections:
            box = [int(value) for value in detection[:4]]
            x1, y1, x2, y2 = box
            people.append({"box": box, "confidence": float(detection[4]),
                           "area": max(0, x2 - x1) * max(0, y2 - y1)})
        return people

    def close(self):
        with self._model_access():
            self.trackers_by_service_key.clear()
            self.tracker_data_by_service_key.clear()
            if self.person_reid is not None:
                self.person_reid.close()
            if self.person_detector is not None:
                self.person_detector.close()
            self.person_reid = self.person_detector = None

    def get_tracker(self, item):
        service_key = self.get_service_key(item)
        tracker = self.trackers_by_service_key.get(service_key)
        if tracker is not None:
            return tracker

        tracker_config = self.build_tracker_config(
            item.get("tracking_config") or self.config.get("model", {}).get("tracking_config"))
        if self.person_reid is None:
            tracker_config.with_reid = False
        tracker_config.frame_rate = item.get("fps", 5)
        tracker_config.person_class_id = self.person_detector.get_label_idx(["person"])[0]
        reid_encoder = _PrecomputedFeatureEncoder() if tracker_config.with_reid else None
        tracker = YOLOv8BoTSORT(
            reid_model_path=None,
            detector_model=self.config.get("model", {}).get("person", "person.engine"),
            config=tracker_config,
            detector=self.person_detector,
            reid=reid_encoder,
            reset_track_count=False,
        )
        self.trackers_by_service_key[service_key] = tracker
        self.tracker_data_by_service_key.setdefault(service_key, {})
        return tracker

    def reset_tracker(self, item):
        service_key = self.get_service_key(item)
        if not service_key:
            return

        self.trackers_by_service_key.pop(service_key, None)
        self.tracker_data_by_service_key.pop(service_key, None)

    def build_tracker_config(self, tracking_config_path):
        tracker_settings = {}
        if tracking_config_path:
            with open(tracking_config_path) as file:
                tracker_settings = yaml.safe_load(file) or {}

        if not isinstance(tracker_settings, dict):
            raise ValueError("Tracker YAML must contain an object")
        allowed_fields = {field.name for field in fields(BoTSORTConfig)}
        config_values = {
            key: value
            for key, value in tracker_settings.items()
            if key in allowed_fields
        }
        return BoTSORTConfig(**config_values)

    def filter_tracker_roi(self, tracker, detections, features):
        if tracker.roi is None or len(detections) == 0:
            return detections, features

        keep = []
        for detection in detections:
            polygon = xyxy_polygon(detection[:4])
            if polygon.area <= 0:
                keep.append(False)
                continue
            overlap = polygon.intersection(tracker.roi).area / polygon.area
            keep.append(
                overlap >= tracker.config.roi_min_overlap
                and polygon.intersects(tracker.roi)
            )

        keep = np.asarray(keep, dtype=bool)
        filtered_features = [
            feature
            for feature, should_keep in zip(features, keep)
            if should_keep
        ]
        return detections[keep], filtered_features

    def convert_tracks(self, item, tracker, track_results):
        service_key = self.get_service_key(item)
        tracker_data = self.tracker_data_by_service_key.setdefault(service_key, {})
        active_ids = set()
        tracks_by_id = {
            track.track_id: track
            for track in tracker.tracked_stracks
        }

        for result in track_results:
            track_id = int(result.track_id)
            active_ids.add(track_id)

            data = tracker_data.setdefault(track_id, {})
            data["bbox"] = tuple(float(value) for value in result.tlwh)
            data["score"] = result.score
            data["class_id"] = result.class_id
            data["iou_distance"] = result.iou_distance
            data["embedding_distance"] = result.embedding_distance

            track = tracks_by_id.get(track_id)
            if track is None:
                continue

            if track.smooth_feat is not None:
                smooth_feat = track.smooth_feat.copy()
                data.setdefault("ref_feat", smooth_feat)
                data["smooth_feat"] = smooth_feat

            if track.curr_feat is not None:
                data["curr_feat"] = track.curr_feat.copy()

        for track_id in list(tracker_data):
            if track_id not in active_ids:
                del tracker_data[track_id]

        return {
            track_id: data.copy()
            for track_id, data in tracker_data.items()
        }

    def get_service_key(self, item):
        service_key = item.get("service_key")
        if service_key:
            return service_key

        camera_id = item.get("camera_id")
        service_id = item.get("service_id")
        if camera_id is None or service_id is None:
            return None

        return f"{camera_id}_{service_id}"

