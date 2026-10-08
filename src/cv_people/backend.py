"""Shared detector/ReID and session-local BoT-SORT, adapted from the workers."""
import logging
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from threading import RLock
import numpy as np
import yaml
from person_inference import Detector
from .botsort_reid.botsort_tracker import BoTSORTConfig, YOLOv8BoTSORT, xyxy_polygon


class _PrecomputedFeatureEncoder:
    def inference(self, frame, bboxes):
        raise RuntimeError("The runtime must provide precomputed ReID features")

class PeopleBackend:
    def __init__(self, config, *, enable_reid=True):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._model_lock = RLock()
        self._cuda_context = None
        reid_path = config["model"].get("reid")
        if Path(config["model"]["person"]).suffix.lower() == ".trt" or (enable_reid and reid_path):
            # Both TensorRT dependencies use autoinit. Its import may have run
            # on another thread; activate that same context before creating
            # engines/streams, and again for every call using their resources.
            import pycuda.autoinit
            self._cuda_context = pycuda.autoinit.context
        self.person_reid = None
        with self._model_access():
            self.person_detector = Detector(config["model"]["person"])
            if enable_reid and reid_path:
                from osnet_reid_trt import ReIDTRT
                self.person_reid = ReIDTRT(reid_path)
        self.trackers_by_service_key = {}
        self.tracker_data_by_service_key = {}

    @contextmanager
    def _model_access(self):
        with self._model_lock:
            if self._cuda_context is None:
                yield
            else:
                self._cuda_context.push()
                try:
                    yield
                finally:
                    self._cuda_context.pop()

    def close(self):
        with self._model_access():
            self.trackers_by_service_key.clear()
            self.tracker_data_by_service_key.clear()
            self.person_reid = None
            self.person_detector = None

    def detect(self, image, threshold):
        with self._model_access():
            result = self.person_detector.detect_n_seg(
                image, labels=["person"], score_threshold=threshold,
            )
        return self.convert_detection_result(result)

    def convert_detections_to_people(self, detections):
        people = []
        for detection in detections:
            box = [int(value) for value in detection[:4]]
            x1, y1, x2, y2 = box
            people.append({
                "box": box,
                "confidence": float(detection[4]),
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            })
        return people

    def get_tracker(self, item):
        service_key = self.get_service_key(item)
        tracker = self.trackers_by_service_key.get(service_key)
        if tracker is not None:
            return tracker

        tracker_config = self.build_tracker_config(item.get("tracking_config"))
        if self.person_reid is None:
            tracker_config.with_reid = False
        tracker_config.frame_rate = item.get("fps", 5)
        tracker_config.person_class_id = self.person_detector.get_label_idx(["person"])[0]
        reid_encoder = _PrecomputedFeatureEncoder() if tracker_config.with_reid else None
        tracker = YOLOv8BoTSORT(
            reid_model_path=None,
            detector_model=self.config.get("model", {}).get("person", "yolov8s.pt"),
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

    def convert_detection_result(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=float)
        boxes = np.asarray(result.boxes, dtype=float)[:, :6]
        keep = np.asarray([
            result.names[int(class_id)] == "person" for class_id in boxes[:, 5]
        ], dtype=bool)
        return boxes[keep]

    def extract_reid_features_by_frame(self, frames, detections_by_frame, trackers):
        features_by_frame = [
            [None] * len(detections)
            for detections in detections_by_frame
        ]
        if self.person_reid is None:
            return features_by_frame

        crop_contexts = []
        crops = []
        for frame_index, (frame, detections, tracker) in enumerate(
            zip(frames, detections_by_frame, trackers)
        ):
            if not tracker.config.with_reid:
                continue

            for detection_index, detection in enumerate(detections):
                if detection[4] <= tracker.config.track_high_thresh:
                    continue

                crop = self.crop_person(frame, detection[:4])
                if crop is None:
                    continue

                crop_contexts.append((frame_index, detection_index))
                crops.append(crop)

        crop_features = self.extract_reid_batch(crops)
        for context, feature in zip(crop_contexts, crop_features):
            frame_index, detection_index = context
            features_by_frame[frame_index][detection_index] = feature

        return features_by_frame

    def extract_reid_batch(self, crops):
        if not crops:
            return []

        with self._model_access():
            return self._extract_reid_batch(crops)

    def _extract_reid_batch(self, crops):
        try:
            return self.person_reid.extract_features(crops)
        except Exception as exc:
            self.logger.info(f"Batch people ReID failed, falling back to single crops: {exc}")

        features = [None] * len(crops)
        for index, crop in enumerate(crops):
            try:
                features[index] = self.to_feature_numpy(self.person_reid.extract_feature(crop))
            except Exception as exc:
                self.logger.info(f"People ReID crop failed: {exc}")

        return features

    def crop_person(self, frame, box):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in box[:4]]
        x1 = max(0, min(frame_width, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height, y1))
        y2 = max(0, min(frame_height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

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

    def to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def to_feature_numpy(self, feature):
        if feature is None:
            return None
        feature = self.to_numpy(feature)
        feature = np.asarray(feature, dtype=float).reshape(-1)
        if feature.size == 0:
            return None
        return feature
