# encoding: utf-8
"""YOLOv8 + OSNet BoT-SORT tracker with Shapely IoU."""

from collections import deque
from dataclasses import dataclass, fields
import os
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, box
import yaml

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


__all__ = [
    "BoTSORTConfig",
    "TrackResult",
    "BoTSORTTrack",
    "PersonTracker",
    "YOLOv8BoTSORT",
    "BoTSORTTracker",
    "shapely_iou",
]


@dataclass
class BoTSORTConfig:
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    frame_rate: int = 30
    fuse_score: bool = True
    with_reid: bool = True
    person_class_id: int = 0
    roi: Optional[object] = None
    roi_min_overlap: float = 0.0


@dataclass
class TrackResult:
    track_id: int
    tlwh: np.ndarray
    tlbr: np.ndarray
    score: float
    class_id: int
    polygon: Polygon
    iou_distance: Optional[float]
    embedding_distance: Optional[float]

    @classmethod
    def from_track(cls, track: "BoTSORTTrack") -> "TrackResult":
        tlbr = track.tlbr
        return cls(
            track_id=track.track_id,
            tlwh=track.tlwh,
            tlbr=tlbr,
            score=track.score,
            class_id=track.class_id,
            polygon=xyxy_polygon(tlbr),
            iou_distance=track.iou_distance,
            embedding_distance=track.embedding_distance,
        )


class BoTSORTTrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, class_id=0, feat=None, feat_history=50):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = float(score)
        self.class_id = int(class_id)
        self.tracklet_len = 0
        self.iou_distance = None
        self.embedding_distance = None

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        feat = np.asarray(feat, dtype=float).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        smooth_norm = np.linalg.norm(self.smooth_feat)
        if smooth_norm > 0:
            self.smooth_feat = self.smooth_feat / smooth_norm
        self.features.append(feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, track in enumerate(stracks):
            if track.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BoTSORTTrack.shared_kalman.multi_predict(
            multi_mean,
            multi_covariance,
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.iou_distance = None
        self.embedding_distance = None

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xywh(new_track.tlwh),
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xywh(new_track.tlwh),
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=float).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class YOLOv8BoTSORT:
    """BoT-SORT tracker using installed detection and OSNet TensorRT packages."""

    def __init__(
        self,
        reid_model_path: Optional[str],
        detector_model: str = "yolov8s.pt",
        config: Optional[BoTSORTConfig] = None,
        device: Optional[str] = None,
        detector=None,
        reid=None,
        reset_track_count: bool = True,
    ):
        self.config = config or BoTSORTConfig()
        self.device = device
        self.detector = detector or self._load_detector(detector_model)

        if self.config.with_reid:
            if reid is not None:
                self.encoder = reid
            elif reid_model_path:
                self.encoder = PersonReIDBatchEncoder(reid_model_path, device=device)
            else:
                raise ValueError("reid_model_path or reid must be provided when with_reid=True.")
        else:
            self.encoder = None

        self.roi = self._coerce_roi(self.config.roi)
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        if reset_track_count:
            BaseTrack.clear_count()

        self.frame_id = 0
        self.max_time_lost = int(
            self.config.frame_rate / 30.0 * self.config.track_buffer
        )
        self.kalman_filter = KalmanFilter()

    def _load_detector(self, detector_model):
        from person_inference import Detector
        return Detector(detector_model)

    def detect(self, frame) -> np.ndarray:
        result = self.detector.detect_n_seg(
            frame, labels=["person"], score_threshold=self.config.track_low_thresh,
        )
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=float)
        detections = np.asarray(result.boxes, dtype=float)[:, :6]
        self.config.person_class_id = self.detector.get_label_idx(["person"])[0]
        return self._filter_roi(detections)

    def update(self, frame) -> List[TrackResult]:
        detections = self.detect(frame)
        return self.update_from_detections(detections, frame)

    def update_from_detections(self, detections, frame, detection_features=None) -> List[TrackResult]:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes, scores, classes = self._parse_detections(detections)
        detection_features = self._coerce_detection_features(detection_features, len(scores))

        if len(scores):
            keep_low = scores > self.config.track_low_thresh
            bboxes = bboxes[keep_low]
            scores = scores[keep_low]
            classes = classes[keep_low]
            if detection_features is not None:
                detection_features = [
                    feature
                    for feature, keep in zip(detection_features, keep_low)
                    if keep
                ]

            keep_high = scores > self.config.track_high_thresh
            high_bboxes = bboxes[keep_high]
            high_scores = scores[keep_high]
            high_classes = classes[keep_high]
            if detection_features is not None:
                high_features = [
                    feature
                    for feature, keep in zip(detection_features, keep_high)
                    if keep
                ]
            else:
                high_features = self._extract_features(frame, high_bboxes)
        else:
            high_bboxes = np.empty((0, 4), dtype=float)
            high_scores = np.empty((0,), dtype=float)
            high_classes = np.empty((0,), dtype=float)
            high_features = None

        detections_high = self._make_tracks(high_bboxes, high_scores, high_classes, high_features)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        BoTSORTTrack.multi_predict(strack_pool)

        frame_shape = frame.shape if frame is not None else None
        dists, iou_debug_dists, embedding_debug_dists = self._association_distance_components(
            strack_pool,
            detections_high,
            frame_shape,
        )
        matches, u_track, u_detection = linear_assignment(dists, self.config.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            self._set_match_debug(
                track,
                iou_debug_dists[itracked, idet],
                embedding_debug_dists[itracked, idet],
            )

        if len(scores):
            second_mask = np.logical_and(
                scores < self.config.track_high_thresh,
                scores > self.config.track_low_thresh,
            )
            second_bboxes = bboxes[second_mask]
            second_scores = scores[second_mask]
            second_classes = classes[second_mask]
        else:
            second_bboxes = np.empty((0, 4), dtype=float)
            second_scores = np.empty((0,), dtype=float)
            second_classes = np.empty((0,), dtype=float)

        detections_second = self._make_tracks(second_bboxes, second_scores, second_classes)
        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = shapely_iou_distance(r_tracked_stracks, detections_second)
        matches, u_track_second, _ = linear_assignment(dists, 0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            self._set_match_debug(track, dists[itracked, idet], None)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections_unconfirmed = [detections_high[i] for i in u_detection]
        dists, iou_debug_dists, embedding_debug_dists = self._association_distance_components(
            unconfirmed,
            detections_unconfirmed,
            frame_shape,
        )
        matches, u_unconfirmed, u_detection = linear_assignment(dists, 0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unconfirmed[idet], self.frame_id)
            self._set_match_debug(
                unconfirmed[itracked],
                iou_debug_dists[itracked, idet],
                embedding_debug_dists[itracked, idet],
            )
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections_unconfirmed[inew]
            if track.score < self.config.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            track for track in self.tracked_stracks if track.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # The previous frame's removed tracks have now been applied to the lost
        # list, so retaining older entries serves no tracking purpose and keeps
        # their ReID feature histories alive indefinitely.
        self.removed_stracks = removed_stracks
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
        )

        return [TrackResult.from_track(track) for track in self.tracked_stracks]

    def _parse_detections(self, detections):
        detections = np.asarray(detections, dtype=float)
        if detections.size == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        if detections.ndim != 2 or detections.shape[1] < 5:
            raise ValueError("Detections must be an Nx5 or Nx6 array: x1,y1,x2,y2,score[,class].")

        bboxes = detections[:, :4]
        scores = detections[:, 4]
        if detections.shape[1] >= 6:
            classes = detections[:, 5]
        else:
            classes = np.full((len(detections),), self.config.person_class_id, dtype=float)
        return bboxes, scores, classes

    def _coerce_detection_features(self, detection_features, detection_count):
        if detection_features is None:
            return None

        detection_features = list(detection_features)
        if len(detection_features) != detection_count:
            raise ValueError("detection_features must match the number of detections.")

        return detection_features

    def _extract_features(self, frame, bboxes):
        if not self.config.with_reid or self.encoder is None or len(bboxes) == 0:
            return None
        return self.encoder.inference(frame, bboxes)

    def _make_tracks(self, bboxes, scores, classes, features=None):
        if len(bboxes) == 0:
            return []

        if features is None:
            features = [None] * len(bboxes)
        return [
            BoTSORTTrack(
                BoTSORTTrack.tlbr_to_tlwh(tlbr),
                score,
                class_id=class_id,
                feat=feat,
            )
            for tlbr, score, class_id, feat in zip(bboxes, scores, classes, features)
        ]

    def _first_association_distance(self, tracks, detections, frame_shape=None):
        cost, _, _ = self._association_distance_components(tracks, detections, frame_shape)
        return cost

    def _unconfirmed_distance(self, tracks, detections, frame_shape=None):
        cost, _, _ = self._association_distance_components(tracks, detections, frame_shape)
        return cost

    def _association_distance_components(self, tracks, detections, frame_shape=None):
        raw_iou_dists = shapely_iou_distance(tracks, detections)
        iou_dists = raw_iou_dists.copy()

        if self.config.fuse_score:
            iou_dists = fuse_score(iou_dists, detections)

        if not self.config.with_reid or self.encoder is None:
            embedding_debug_dists = np.full_like(raw_iou_dists, np.nan, dtype=float)
            return iou_dists, raw_iou_dists, embedding_debug_dists

        emb_dists = embedding_distance(tracks, detections) / 2.0
        embedding_debug_dists = emb_dists.copy()
        missing_track_features = [
            i for i, track in enumerate(tracks) if track.smooth_feat is None
        ]
        missing_detection_features = [
            i for i, detection in enumerate(detections) if detection.curr_feat is None
        ]
        if missing_track_features:
            embedding_debug_dists[missing_track_features, :] = np.nan
        if missing_detection_features:
            embedding_debug_dists[:, missing_detection_features] = np.nan

        emb_dists[emb_dists > self.config.appearance_thresh] = 1.0

        usual_emb_dists = emb_dists.copy()
        iou_mask = iou_dists > self.config.proximity_thresh
        usual_emb_dists[iou_mask] *= 1.7

        usual_cost = (0.3 * iou_dists) + (0.7 * usual_emb_dists)
        object_dists = normalized_center_distance(tracks, detections, frame_shape)
        no_overlap_cost = usual_cost + (3.0 * object_dists)
        no_overlap_mask = raw_iou_dists >= 1.0 - 1e-9
        cost = np.where(no_overlap_mask, no_overlap_cost, usual_cost)

        return cost, raw_iou_dists, embedding_debug_dists

    def _set_match_debug(self, track, iou_distance, embedding_distance_value):
        track.iou_distance = self._debug_float(iou_distance)
        track.embedding_distance = self._debug_float(embedding_distance_value)

    def _debug_float(self, value):
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    def _coerce_roi(self, roi):
        if roi is None:
            return None
        if isinstance(roi, Polygon):
            return roi
        return Polygon(roi)

    def _filter_roi(self, detections):
        if self.roi is None or len(detections) == 0:
            return detections

        keep = []
        for det in detections:
            polygon = xyxy_polygon(det[:4])
            if polygon.area <= 0:
                keep.append(False)
                continue
            overlap = polygon.intersection(self.roi).area / polygon.area
            keep.append(overlap >= self.config.roi_min_overlap and polygon.intersects(self.roi))
        return detections[np.asarray(keep, dtype=bool)]


class PersonReIDBatchEncoder:
    def __init__(self, model_path, device=None):
        from osnet_reid_trt import ReIDTRT
        self.model = ReIDTRT(model_path)

    def inference(self, frame, bboxes):
        features = [None] * len(bboxes)
        crops = []
        crop_indexes = []
        frame_h, frame_w = frame.shape[:2]

        for index, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(frame_w, x1))
            x2 = max(0, min(frame_w, x2))
            y1 = max(0, min(frame_h, y1))
            y2 = max(0, min(frame_h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crops.append(crop)
            crop_indexes.append(index)

        crop_features = self.model.extract_features(crops)
        for index, feature in zip(crop_indexes, crop_features):
            features[index] = feature

        return features


class PersonTracker:
    """Compatibility wrapper for AI runtimes that expect PersonTracker.track()."""

    def __init__(self, config):
        self.config = config
        self.trackers = {}

        model_config = self.config.get("model", {})
        tracker_config = self._build_tracker_config(model_config.get("tracking_config"))

        self.tracker = YOLOv8BoTSORT(
            reid_model_path=model_config.get("reid"),
            detector_model=model_config.get("person", "yolov8s.pt"),
            config=tracker_config,
        )

    def track(self, frame):
        results = self.tracker.update(frame)
        active_ids = set()
        tracks_by_id = {
            track.track_id: track
            for track in self.tracker.tracked_stracks
        }

        for result in results:
            track_id = int(result.track_id)
            active_ids.add(track_id)

            data = self.trackers.setdefault(track_id, {})
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

        for track_id in list(self.trackers):
            if track_id not in active_ids:
                del self.trackers[track_id]

        return self.trackers

    def draw_trackers(self, frame):
        try:
            import cv2
        except ImportError:
            return frame

        for track_id, data in self.trackers.items():
            x, y, w, h = map(int, data["bbox"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame

    def _build_tracker_config(self, tracking_config_path):
        tracker_settings = {}
        if tracking_config_path and os.path.exists(tracking_config_path):
            with open(tracking_config_path) as file:
                tracker_settings = yaml.safe_load(file) or {}

        allowed_fields = {field.name for field in fields(BoTSORTConfig)}
        config_values = {
            key: value
            for key, value in tracker_settings.items()
            if key in allowed_fields
        }

        return BoTSORTConfig(**config_values)


def xyxy_polygon(tlbr) -> Polygon:
    x1, y1, x2, y2 = np.asarray(tlbr, dtype=float)[:4]
    if x2 <= x1 or y2 <= y1:
        return Polygon()
    return box(x1, y1, x2, y2)


def shapely_iou(a_tlbr, b_tlbr) -> float:
    a = xyxy_polygon(a_tlbr)
    b = xyxy_polygon(b_tlbr)
    if a.is_empty or b.is_empty:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(a.intersection(b).area / union)


def shapely_ious(a_boxes, b_boxes) -> np.ndarray:
    ious = np.zeros((len(a_boxes), len(b_boxes)), dtype=float)
    if ious.size == 0:
        return ious

    a_polygons = [xyxy_polygon(box_) for box_ in a_boxes]
    b_polygons = [xyxy_polygon(box_) for box_ in b_boxes]
    for i, a_poly in enumerate(a_polygons):
        for j, b_poly in enumerate(b_polygons):
            if a_poly.is_empty or b_poly.is_empty:
                continue
            union = a_poly.union(b_poly).area
            if union > 0:
                ious[i, j] = a_poly.intersection(b_poly).area / union
    return ious


def shapely_iou_distance(atracks, btracks) -> np.ndarray:
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=float)

    atlbrs = _as_tlbrs(atracks)
    btlbrs = _as_tlbrs(btracks)
    return 1.0 - shapely_ious(atlbrs, btlbrs)


def normalized_center_distance(atracks, btracks, frame_shape=None) -> np.ndarray:
    distances = np.zeros((len(atracks), len(btracks)), dtype=float)
    if distances.size == 0:
        return distances

    atlbrs = np.asarray(_as_tlbrs(atracks), dtype=float)
    btlbrs = np.asarray(_as_tlbrs(btracks), dtype=float)
    a_centers = np.column_stack(
        (
            (atlbrs[:, 0] + atlbrs[:, 2]) / 2.0,
            (atlbrs[:, 1] + atlbrs[:, 3]) / 2.0,
        )
    )
    b_centers = np.column_stack(
        (
            (btlbrs[:, 0] + btlbrs[:, 2]) / 2.0,
            (btlbrs[:, 1] + btlbrs[:, 3]) / 2.0,
        )
    )
    width, height = _center_distance_scale(frame_shape, atlbrs, btlbrs)

    dx = (b_centers[None, :, 0] - a_centers[:, None, 0]) / width
    dy = (b_centers[None, :, 1] - a_centers[:, None, 1]) / height
    return np.sqrt(dx**2 + dy**2)


def embedding_distance(tracks, detections, metric="cosine") -> np.ndarray:
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix

    missing_tracks = [i for i, track in enumerate(tracks) if track.smooth_feat is None]
    missing_dets = [i for i, det in enumerate(detections) if det.curr_feat is None]
    if len(missing_tracks) == len(tracks) or len(missing_dets) == len(detections):
        return np.ones_like(cost_matrix)

    valid_tracks = [i for i in range(len(tracks)) if i not in missing_tracks]
    valid_dets = [i for i in range(len(detections)) if i not in missing_dets]
    track_features = np.asarray([tracks[i].smooth_feat for i in valid_tracks], dtype=float)
    det_features = np.asarray([detections[i].curr_feat for i in valid_dets], dtype=float)

    cost_matrix.fill(1.0)
    valid_costs = np.maximum(0.0, cdist(track_features, det_features, metric))
    for row_idx, row in enumerate(valid_tracks):
        for col_idx, col in enumerate(valid_dets):
            cost_matrix[row, col] = valid_costs[row_idx, col_idx]
    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.asarray([det.score for det in detections], dtype=float)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    return 1.0 - iou_sim * det_scores


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=1e5)
    rows, cols = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for row, col in zip(rows, cols):
        if cost_matrix[row, col] <= thresh:
            matches.append([row, col])
            unmatched_rows.discard(row)
            unmatched_cols.discard(col)

    return (
        np.asarray(matches, dtype=int),
        np.asarray(sorted(unmatched_rows), dtype=int),
        np.asarray(sorted(unmatched_cols), dtype=int),
    )


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for track in tlista:
        exists[track.track_id] = 1
        res.append(track)
    for track in tlistb:
        if not exists.get(track.track_id, 0):
            exists[track.track_id] = 1
            res.append(track)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {track.track_id: track for track in tlista}
    for track in tlistb:
        stracks.pop(track.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = shapely_iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [track for i, track in enumerate(stracksa) if i not in dupa]
    resb = [track for i, track in enumerate(stracksb) if i not in dupb]
    return resa, resb


def _as_tlbrs(items) -> List[np.ndarray]:
    if len(items) == 0:
        return []
    if isinstance(items[0], np.ndarray):
        return items
    return [item.tlbr for item in items]


def _center_distance_scale(frame_shape, a_boxes, b_boxes):
    if frame_shape is not None and len(frame_shape) >= 2:
        height = float(frame_shape[0])
        width = float(frame_shape[1])
        if width > 0.0 and height > 0.0:
            return width, height

    boxes = np.vstack((a_boxes, b_boxes))
    width = max(1.0, float(np.nanmax(boxes[:, [0, 2]])))
    height = max(1.0, float(np.nanmax(boxes[:, [1, 3]])))
    return width, height


BoTSORTTracker = YOLOv8BoTSORT
