"""GPU frame preprocessing, batched detection, NMS, crops and ArcFace."""
import math
from pathlib import Path

import cupy as cp

from .database import FaceSearch
from .gpu import face_crops, letterbox, nms
from .trt import TensorRTEngine


class FaceBackend:
    def __init__(self, config):
        self.device = int(config.get("device", 0))
        self.detector_path = config["model"]["face_detection"]["engine"]
        # Match person-inference: class IDs follow line order in model.txt
        # beside model.engine/model.trt. Ignore trailing empty lines only.
        labels_path = Path(self.detector_path).with_suffix(".txt")
        self.classes = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        while self.classes and not self.classes[-1]:
            self.classes.pop()
        if not self.classes or any(not name for name in self.classes):
            raise ValueError(f"{labels_path} must contain one non-empty class name per line")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"{labels_path} must contain unique class names")
        self.face_label = config.get("face_label", "face")
        if not isinstance(self.face_label, str) or self.face_label not in self.classes:
            raise ValueError(f"Face label {self.face_label!r} not found in {labels_path}")
        self.face_class_id = self.classes.index(self.face_label)
        self.arcface_path = config["model"]["face_extraction"]["engine"]
        self.size = tuple(config.get("detector_size", [640, 640]))
        if len(self.size) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.size):
            raise ValueError("detector_size must contain positive height and width")
        self.score = float(config.get("score_threshold", 0.7))
        self.iou = float(config.get("iou_threshold", 0.4))
        if not 0 <= self.score <= 1 or not 0 <= self.iou <= 1:
            raise ValueError("Detection thresholds must be in [0, 1]")
        self.max_faces = int(config.get("max_faces", 100))
        self.batch_size = int(config.get("batch_size", 8))
        self.minimum_area = float(config.get("minimum_face_size", 0))
        self.margin = float(config.get("crop_margin", 0.2))
        if (self.max_faces <= 0 or self.batch_size <= 0
                or not 0 <= self.margin <= 1 or not 0 <= self.minimum_area < float("inf")):
            raise ValueError("Invalid batch_size, max_faces, crop_margin or minimum_face_size")
        self.layout = config.get("detector_layout", "BCN")
        if self.layout not in ("BCN", "BNC"):
            raise ValueError("detector_layout must be BCN or BNC")
        self.search = FaceSearch(config["db"])
        self.detector = self.arcface = None

    def _load(self):
        if self.detector is None:
            # Initialize on the CUDA primary context used by ingestion.
            self.detector = TensorRTEngine(self.detector_path, self.device)
        if self.arcface is None:
            self.arcface = TensorRTEngine(self.arcface_path, self.device)

    def _frame_array(self, image):
        if not hasattr(image, "__cuda_array_interface__"):
            raise ValueError("nba.face requires a CUDA BGR frame; CPU frames are unsupported")
        # CuPy arrays are used directly; other CUDA array providers are wrapped.
        # Do not request dtype/layout changes that could copy the source frame.
        array = image if isinstance(image, cp.ndarray) else cp.asarray(image)
        if (array.device.id != self.device or array.dtype != cp.uint8
                or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0):
            raise ValueError("Expected a non-empty HWC uint8 CUDA BGR frame on the configured device")
        if array.data.ptr != image.__cuda_array_interface__["data"][0]:
            raise ValueError("CUDA input must support wrapping without copying its storage")
        return array

    def _detections(self, prediction, image, transform, *, max_faces=None):
        # Match person-inference: choose the best class per anchor, then filter
        # by the requested label's ID. Raw YOLOv8 has no objectness channel.
        if self.layout == "BCN":
            prediction = prediction.T
        expected = 4 + len(self.classes)
        if prediction.ndim != 2 or prediction.shape[1] != expected:
            raise ValueError(
                f"Detector must output raw YOLOv8 [B,{expected},N] or [B,N,{expected}] "
                f"for {len(self.classes)} classes in its .txt file"
            )
        prediction = prediction.astype(cp.float32, copy=False)
        prediction = prediction[cp.isfinite(prediction).all(1)]
        class_scores = prediction[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = cp.take_along_axis(class_scores, class_ids[:, None], axis=1)[:, 0]
        selected = (class_ids == self.face_class_id) & (scores >= self.score)
        prediction = cp.concatenate((prediction[selected, :4], scores[selected][:, None]), axis=1)
        prediction = prediction[cp.argsort(-prediction[:, 4])[:3000]]
        boxes = cp.concatenate((prediction[:, :2] - prediction[:, 2:4] / 2,
                                prediction[:, :2] + prediction[:, 2:4] / 2), axis=1)
        left, top, scale_x, scale_y = transform
        boxes[:, 0::2] = (boxes[:, 0::2] - left) / scale_x
        boxes[:, 1::2] = (boxes[:, 1::2] - top) / scale_y
        height, width = image.shape[:2]
        cp.clip(boxes[:, 0::2], 0, width, out=boxes[:, 0::2])
        cp.clip(boxes[:, 1::2], 0, height, out=boxes[:, 1::2])
        sizes = boxes[:, 2:] - boxes[:, :2]
        valid = (sizes > 0).all(1) & (sizes.prod(1) >= self.minimum_area)
        boxes, scores = boxes[valid], prediction[valid, 4]
        keep = nms(boxes, self.iou, self.max_faces if max_faces is None else max_faces)
        return boxes[keep], scores[keep]

    def infer_batch(self, frames):
        """Return one face list per CUDA frame, batching both TRT models.

        Only final 512D embeddings and compact metadata are downloaded for
        PostgreSQL search and JSON output. No image pixels leave GPU memory.
        """
        if not frames:
            return []
        with cp.cuda.Device(self.device):
            try:
                return self._infer_batch(frames)
            finally:
                # Also finish kernels on errors, before borrowed ingestion frame
                # storage can be recycled by the caller.
                cp.cuda.get_current_stream().synchronize()

    def _infer_batch(self, frames):
        results = [[] for _ in frames]
        self._load()
        for start in range(0, len(frames), self.batch_size):
            images = [self._frame_array(frame) for frame in frames[start:start + self.batch_size]]
            prepared = [letterbox(image, self.size) for image in images]
            predictions = self.detector.infer_batch(cp.concatenate([item[0] for item in prepared]))
            crops, metadata = [], []
            for index, (image, prediction, (_, transform)) in enumerate(zip(images, predictions, prepared)):
                boxes, scores = self._detections(prediction, image, transform)
                if not len(boxes):
                    continue
                sizes = boxes[:, 2:] - boxes[:, :2]
                expanded = cp.concatenate((boxes[:, :2] - sizes * self.margin,
                                           boxes[:, 2:] + sizes * self.margin), axis=1)
                cp.clip(expanded[:, 0::2], 0, image.shape[1], out=expanded[:, 0::2])
                cp.clip(expanded[:, 1::2], 0, image.shape[0], out=expanded[:, 1::2])
                # Fused GPU bilinear sampling, BGR->RGB and normalization.
                crops.append(face_crops(image, expanded))
                metadata.append(cp.concatenate((cp.full((len(scores), 1), start + index, dtype=cp.float32),
                                                boxes, scores[:, None]), axis=1))
            if not crops:
                continue
            raw = self.arcface.infer_batch(cp.concatenate(crops))
            embeddings = raw.astype(cp.float32, copy=False).reshape(len(raw), -1)
            if embeddings.shape[1] != 512:
                raise ValueError("ArcFace must output 512D embeddings for identity.embedding")
            norms = cp.linalg.norm(embeddings, axis=1, keepdims=True)
            valid = cp.isfinite(embeddings).all(1) & cp.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
            normalized = embeddings / cp.maximum(norms, 1e-12)
            # Only search vectors and result metadata are explicitly downloaded.
            # GPU compaction may synchronize counts, never frame/crop pixels.
            packed = cp.concatenate((cp.concatenate(metadata), normalized), axis=1)[valid]
            rows = cp.asnumpy(packed).tolist()
            matches = self.search.search_batch([row[6:] for row in rows])
            for row, match in zip(rows, matches):
                frame_index = int(row[0])
                box = row[1:5]
                height, width = frames[frame_index].shape[:2]
                area = (box[2] - box[0]) * (box[3] - box[1])
                ratio = area / (height * width)
                results[frame_index].append({
                    "box": [int(v) for v in box], "confidence": row[5],
                    "face_area": area, "face_area_ratio": ratio,
                    "estimated_distance_cm": 60 * (0.065 / ratio) ** 0.5 if ratio > 0 else None,
                    "identity_id": match["identity_id"], "employee_id": match["employee_id"],
                    "image_url": match["image_url"], "similarity": match["similarity"],
                    "pose": match["pose"],
                })
        return results

    def registration_embedding(self, image, crop_type):
        """Upload an enrollment image and return its embedding and CPU crop box."""
        with cp.cuda.Device(self.device):
            try:
                self._load()
                frame = cp.asarray(image)
                tensor, transform = letterbox(frame, self.size)
                prediction = self.detector.infer_batch(tensor)[0]
                # Detect at least two faces even when runtime max_faces is one,
                # so an ambiguous enrollment image is always rejected.
                boxes, _ = self._detections(prediction, frame, transform,
                                            max_faces=max(2, self.max_faces))
                if len(boxes) != 1:
                    raise ValueError("Registration image must contain exactly one encodable face")
                height, width = image.shape[:2]
                if crop_type == "full":
                    bounds = (0, 0, width, height)
                else:
                    box = boxes[0]
                    if crop_type == "loose":
                        size = box[2:] - box[:2]
                        box = cp.concatenate((box[:2] - size * 0.25, box[2:] + size * 0.25))
                    x1, y1, x2, y2 = cp.asnumpy(box).tolist()
                    bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                              min(width, math.ceil(x2)), min(height, math.ceil(y2)))
                x1, y1, x2, y2 = bounds
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Detected face crop is empty")
                roi = cp.asarray([bounds], dtype=cp.float32)
                raw = self.arcface.infer_batch(face_crops(frame, roi))
                embedding = raw.astype(cp.float32, copy=False).reshape(-1)
                if embedding.size != 512:
                    raise ValueError("ArcFace must output 512D embeddings for identity.embedding")
                norm = cp.linalg.norm(embedding)
                if not bool(cp.isfinite(embedding).all() & cp.isfinite(norm) & (norm > 0)):
                    raise ValueError("Registration image must contain exactly one encodable face")
                return cp.asnumpy(embedding / norm).tolist(), bounds
            finally:
                cp.cuda.get_current_stream().synchronize()

    def close(self):
        self.search.close()
        if self.detector is not None or self.arcface is not None:
            with cp.cuda.Device(self.device):
                cp.cuda.get_current_stream().synchronize()
                self.detector = self.arcface = None
