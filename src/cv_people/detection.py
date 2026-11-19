"""Package-owned raw YOLOv8 TensorRT detection; all numerical work is CUDA."""
import json
from pathlib import Path

import cupy as cp

from .gpu import letterbox, nms
from .trt import TensorRTEngine


class PersonDetector:
    def __init__(self, path, device=0, *, size=(640, 640), layout="BCN",
                 iou_threshold=0.4, max_people=300, resize="stretch",
                 classes=None, manifest=None):
        self.device = device
        self.size = tuple(size)
        if len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError("detector_size must contain positive height and width")
        self.layout = layout
        if resize not in ("stretch", "letterbox"):
            raise ValueError("detector_resize must be stretch or letterbox")
        self.resize = resize
        self.iou = float(iou_threshold)
        self.limit = int(max_people)
        if layout not in ("BCN", "BNC") or not 0 <= self.iou <= 1 or self.limit <= 0:
            raise ValueError("Invalid detector_layout, iou_threshold or max_people")
        if classes is None:
            manifest_path = Path(manifest) if manifest is not None else Path(path).parent / "manifest.json"
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"{manifest_path} must contain a JSON object")
            classes = metadata.get("classes")
        if (not isinstance(classes, list) or not classes
                or any(not isinstance(name, str) or not name.strip() for name in classes)
                or len(set(classes)) != len(classes) or "person" not in classes):
            raise ValueError("Detection model classes must be an array of unique non-empty strings including person")
        self.classes = list(classes)
        self.person_class_id = self.classes.index("person")
        self.engine = TensorRTEngine(path, device)

    def get_label_idx(self, labels):
        return [self.classes.index(label) for label in labels]

    def detect(self, image, threshold):
        """Return CUDA [N,6] xyxy/score/class; input storage remains borrowed."""
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        tensor, (left, top, scale_x, scale_y) = letterbox(
            image, self.size, preserve_aspect=self.resize == "letterbox")
        prediction = self.engine.infer_batch(tensor)[0]
        if self.layout == "BCN":
            prediction = prediction.T
        if prediction.ndim != 2 or prediction.shape[1] != 4 + len(self.classes):
            raise ValueError("Expected raw YOLOv8 boxes and class scores matching the model classes")
        prediction = prediction.astype(cp.float32, copy=False)
        prediction = prediction[cp.isfinite(prediction).all(1)]
        class_scores = prediction[:, 4:]
        class_ids = class_scores.argmax(1)
        scores = cp.take_along_axis(class_scores, class_ids[:, None], axis=1)[:, 0]
        selected = (class_ids == self.person_class_id) & (scores >= threshold)
        prediction, scores = prediction[selected], scores[selected]
        order = cp.argsort(-scores)[:3000]
        prediction, scores = prediction[order], scores[order]
        boxes = cp.concatenate((prediction[:, :2] - prediction[:, 2:4] / 2,
                                prediction[:, :2] + prediction[:, 2:4] / 2), axis=1)
        boxes[:, 0::2] = (boxes[:, 0::2] - left) / scale_x
        boxes[:, 1::2] = (boxes[:, 1::2] - top) / scale_y
        cp.clip(boxes[:, 0::2], 0, image.shape[1], out=boxes[:, 0::2])
        cp.clip(boxes[:, 1::2], 0, image.shape[0], out=boxes[:, 1::2])
        valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
        boxes, scores = boxes[valid], scores[valid]
        keep = nms(boxes, self.iou, self.limit)
        return cp.concatenate((boxes[keep], scores[keep, None],
                               cp.full((len(keep), 1), self.person_class_id, dtype=cp.float32)), axis=1)

    def close(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = None
