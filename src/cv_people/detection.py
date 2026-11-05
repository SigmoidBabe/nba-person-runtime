"""Package-owned raw YOLOv8 TensorRT detection; all numerical work is CUDA."""
from pathlib import Path

import cupy as cp

from .gpu import letterbox, nms
from .trt import TensorRTEngine


class PersonDetector:
    def __init__(self, path, device=0, *, size=(640, 640), layout="BCN",
                 iou_threshold=0.4, max_people=300, resize="stretch"):
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
        labels_path = Path(path).with_suffix(".txt")
        self.classes = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        while self.classes and not self.classes[-1]:
            self.classes.pop()
        if (not self.classes or any(not name for name in self.classes)
                or len(set(self.classes)) != len(self.classes) or "person" not in self.classes):
            raise ValueError(f"{labels_path} must contain unique non-empty labels including person")
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
            raise ValueError("Expected raw YOLOv8 boxes and class scores matching the label file")
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
        self.engine = None
