"""Calculate workload TOPS from configured per-model operation counts."""
import math


def _positive_number(value, name):
    message = f"{name} must be a finite positive number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(message)
    return number


class FacePerformanceAPI:
    def __init__(self, config):
        self.config = config

    def invoke(self, method, payload):
        if method != "calculate_tops":
            raise NotImplementedError(f"Unknown method: {method}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        fps = _positive_number(payload.get("fps"), "fps")
        batchsize = _positive_number(payload.get("batchsize"), "batchsize")
        if not batchsize.is_integer() or batchsize > 2**53 - 1:
            raise ValueError("batchsize must be a positive integer no greater than 2**53 - 1")
        models = self.config.get("model")
        if not isinstance(models, dict):
            raise ValueError("config.model must be an object")
        counts = []
        for name in ("face_detection", "face_extraction"):
            model = models.get(name)
            if not isinstance(model, dict):
                raise ValueError(f"config.model.{name} must be an object")
            counts.append(_positive_number(
                model.get("operations_per_inference"),
                f"config.model.{name}.operations_per_inference",
            ))
        detector, arcface = counts
        # One detector frame and one extracted face per item; FPS counts batches.
        operations = detector + arcface
        detector_tops = (detector / 1e12) * fps * batchsize
        arcface_tops = (arcface / 1e12) * fps * batchsize
        tops = detector_tops + arcface_tops
        if not all(math.isfinite(value) for value in (operations, detector_tops, arcface_tops, tops)):
            raise ValueError("Calculated result exceeds the supported numeric range")
        return {
            "fps": fps,
            "batchsize": int(batchsize),
            "faces_per_frame": 1,
            "detector_operations_per_inference": detector,
            "arcface_operations_per_inference": arcface,
            "operations_per_inference": operations,
            "detector_tops": detector_tops,
            "arcface_tops": arcface_tops,
            "tops": tops,
        }

    def close(self):
        pass
