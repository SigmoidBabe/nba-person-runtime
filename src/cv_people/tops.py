"""Estimate model throughput from registry operation counts without loading CUDA."""
import math


def calculate_tops(config, fps, batch_size, cv_capability):
    """Estimate detection or tracking TOPS, assuming 20 people per tracking frame."""
    capabilities = {
        "person.detection": "person.detection",
        "person detection": "person.detection",
        "person.tracking": "person.tracking",
        "person tracking": "person.tracking",
    }
    if not isinstance(cv_capability, str) or cv_capability not in capabilities:
        raise ValueError("cv_capability must be person.detection or person.tracking")
    cv_capability = capabilities[cv_capability]
    if type(fps) not in (int, float) or fps <= 0:
        raise ValueError("fps must be a finite number greater than zero")
    if (type(batch_size) not in (int, float)
            or not 0 < batch_size <= 2**53 - 1
            or int(batch_size) != batch_size):
        raise ValueError("batch_size must be a positive integer no greater than 2**53 - 1")
    # Protobuf Struct represents JSON numbers as doubles, including integers.
    batch_size = int(batch_size)
    try:
        finite_fps = math.isfinite(fps)
    except OverflowError:
        finite_fps = False
    if not finite_fps:
        raise ValueError("fps must be a finite number greater than zero")

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Registry config.model must be an object")
    models = {}
    inferences_per_frame = {"person_detection": 1}
    if cv_capability == "person.tracking":
        inferences_per_frame["person_reidentification"] = 20
    for name, count in inferences_per_frame.items():
        model = model_config.get(name)
        if not isinstance(model, dict):
            raise ValueError(f"Registry config.model.{name} must be an object")
        operations = model.get("operations_per_inference")
        if type(operations) not in (int, float) or operations < 0:
            raise ValueError(
                f"Registry config.model.{name}.operations_per_inference "
                "must be a finite nonnegative number"
            )
        try:
            tops = operations * count * fps * batch_size / 1_000_000_000_000
            finite = math.isfinite(operations) and math.isfinite(tops)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"Operation count or calculated TOPS for {name} is not finite")
        models[name] = {"operations_per_inference": operations, "tops": tops}
    total = sum(model["tops"] for model in models.values())
    if not math.isfinite(total):
        raise ValueError("Calculated total TOPS is not finite")
    result = {"cv_capability": cv_capability, "fps": fps, "batch_size": batch_size,
              "models": models, "total_tops": total}
    if cv_capability == "person.tracking":
        result["people_per_frame"] = 20
    return result


class PeoplePerformanceAPI:
    """Expose TOPS calculation through the host's PackageAPI/Invoke service."""

    def __init__(self, config):
        self.config = config

    def invoke(self, method, payload):
        if method != "calculate_tops":
            raise NotImplementedError(f"Unknown method: {method}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return calculate_tops(
            self.config, payload.get("fps"), payload.get("batch_size"),
            payload.get("cv_capability"),
        )

    def close(self):
        pass
