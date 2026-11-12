"""Face registration and soft deletion through the host's PackageAPI/Invoke."""
import base64
import binascii
import math
from pathlib import Path
import re
import threading
import uuid

import cv2
import numpy as np

from .backend import FaceBackend
from .database import FaceSearch


def _bigint(value, name, *, positive=False):
    message = f"{name} must be {'a positive' if positive else 'an'} integer or integer string"
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?[0-9]{1,19}", value.strip()):
            raise ValueError(message)
        value = int(value.strip())
    elif (isinstance(value, bool) or not isinstance(value, (int, float))
          or (isinstance(value, float) and (not math.isfinite(value) or value != int(value)))):
        raise ValueError(message)
    value = int(value)
    if not -(2**63) <= value < 2**63 or (positive and value <= 0):
        raise ValueError(message)
    return value


class FaceRegistrationAPI:
    def __init__(self, config):
        self.config = config
        self.database = FaceSearch(config["db"])
        self.image_directory = Path(config["registration_image_dir"])
        self.backend = None
        self.lock = threading.Lock()

    def invoke(self, method, payload):
        if method not in ("register_face", "delete_face"):
            raise NotImplementedError(f"Unknown method: {method}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if method == "delete_face":
            identity_id = _bigint(payload.get("identity_id"), "identity_id", positive=True)
            with self.lock:
                return self.database.delete(identity_id)

        employee_id = _bigint(payload.get("employee_id"), "employee_id")
        crop_type = payload.get("crop_type", "loose")
        if not isinstance(crop_type, str) or crop_type.strip().lower() not in {"full", "tight", "loose"}:
            raise ValueError("crop_type must be one of: full, tight, loose")
        crop_type = crop_type.strip().lower()
        encoded = payload.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_base64 is required")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image_base64 must contain valid base64") from exc
        if not content:
            raise ValueError("Image is empty")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
        with self.lock:
            if self.backend is None:
                self.backend = FaceBackend(self.config)
            embedding, (x1, y1, x2, y2) = self.backend.registration_embedding(image, crop_type)
            success, jpeg = cv2.imencode(".jpg", image[y1:y2, x1:x2])
            if not success:
                raise RuntimeError("Cannot encode registration image")
            self.image_directory.mkdir(parents=True, exist_ok=True)
            path = self.image_directory / f"{uuid.uuid4()}.jpg"
            created = False
            try:
                with path.open("xb") as output:
                    created = True
                    output.write(jpeg.tobytes())
                return self.database.register(employee_id, embedding, str(path))
            except Exception:
                if created:
                    path.unlink(missing_ok=True)
                raise

    def close(self):
        with self.lock:
            try:
                self.database.close()
            finally:
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
