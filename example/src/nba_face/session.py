"""Session thresholds, snapshots and periodic/appearance event delivery."""
import math
import time


def similarity_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("similarity_threshold must be a number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("similarity_threshold must be a number in [0, 1]")
    return float(value)


class FaceSession:
    def __init__(self, session, threshold):
        if session["capability"] != "face.recognition":
            raise ValueError("Only face.recognition is supported")
        if session.get("zones"):
            raise ValueError("Face recognition supports full-frame sessions; omit zones")
        self.threshold = similarity_threshold(session.get("config", {}).get("similarity_threshold", threshold))
        output = session.get("output", {})
        self.mode = output.get("mode", "periodic")
        if self.mode not in ("event", "periodic"):
            raise ValueError("output.mode must be event or periodic")
        interval = output.get("interval", 0)
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval < 0):
            raise ValueError("output.interval must be finite and non-negative")
        self.interval = float(interval)
        self.last_emit = float("-inf")
        self.previous = set()

    def process(self, candidates):
        faces = []
        for candidate in candidates:
            face = dict(candidate)
            similarity = face["similarity"]
            if similarity is None or not math.isfinite(similarity) or similarity < self.threshold:
                for key in ("identity_id", "employee_id", "image_url", "pose"):
                    face[key] = None
            faces.append(face)
        snapshot = {"faces_count": len(faces), "faces": faces}
        current = {f["identity_id"] for f in faces if f["identity_id"] is not None}
        now = time.monotonic()
        emit = (now - self.last_emit >= self.interval if self.mode == "periodic"
                else bool(current - self.previous))
        self.previous = current
        if emit:
            self.last_emit = now
        return {"snapshot": snapshot, "events": [{"data": snapshot}] if emit else []}
