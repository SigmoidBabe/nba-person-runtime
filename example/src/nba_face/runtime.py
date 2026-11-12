"""Host runtime entry point plus an explicit multi-frame batch API."""
import threading

from .backend import FaceBackend
from .session import FaceSession, similarity_threshold


class FaceRuntime:
    def __init__(self, config):
        self.threshold = similarity_threshold(config.get("similarity_threshold", 0.6))
        self.backend = FaceBackend(config)
        self.sessions = {}
        self.lock = threading.RLock()

    def create_session(self, session):
        with self.lock:
            self.sessions[session["session_id"]] = FaceSession(session, self.threshold)

    def infer_batch(self, images):
        """Batched GPU inference + DB search; one candidate list per HWC image."""
        with self.lock:
            return self.backend.infer_batch(images)

    def process_batch(self, frames, session_ids_by_frame):
        """Ordered frame dictionaries and corresponding eligible session ID lists.

        Returns a list of per-frame {session_id: result} mappings. Session timing
        and appearance state advance in frame order. The current host calls the
        single-frame adapter; a batching host may call this API directly.
        """
        if len(frames) != len(session_ids_by_frame):
            raise ValueError("Each frame requires a corresponding session ID list")
        with self.lock:
            requested = [tuple(ids) for ids in session_ids_by_frame]
            for ids in requested:
                for sid in ids:
                    if sid not in self.sessions:
                        raise KeyError(f"Unknown session: {sid}")
            active = [i for i, ids in enumerate(requested) if ids]
            candidates = self.backend.infer_batch([frames[i]["image"] for i in active])
            results = [{} for _ in frames]
            for index, faces in zip(active, candidates):
                results[index] = {sid: self.sessions[sid].process(faces) for sid in requested[index]}
            return results

    def process_frame(self, frame, session_ids):
        return self.process_batch([frame], [session_ids])[0]

    def close_session(self, session_id):
        with self.lock:
            self.sessions.pop(session_id, None)

    def close(self):
        with self.lock:
            self.sessions.clear()
            self.backend.close()
