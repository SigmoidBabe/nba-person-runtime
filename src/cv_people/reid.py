"""Package-owned OSNet TensorRT, GPU crops and embedding postprocessing."""
import cupy as cp

from .gpu import person_crops
from .trt import TensorRTEngine


class PersonReID:
    def __init__(self, path, device=0, batch_size=32):
        self.engine = TensorRTEngine(path, device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("reid_batch_size must be positive")
        if any(not low <= dim <= high for low, dim, high in
               zip(self.engine.minimum[1:], (3, 256, 128), self.engine.maximum[1:])):
            raise ValueError("OSNet engine must accept [N,3,256,128]")

    def extract(self, image, boxes):
        """Return normalized CUDA embeddings aligned to the CUDA box rows."""
        parts = []
        for start in range(0, len(boxes), self.batch_size):
            crops = person_crops(image, boxes[start:start + self.batch_size, :4])
            output = self.engine.infer_batch(crops)
            if output.ndim != 2 or output.shape[1] <= 0:
                raise ValueError("OSNet engine must output [N,embedding_dim]")
            output = output.astype(cp.float32, copy=False)
            norms = cp.linalg.norm(output, axis=1, keepdims=True)
            valid = cp.isfinite(output).all(1) & cp.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
            # Zero vectors are marked missing at the compact metadata boundary.
            parts.append(cp.where(valid[:, None], output / cp.maximum(norms, 1e-12), 0))
        return cp.concatenate(parts) if parts else cp.empty((0, 0), dtype=cp.float32)

    def close(self):
        self.engine = None
