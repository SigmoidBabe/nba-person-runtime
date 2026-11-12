"""TensorRT tensor-name API with CuPy device storage and no host staging."""
from pathlib import Path

import tensorrt as trt
import cupy as cp


class TensorRTEngine:
    """One NCHW input, one output, optimization profile 0, serialized access."""

    def __init__(self, path, device):
        self.device = int(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create TensorRT context: {path}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Engines must have exactly one input and one output")
        self.input, self.output = inputs[0], outputs[0]
        dtypes = {trt.float32: cp.float32, trt.float16: cp.float16}
        self.dtypes = {}
        for name in names:
            if (self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError("TensorRT I/O must use LINEAR device tensors")
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in dtypes:
                raise ValueError("TensorRT I/O must be FP16 or FP32")
            self.dtypes[name] = dtypes[dtype]
        self.shape = tuple(self.engine.get_tensor_shape(self.input))
        if len(self.shape) != 4:
            raise ValueError("TensorRT input must be NCHW")
        self.dynamic = -1 in self.shape
        if self.dynamic:
            self.minimum, _, self.maximum = self.engine.get_tensor_profile_shape(self.input, 0)
        else:
            self.minimum = self.maximum = self.shape
        self.max_batch = int(self.maximum[0])
        if self.max_batch <= 0:
            raise ValueError("Engine must have a bounded batch dimension")

    def infer_batch(self, batch):
        """Chunk or pad to engine batch bounds; return CUDA tensors in input order."""
        if (not isinstance(batch, cp.ndarray) or batch.device.id != self.device
                or batch.ndim != 4 or not len(batch)):
            raise ValueError("Expected a non-empty CUDA NCHW batch on the engine device")
        outputs = []
        for start in range(0, len(batch), self.max_batch):
            tensor = cp.ascontiguousarray(batch[start:start + self.max_batch], dtype=self.dtypes[self.input])
            count = len(tensor)
            target = max(count, int(self.minimum[0]))
            if target != count:
                tensor = cp.concatenate((tensor, cp.broadcast_to(tensor[-1:], (target - count, *tensor.shape[1:]))))
            shape = tuple(tensor.shape)
            if any(not low <= dim <= high for low, dim, high in zip(self.minimum, shape, self.maximum)):
                raise ValueError(f"Input {shape} is outside engine profile {self.minimum}..{self.maximum}")
            if self.dynamic and not self.context.set_input_shape(self.input, shape):
                raise ValueError(f"Cannot set TensorRT input shape {shape}")
            output_shape = tuple(self.context.get_tensor_shape(self.output))
            if not output_shape or any(dim <= 0 for dim in output_shape) or output_shape[0] != target:
                raise ValueError(f"Output must have a resolved batch dimension: {output_shape}")
            output = cp.empty(output_shape, dtype=self.dtypes[self.output])
            stream = cp.cuda.get_current_stream()
            if not self.context.set_tensor_address(self.input, tensor.data.ptr):
                raise RuntimeError("Cannot bind TensorRT input")
            if not self.context.set_tensor_address(self.output, output.data.ptr):
                raise RuntimeError("Cannot bind TensorRT output")
            try:
                if not self.context.execute_async_v3(stream_handle=stream.ptr):
                    raise RuntimeError("TensorRT inference failed")
            finally:
                # TensorRT does not register CuPy storage lifetimes. Finish before
                # input storage is released or this context changes shape/bindings.
                stream.synchronize()
            outputs.append(output[:count])
        return cp.concatenate(outputs)
