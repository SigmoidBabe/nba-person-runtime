# Package-owned GPU detection

`cv_people.detection.PersonDetector` replaces the external
`person_inference.Detector` dependency. It uses `cv_people.trt.TensorRTEngine`
and `cv_people.gpu` CUDA kernels; no sibling package is imported.

Within the configured CuPy device context, pass a validated CUDA HWC uint8 BGR
frame to `detect(image, threshold)`. The result is a CUDA `[N,6]` array containing
xyxy boxes, confidence and class ID in original image coordinates. Raw output
must use YOLOv8 class scores without a separate objectness channel; the best
class per anchor is chosen before filtering for the model's `person` class.
Non-finite predictions and zero-area boxes are removed on CUDA.

The default stretch resize preserves the previous TensorRT detector's behavior;
`detector_resize: letterbox` enables aspect-preserving resize/padding. Layout,
score filtering, NMS and coordinate restoration run on GPU. No source image or
raw prediction is downloaded. The backend transfers compact final boxes only
for session tracking/analytics and gRPC JSON output.

See the package README and registry for engine requirements and configuration.
