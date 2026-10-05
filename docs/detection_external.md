# person-inference

Image detection and segmentation with optional Ultralytics (`.pt`), ONNX (`.onnx`), and TensorRT (`.trt`) backends.

## Install

Choose the backend you need. The base package contains shared NumPy, OpenCV, and Shapely code; backend libraries are installed only through the selected extra.

```bash
pip install '.[ultralytics]'
pip install '.[onnx]'
pip install '.[tensorrt]'
```

After pushing this directory to GitHub, install directly from the repository. Replace `OWNER/REPO` with its actual location:

```bash
pip install 'nba-face[ultralytics] @ git+https://github.com/OWNER/REPO.git'
pip install 'nba-face[onnx] @ git+https://github.com/OWNER/REPO.git'
pip install 'nba-face[tensorrt] @ git+https://github.com/OWNER/REPO.git'
```

TensorRT and PyCUDA require a compatible CUDA environment. The ONNX extra uses `onnxruntime`; install a suitable GPU runtime separately if you need ONNX inference on CUDA.

## Inference

Place a text file with the same stem as the model next to it, with one class name per line (for example, `model.txt` for `model.trt`).

```python
import cv2
from person_inference import Detector

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")

results = Detector("model.trt").detect_n_seg(image)
print([obj.object_class for obj in results])
cv2.imwrite("output.jpg", results.plot(image))
```

`results.boxes` is a NumPy array with columns `x1, y1, x2, y2, score, class`, plus an optional tracking ID. `results.masks` is a batched boolean array when segmentation masks are available. `results.intersects(other)` and `results.intersection_areas(other)` calculate pairwise box intersections.

## TensorRT batch inference

Pass a list/tuple of BGR images (including different image sizes), or an NHWC
NumPy array. Batch input always returns a list of `DetectionResults` in input
order, including for a one-image batch; a single HWC image keeps the original
single-result return type. An empty batch returns an empty list.

```python
detector = Detector("model.trt")
images = [cv2.imread("first.jpg"), cv2.imread("second.jpg")]
results = detector.detect_n_seg(images, labels=["person"])
for image, result in zip(images, results):
    print(result.boxes.shape, result.image_shape)
```

The engine must expose an explicit NCHW float16/float32 input and raw YOLOv8
outputs, using linear device tensors. Optimization profile zero controls batch
limits; dynamic spatial dimensions use that profile's optimum height and width.
Requests larger than its maximum batch size are split into chunks. Fixed-batch
engines and profile minima are handled by repeating the last image in a short
chunk and discarding its extra predictions. An engine built for batch size one
still executes one image at a time; build an engine with a larger batch dimension
or profile to execute multiple images together.

Preprocessing normalizes and converts layout across each batch, retaining each
image's resize/padding metadata. Confidence and class filtering are vectorized
across the batch. NMS runs independently per image; box transforms and mask
reconstruction remain vectorized over detections, with bounded mask work chunks.
Boxes and masks are returned in each original image's coordinates. Detection
models retain stretch resizing, while segmentation models use letterboxing.

For incremental consumption, `detector.model.predict(images, stream=True)` yields
one result per image and executes one chunk at a time. `detect_n_seg` collects
these results into a list. Each TensorRT instance owns reusable pinned host/device
buffers and one execution context; use separate instances for concurrent calls.
ONNX batch inference is not supported by this wrapper.

### Batch result format

For `B` input images, the return value is a `list[DetectionResults]` of length
`B`. `results[i]` belongs to `images[i]`. Each image can have a different number
of detections `N` and different dimensions `H` and `W`.

| Field on each result | Format | Meaning |
| --- | --- | --- |
| `boxes` | `float32`, `(N, 6)` | Rows: `[x1, y1, x2, y2, score, class_id]`. Coordinates are pixels in the original image. Tracking results may add a seventh `id` column. |
| `xyxy` | `float32`, `(N, 4)` | Bounding boxes only. |
| `scores` | `float32`, `(N,)` | Detection confidence. |
| `class_ids` | Integer array, `(N,)` | Class indices. |
| `names` | List or dictionary | Resolve a label with `result.names[int(class_id)]`. |
| `masks` | Boolean array, `(N, H, W)`, or `None` | TensorRT segmentation masks aligned with box rows; `None` for detection-only models. |
| `image_shape` | `(H, W, 3)` | Original TensorRT image shape; use `image_shape[:2]` for height and width. |
| `areas` | `float32`, `(N,)` | Bounding-box areas in square pixels, not mask areas. |

An image with no detections still has a result with `len(result) == 0` and
`boxes.shape == (0, 6)`. Segmentation masks then have shape `(0, H, W)`;
detection-only masks remain `None`. There is no single rectangular array for
the batch because detection counts and image sizes may differ.

For example, two images with two and zero detections produce:

```text
results = [DetectionResults, DetectionResults]
results[0].boxes = [[10.0, 20.0, 110.0, 220.0, 0.95, 0.0],
                    [150.0, 30.0, 250.0, 230.0, 0.88, 0.0]]
results[1].boxes.shape = (0, 6)
```

Class `0` means `results[0].names[0]`, which depends on the model's labels.

### Use batch results for further processing

Keep each source image paired with its result. Boolean indexing filters boxes
and masks together. Iterating a result yields objects with `object_class`,
`conf`, integer `box`, `area`, and `maskxy` (the largest external mask contour,
or an empty list when unavailable). Without a tracking column, `obj.id` is only
the row index, not a persistent identity across images.

This example filters detections, saves crops and annotated images, optionally
saves foreground images, and exports metadata for another application:

```python
import json
from pathlib import Path

import cv2
from nba-face import Detector

paths = [Path("first.jpg"), Path("second.jpg")]
images = []
for path in paths:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    images.append(image)

detector = Detector("model.trt")
results = detector.detect_n_seg(images, labels=["person"], score_threshold=0.5)
output_dir = Path("batch_output")
output_dir.mkdir(parents=True, exist_ok=True)
records = []

for image_index, (path, image, result) in enumerate(zip(paths, images, results)):
    selected = result[(result.scores >= 0.7) & (result.areas >= 1000)]
    prefix = f"{image_index}_{path.stem}"
    cv2.imwrite(str(output_dir / f"{prefix}_annotated.jpg"), selected.plot(image))

    detections = []
    for object_index, obj in enumerate(selected):
        crop = detector.crop(image, obj.box)
        if crop.size:
            cv2.imwrite(str(output_dir / f"{prefix}_crop_{object_index}.jpg"), crop)
        detections.append({
            "label": obj.object_class,
            "score": obj.conf,
            "box_xyxy": obj.box,
        })

    if selected.masks is not None:
        # Union of selected masks: uint8 (H, W), values 0 or 1.
        mask = selected.image_mask()
        foreground = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(str(output_dir / f"{prefix}_foreground.png"), foreground)

    # Preserve the source record even when there are no selected detections.
    records.append({"source": str(path), "detections": detections})

(output_dir / "detections.json").write_text(json.dumps(records, indent=2))
```

For array-based processing, use `selected.xyxy`, `selected.scores`, and
`selected.class_ids` directly; call `.tolist()` before JSON serialization.
To retain individual masks, save `selected.masks` with `numpy.save` instead of
expanding the arrays into JSON. For spatial matching within an image,
`selected.intersects(other)` returns an `(N, M)` boolean matrix against another
`DetectionResults` with `M` boxes in the same coordinate system.