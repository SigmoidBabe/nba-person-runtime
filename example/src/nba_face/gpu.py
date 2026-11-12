"""CuPy CUDA kernels operating directly on strided ingestion frame storage."""
import cupy as cp


_SOURCE = r'''
// Source strides are bytes; HWC uint8 views need no contiguous frame copy.
__device__ float sample_bgr(
    const unsigned char* image, int height, int width,
    long long sy, long long sx, long long sc, float y, float x, int channel
) {
    y = fminf(fmaxf(y, 0.0f), height - 1.0f);
    x = fminf(fmaxf(x, 0.0f), width - 1.0f);
    int y0 = (int)y, x0 = (int)x;
    int y1 = min(y0 + 1, height - 1), x1 = min(x0 + 1, width - 1);
    float dy = y - y0, dx = x - x0;
    long long c = (2 - channel) * sc;
    float a = image[y0 * sy + x0 * sx + c];
    float b = image[y0 * sy + x1 * sx + c];
    float d = image[y1 * sy + x0 * sx + c];
    float e = image[y1 * sy + x1 * sx + c];
    return (a + (b - a) * dx) * (1.0f - dy) + (d + (e - d) * dx) * dy;
}

extern "C" __global__ void letterbox(
    const unsigned char* image, float* output, int height, int width,
    long long sy, long long sx, long long sc,
    int out_h, int out_w, int resized_h, int resized_w, int top, int left
) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= 3 * out_h * out_w) return;
    int x = index % out_w, y = (index / out_w) % out_h;
    int channel = index / (out_h * out_w);
    float pixel = 114.0f;
    if (x >= left && x < left + resized_w && y >= top && y < top + resized_h) {
        float src_x = (x - left + 0.5f) * width / resized_w - 0.5f;
        float src_y = (y - top + 0.5f) * height / resized_h - 0.5f;
        pixel = sample_bgr(image, height, width, sy, sx, sc, src_y, src_x, channel);
    }
    output[index] = pixel / 255.0f;
}

extern "C" __global__ void face_crops(
    const unsigned char* image, const float* boxes, float* output,
    int height, int width, long long sy, long long sx, long long sc, int count
) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count * 3 * 112 * 112) return;
    int x = index % 112, y = (index / 112) % 112;
    int channel = (index / (112 * 112)) % 3;
    int face = index / (3 * 112 * 112);
    const float* box = boxes + face * 4;
    float src_x = box[0] + (x + 0.5f) * (box[2] - box[0]) / 112.0f - 0.5f;
    float src_y = box[1] + (y + 0.5f) * (box[3] - box[1]) / 112.0f - 0.5f;
    float pixel = sample_bgr(image, height, width, sy, sx, sc, src_y, src_x, channel);
    output[index] = pixel / 127.5f - 1.0f;
}

// Scores are already sorted descending. One block makes greedy choices in
// order, while its threads suppress overlapping candidates in parallel.
extern "C" __global__ void greedy_nms(
    const float* boxes, unsigned char* suppressed, unsigned char* keep,
    int count, int limit, float threshold
) {
    __shared__ int selected;
    __shared__ int kept;
    if (threadIdx.x == 0) kept = 0;
    __syncthreads();
    for (int i = 0; i < count && kept < limit; ++i) {
        if (threadIdx.x == 0) {
            selected = !suppressed[i];
            if (selected) { keep[i] = 1; ++kept; }
        }
        __syncthreads();
        if (selected) {
            const float* a = boxes + i * 4;
            float area_a = (a[2] - a[0]) * (a[3] - a[1]);
            for (int j = i + 1 + threadIdx.x; j < count; j += blockDim.x) {
                if (suppressed[j]) continue;
                const float* b = boxes + j * 4;
                float w = fmaxf(0.0f, fminf(a[2], b[2]) - fmaxf(a[0], b[0]));
                float h = fmaxf(0.0f, fminf(a[3], b[3]) - fmaxf(a[1], b[1]));
                float intersection = w * h;
                float union_area = area_a + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
                if (intersection > threshold * union_area) suppressed[j] = 1;
            }
        }
        __syncthreads();
    }
}
'''

_LETTERBOX = cp.RawKernel(_SOURCE, "letterbox")
_CROPS = cp.RawKernel(_SOURCE, "face_crops")
_NMS = cp.RawKernel(_SOURCE, "greedy_nms")


def _source_args(image):
    height, width = image.shape[:2]
    return (cp.int32(height), cp.int32(width), *(cp.int64(s) for s in image.strides))


def letterbox(image, size):
    height, width = image.shape[:2]
    target_h, target_w = size
    ratio = min(target_h / height, target_w / width)
    resized_h, resized_w = max(1, round(height * ratio)), max(1, round(width * ratio))
    top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
    output = cp.empty((1, 3, target_h, target_w), dtype=cp.float32)
    _LETTERBOX(((output.size + 255) // 256,), (256,), (
        image, output, *_source_args(image),
        *(cp.int32(v) for v in (target_h, target_w, resized_h, resized_w, top, left)),
    ))
    return output, (left, top, resized_w / width, resized_h / height)


def face_crops(image, boxes):
    boxes = cp.ascontiguousarray(boxes, dtype=cp.float32)
    output = cp.empty((len(boxes), 3, 112, 112), dtype=cp.float32)
    if output.size:
        _CROPS(((output.size + 255) // 256,), (256,), (
            image, boxes, output, *_source_args(image), cp.int32(len(boxes)),
        ))
    return output


def nms(boxes, threshold, limit):
    """Return GPU indices for score-sorted xyxy boxes, bounded by limit."""
    boxes = cp.ascontiguousarray(boxes, dtype=cp.float32)
    suppressed = cp.zeros(len(boxes), dtype=cp.uint8)
    keep = cp.zeros(len(boxes), dtype=cp.uint8)
    if len(boxes):
        _NMS((1,), (256,), (boxes, suppressed, keep, cp.int32(len(boxes)),
                            cp.int32(limit), cp.float32(threshold)))
    return cp.flatnonzero(keep)
