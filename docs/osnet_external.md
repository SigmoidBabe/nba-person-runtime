# Package-owned GPU OSNet ReID

`cv_people.reid.PersonReID` replaces the external `osnet-reid-trt` dependency.
It follows the preprocessing and engine contract in `packages/person-reid`,
implemented using CuPy CUDA kernels and the package's TensorRT engine module.

`extract(image, boxes)` consumes the original CUDA BGR frame and CUDA xyxy box
rows. Cropping, bilinear resize to 256×128, BGR-to-RGB conversion, division by
255, ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), and
NCHW layout run in a fused GPU kernel. Inference and L2 normalization also run
on GPU. Float sampling can differ slightly from the old OpenCV uint8 resize.

The result is a CUDA `[N,embedding_dim]` tensor. Invalid/zero-norm embeddings
become zero vectors; the backend marks those metadata rows as missing features.
Only normalized vectors used by the existing BoT-SORT metadata association are
downloaded. No crop or image pixels are staged on CPU.

The serialized engine must contain the trained OSNet backbone and BN neck,
with one FP16/FP32 linear device input `[B,3,256,128]` and one embedding output
`[B,embedding_dim]`. Profile zero must support the spatial dimensions. Crops are
chunked by `reid_batch_size` and engine batch bounds; short batches are padded on
GPU when necessary. Engines/weights are supplied separately.
