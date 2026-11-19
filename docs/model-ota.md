# Model OTA from ONNX archives

`cv_demo.api.v1.ModelService/UpdateModelPath` accepts local `.tar.gz` paths on the
server. Each archive contains one `manifest.json` and one self-contained `.onnx`
file. Nested directories are allowed. ONNX external tensor files, archive links,
special files, duplicate paths and directory traversal are rejected. The example
[manifest.json](../../manifest.json) defines the supported metadata and profiles.

```json
{
  "cv_capability": "person tracking",
  "model": [
    {"person detection": "/srv/updates/person-detection.tar.gz"},
    {"person reidentification": "/srv/updates/person-reid.tar.gz"}
  ]
}
```

Names resolve to existing `config.model` keys. `person detection` prefers
`person_detection`, then the legacy `person` key; `person reidentification`
supports `person_reidentification` and `reid`. Existing keys can also be supplied
directly. Aliases resolving to the same key are rejected. Space or dotted capability
selectors are accepted. URLs and unarchived model paths are not supported.

## Update sequence

1. Resolve every requested key, extract the archives into unique release directories,
   and validate the manifests and ONNX models while existing inference continues.
2. Block new package API calls and runtime start/stop requests at the gRPC gateway.
   Calls arriving during maintenance receive `UNAVAILABLE`. Existing package work
   drains before shutdown, as do queued runtime controls.
3. Pause the supervised GPU pipeline. The worker closes sessions and package API
   instances, joins frame consumer threads, finishes pending recordings, stops
   ingestion readers, and exits. The supervisor acknowledges only after the worker
   exits successfully. This releases its CUDA contexts and framework caches before
   TensorRT export begins. A shutdown timeout fails the update without exporting.
4. Run `trtexec` sequentially for every requested ONNX file, building `.engine` files
   with the manifest's optimization profiles and precision. The build uses
   `--skipInference`; it does not run an inference benchmark.
5. After every export succeeds, atomically save all new registry paths, recreate
   the GPU pipeline with the updated configuration, and reload enabled sessions
   from the database. Disabled sessions remain disabled. The gRPC process stays up.
6. Return the original request fields, generated artifact paths, and
   `sessions_restarted: true` only after worker startup succeeds.

All GPU work managed by this application lives in the pipeline worker and its
ingestion children. This does not stop other applications using the same GPU;
use a dedicated device if exports require exclusive GPU capacity.

Existing runtime streams receive `PAUSED` during worker shutdown and `STARTING`
when restored. They remain connected unless a separate terminal `FAILED` or
`STOPPED` occurs. Camera streams and package model instances are recreated;
tracker IDs, temporal state and in-memory counters restart. Local video sources
replay from the beginning. Runtime sessions keep their persisted enabled settings.

On export failure the registry stays unchanged and the previous pipeline resumes.
If the new worker cannot load the models or restore enabled sessions, the old
registry is restored and the previous models are restarted. Old model files are
never overwritten. Failed staging releases are removed after successful recovery;
export logs remain under `models/releases/logs`. If rollback or recovery itself
fails, the RPC reports that failure explicitly. A client timeout or disconnect does
not cancel server-side maintenance; allow a long RPC deadline and inspect the
registry and runtime streams before retrying an interrupted request.

## Manifest and TensorRT environment

The manifest is a JSON object. `model_type` must be a non-empty string, `classes`
is an optional array, and `operations_per_inference` and
`instance_count_per_frame` may be null or nonnegative finite numbers. These metadata
fields are copied into nested model settings. The core does not infer the model's
semantic type from its filename or calculate operation counts during OTA.

`quantization` supports `fp16` and `fp32` (default). Other precisions are rejected.
For dynamic inputs, supply `minShapes`, `optShapes` and `maxShapes` together using
`input:1x3x256x128` syntax. Multiple inputs are comma-separated. Input sets and ranks
must agree, with min <= opt <= max for each dimension. Input names and dimensions
must match the actual ONNX model. Static models may omit all three profiles.

Install the NVIDIA TensorRT `trtexec` executable matching the deployment GPU,
driver and TensorRT runtime. TensorRT versions exposing `--fp16` use that build
flag. On strongly typed versions without that flag, the core converts the staged
ONNX graph to FP16 with `onnxconverter-common`, preserving input/output types,
then builds it. The deployment requirements include that converter. Conversion
failures abort the batch and restore the previous pipeline.

The precision and profile behavior follows NVIDIA's
[trtexec documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
and [TensorRT 11 migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html).
Building engines still uses the GPU and may take minutes; install dependencies
and run this repository through `python main.py`. A standalone gRPC server without
the supervisor rejects OTA with `FAILED_PRECONDITION`.

```yaml
model_ota:
  directory: models/releases
  # trtexec: /usr/src/tensorrt/bin/trtexec
  device_id: 0
  max_extracted_bytes: 4294967296
  export_timeout_seconds: 3600
  start_timeout_seconds: 120
  stop_timeout_seconds: 120
```

`trtexec` is discovered on PATH, with `/usr/src/tensorrt/bin/trtexec` as a fallback.
`device_id` should select the same GPU used for deployment inference. Archive limits
apply per archive; builds are sequential, so size the RPC deadline for the full
batch plus shutdown and startup. Relative archive and output paths use the server
working directory. Persist both the registry directory and the release directory.
Previous successful releases are retained for rollback; remove unused releases
through a separate storage retention policy.

## Registry and response

For a nested model entry, settings such as these replace the old paths while
preserving unrelated model options:

```yaml
config:
  model:
    person_detection:
      engine: /srv/cv-demo/models/releases/RELEASE/model.engine
      onnx: /srv/cv-demo/models/releases/RELEASE/model.onnx
      manifest: /srv/cv-demo/models/releases/RELEASE/manifest.json
      archive: /srv/updates/person-detection.tar.gz
      model_type: detection
      quantization: fp16
      classes: []
      operations_per_inference: null
      instance_count_per_frame: null
```

Optimization profiles and an `input_shapes` mapping derived from `optShapes` are
also stored when provided. Legacy string model entries remain strings pointing
to the generated engine, preserving their package configuration format. Runtime
packages must already support loading TensorRT engines; the core builds artifacts
and reloads packages, but does not replace their inference backend.

```json
{
  "cv_capability": "person tracking",
  "model": [{"person detection": "/srv/updates/person-detection.tar.gz"}],
  "artifacts": [{
    "name": "person_detection",
    "engine": "/srv/cv-demo/models/releases/RELEASE/model.engine",
    "onnx": "/srv/cv-demo/models/releases/RELEASE/model.onnx",
    "manifest": "/srv/cv-demo/models/releases/RELEASE/manifest.json"
  }],
  "sessions_restarted": true
}
```

Malformed manifests, unsupported precision and unsafe archives use
`INVALID_ARGUMENT`. Unknown capabilities, keys or missing archives use `NOT_FOUND`.
Invalid registry structure or missing supervision uses `FAILED_PRECONDITION`.
Build, shutdown or restart failures use `UNAVAILABLE`; filesystem errors use
`INTERNAL`.
