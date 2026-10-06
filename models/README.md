# Pretrained model

The C++ runtime executes an existing model. No training code belongs here.

Export torchvision's pretrained ResNet18 with a dynamic batch dimension:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 onnx==1.17.0
python scripts/download_model.py --output models/resnet18.onnx
```

Run commands from the project root. The script downloads ImageNet weights, exports
ONNX opset 17, and runs the ONNX checker. It refuses to overwrite an existing model.
The generated model is ignored by Git and excluded from Docker build contexts.

The supported model contract is one float32 input `[batch, 3, 224, 224]` and one
float32 logits output `[batch, 1000]`. The leading dimension must be dynamic on
both tensors. The engine checks this contract before accepting requests.

For meaningful image predictions, use RGB images, resize the shorter side to 256,
center-crop to 224 × 224, convert channels to floats in `[0, 1]`, then normalize
with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
Flatten in contiguous NCHW order. Preprocessing is the caller's responsibility.
Class indices follow torchvision's ImageNet category ordering; the server returns
an index and confidence rather than downloading a separate label file.

The runtime computes stable softmax over output logits. Confidence is a model
score, not a guarantee of prediction accuracy. Synthetic tensors in smoke tests
and benchmarks validate execution and timing, not classification quality.

Supply a CPU ONNX Runtime SDK when building:

```bash
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DINFERENCE_RUNTIME_USE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-sdk
cmake --build build-onnx -j
INFERENCE_RUNTIME_TEST_MODEL="$PWD/models/resnet18.onnx" \
  ctest --test-dir build-onnx --output-on-failure
./build-onnx/inference_server --model models/resnet18.onnx --reuse-buffers
```

Validated SDK baseline: ONNX Runtime CPU 1.22.0. Obtain the archive matching your
OS and architecture from the [official releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0).
Use headers and libraries from the same SDK. The Python wheel alone does not
provide the C++ SDK. On Linux, add its `lib` directory to `LD_LIBRARY_PATH` if
moving the built executable outside its original build directory.

Source references: [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/namespace_ort.html).
