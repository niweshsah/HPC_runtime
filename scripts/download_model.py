#!/usr/bin/env python3
"""Export pretrained ResNet18; Python has no role in runtime execution.

Install: torch==2.6.0 torchvision==0.21.0 onnx==1.17.0
"""

import argparse
from pathlib import Path


def export_pretrained_model(output_path: Path) -> None:
    import onnx
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    example_input = torch.zeros(1, 3, 224, 224)
    # The legacy exporter is pinned deliberately for a small, reproducible script.
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(output_path))
    print(f"Exported and checked pretrained ResNet18: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/resnet18.onnx"))
    options = parser.parse_args()
    if options.output.exists():
        parser.error(f"Output already exists: {options.output}; choose another path")
    export_pretrained_model(options.output)


if __name__ == "__main__":
    main()
