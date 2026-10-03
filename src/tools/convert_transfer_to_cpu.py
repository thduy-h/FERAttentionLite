from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.mobilenet_v3_small import build_mobilenet_v3_small


def load_checkpoint(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]

    if isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(k, str) and torch.is_tensor(v)
        for k, v in checkpoint.items()
    ):
        return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected a dict containing "
        "'model_state' or a raw state_dict."
    )


def convert_checkpoint_to_cpu(input_path: Path, output_path: Path) -> None:
    checkpoint = load_checkpoint(input_path)
    state_dict = extract_state_dict(checkpoint)

    model = build_mobilenet_v3_small(num_classes=7, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.cpu().eval()

    cpu_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        output_checkpoint = dict(checkpoint)
        output_checkpoint["model_state"] = cpu_state
        output_checkpoint["export_device"] = "cpu"
    else:
        output_checkpoint = cpu_state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)

    print(f"Saved CPU checkpoint: {output_path}")


def export_torchscript(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)

    model = build_mobilenet_v3_small(num_classes=7, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.cpu().eval()

    example = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        traced = torch.jit.trace(model, example)
        output = traced(example)

    if tuple(output.shape) != (1, 7):
        raise RuntimeError(
            f"Unexpected TorchScript output shape: {tuple(output.shape)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Saved TorchScript: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the MobileNetV3-Small FER checkpoint to CPU."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="weights/best_transfer_cpu.pt",
    )
    parser.add_argument(
        "--torchscript",
        default=None,
        help="Optional TorchScript output path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    convert_checkpoint_to_cpu(Path(args.input), output_path)

    if args.torchscript:
        export_torchscript(output_path, Path(args.torchscript))


if __name__ == "__main__":
    main()
