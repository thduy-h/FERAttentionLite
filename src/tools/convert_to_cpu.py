from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.fer_attention_lite import FERAttentionLite


def load_checkpoint(path: Path) -> dict[str, Any] | Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]

        # Allow raw state_dict checkpoints too.
        if checkpoint and all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            first_value = next(iter(checkpoint.values()))
            if torch.is_tensor(first_value):
                return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. "
        "Expected a dict containing 'model_state' or a raw state_dict."
    )


def convert_checkpoint_to_cpu(
    input_path: Path,
    output_path: Path,
) -> None:
    checkpoint = load_checkpoint(input_path)
    state_dict = extract_state_dict(checkpoint)

    # Validate compatibility with the exact architecture.
    model = FERAttentionLite(num_classes=7)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_checkpoint,
        output_path,
    )

    print(f"CPU checkpoint saved: {output_path}")


def export_torchscript(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)

    model = FERAttentionLite(num_classes=7)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    example = torch.randn(1, 1, 48, 48)

    with torch.inference_mode():
        scripted = torch.jit.trace(
            model,
            example,
        )

        # Verify basic output shape before saving.
        output = scripted(example)
        if output.shape != (1, 7):
            raise RuntimeError(
                f"Unexpected TorchScript output shape: {tuple(output.shape)}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scripted.save(str(output_path))
    print(f"TorchScript CPU model saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FERAttentionLite Kaggle checkpoint "
            "to a CPU-safe PyTorch checkpoint, and optionally TorchScript."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input Kaggle best.pt",
    )

    parser.add_argument(
        "--output",
        default="weights/best_cpu.pt",
        help="Output CPU checkpoint",
    )

    parser.add_argument(
        "--torchscript",
        default=None,
        help=(
            "Optional output .pt/.ts path for TorchScript, "
            "e.g. weights/fer_attention_lite_cpu.ts"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    convert_checkpoint_to_cpu(
        input_path,
        output_path,
    )

    if args.torchscript:
        export_torchscript(
            output_path,
            Path(args.torchscript),
        )


if __name__ == "__main__":
    main()
