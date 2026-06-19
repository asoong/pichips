#!/usr/bin/env python3
"""Train a PiChip YOLO detector from an exported training-set dataset.

Consumes the YOLO-format dataset exported by the Raya Labs PiChip web client
(``images/``, ``labels/``, ``data.yaml``) and writes the best weights to
``models/pichip_detector.pt`` — the path ``overlay_viewer.py`` loads by default. The 9
class names (white_face, red_edge, ...) come straight from the dataset's ``data.yaml`` and
get embedded into the trained model, so the viewer reads them back via ``model.names`` and
never hardcodes a class list.

Example:
    python train.py --dataset datasets/pichip_abcd --model yolo26n.pt \\
        --epochs 100 --imgsz 1024 --batch 16 --device mps
"""

import argparse
import shutil
from pathlib import Path


def resolve_device(preference: str) -> str:
    """Resolve 'auto' to the best available device (cuda > mps > cpu)."""
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the extracted dataset directory containing data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Base YOLO model to fine-tune (e.g. yolo26n.pt, yolo11s.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto, mps, cuda, 0, cpu",
    )
    parser.add_argument(
        "--out",
        default="models/pichip_detector.pt",
        help="Where to copy the best weights (the viewer's default model path)",
    )
    parser.add_argument(
        "--name",
        default="pichip",
        help="Run name under runs/detect/",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Also export an ONNX copy next to the .pt for portable inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}. Point --dataset at the extracted "
            "training-set folder (the one containing data.yaml, images/, labels/)."
        )

    # Imported lazily so --help works without the heavy ML stack installed.
    from ultralytics import YOLO

    device = resolve_device(args.device)
    print(f"Training {args.model} on {device} for {args.epochs} epochs...")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/detect",
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )

    # Locate the best checkpoint produced by the run.
    best = getattr(model.trainer, "best", None)
    if not best or not Path(best).exists():
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    best = Path(best)
    if not best.exists():
        raise SystemExit(f"Training finished but best weights not found at {best}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nCopied best weights:\n  {best}\n  -> {out}")
    print("Run the viewer with:  python overlay_viewer.py")

    if args.export_onnx:
        print("Exporting ONNX...")
        YOLO(str(out)).export(format="onnx")


if __name__ == "__main__":
    main()
