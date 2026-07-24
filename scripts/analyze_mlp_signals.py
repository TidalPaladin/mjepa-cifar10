import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch

from mjepa_cifar10.research.mlp_diagnostics import analyze_completed_checkpoint, validate_completed_checkpoint


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Analyze layerwise MLP activations and gradients from completed JEPA checkpoints on CPU."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-mode", choices=("eval", "train"), default="eval")
    return parser.parse_args()


def main(args: Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output}")
    if not args.output.parent.is_dir():
        raise NotADirectoryError(args.output.parent)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    completed_checkpoints = [validate_completed_checkpoint(path) for path in args.checkpoints]
    results: list[dict[str, Any]] = []
    for completed in completed_checkpoints:
        results.append(
            analyze_completed_checkpoint(
                completed,
                data_root=args.data,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                seed=args.seed,
                model_mode=args.model_mode,
            )
        )
    payload = {
        "schema_version": 1,
        "analysis": "final-checkpoint-mlp-signals",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def entrypoint() -> None:
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
