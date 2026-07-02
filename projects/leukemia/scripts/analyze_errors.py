"""Run test-set error analysis for a trained Cancer Cell Vision checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_leukemia.error_analysis import ErrorAnalysisConfig, run_error_analysis
from rccia_leukemia.model import SUPPORTED_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model errors on the test split.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default="resnet18")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_error_analysis(
        ErrorAnalysisConfig(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_examples=args.max_examples,
            generate_gradcam=not args.skip_gradcam,
        )
    )

    print(f"Analyzed {summary['total_images']} test images")
    print(f"Correct: {summary['correct_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"False positives: {summary['false_positive_count']}")
    print(f"False negatives: {summary['false_negative_count']}")
    print(f"Saved predictions to {args.output_dir / 'predictions.csv'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
