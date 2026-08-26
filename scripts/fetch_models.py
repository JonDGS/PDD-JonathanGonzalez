"""Download checksum-verified selector models from the project release."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tree_top_detector.model_registry import fetch_model, load_manifest, select_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selector models declared in models/manifest.toml."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "models" / "manifest.toml",
        help="Model manifest path.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root under which manifest target paths are created.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Download every model.")
    selection.add_argument(
        "--task",
        choices=("detection", "classification"),
        help="Download every model for one task.",
    )
    selection.add_argument(
        "--model",
        action="append",
        dest="model_ids",
        help="Download one model ID; repeat for multiple models.",
    )
    selection.add_argument("--list", action="store_true", help="List models without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.list:
        for model in manifest.models:
            size_mib = model.size / (1024 * 1024)
            print(f"{model.model_id:24} {model.task:14} {size_mib:6.1f} MiB  {model.display_name}")
        return 0

    try:
        selected = select_models(
            manifest,
            task=args.task,
            model_ids=tuple(args.model_ids or ()),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for index, model in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {model.display_name}")
        destination = fetch_model(model, manifest.base_url, args.destination_root)
        print(f"  ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
