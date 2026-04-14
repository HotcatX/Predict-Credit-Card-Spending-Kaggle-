from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble aligned OOF matrices from lightweight shard bundles.")
    parser.add_argument(
        "--bundle",
        required=True,
        help="Bundle path relative to repo root, e.g. outputs/forward_selection_oof",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of filename_tag values to assemble. Default: all models in registry_shards.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path relative to repo root. Default: <bundle>/assembled/oof_matrix_dynamic.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = ROOT / args.bundle
    metadata_path = bundle_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json under {bundle_root}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / metadata["registry_shards_file"]
    shared_index_path = ROOT / metadata["shared_index_file"]

    registry_df = pd.read_csv(registry_path)
    if args.models:
        wanted = set(args.models)
        registry_df = registry_df[registry_df["filename_tag"].isin(wanted)].copy()
        missing = wanted.difference(set(registry_df["filename_tag"]))
        if missing:
            raise ValueError(f"Requested models not found in shard registry: {sorted(missing)}")

    shared = np.load(shared_index_path)
    assembled = pd.DataFrame(
        {
            "customer_id": shared["customer_id"],
            "target_monthly_spend": shared["target_monthly_spend"],
            "fold_id": shared["fold_id"],
        }
    )

    for row in registry_df.sort_values("rank").to_dict(orient="records"):
        shard_path = ROOT / row["prediction_shard_file"]
        shard = np.load(shard_path)
        assembled[row["filename_tag"]] = shard["oof_prediction"]

    if args.output:
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = bundle_root / "assembled"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "oof_matrix_dynamic.csv"
    assembled.to_csv(output_path, index=False)

    print(f"Bundle: {bundle_root.relative_to(ROOT)}")
    print(f"CV scheme: {metadata.get('cv_scheme', 'unknown')}")
    print(f"Rows: {len(assembled)}")
    print(f"Models assembled: {len(registry_df)}")
    print(f"Output: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
