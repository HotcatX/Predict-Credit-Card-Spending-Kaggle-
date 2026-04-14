from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

BUNDLES = [
    ROOT / "outputs" / "forward_selection_oof",
    ROOT / "outputs" / "forward_selection_oof_feature_engineered",
]


def convert_bundle(bundle_root: Path) -> None:
    registry_path = bundle_root / "registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry: {registry_path}")

    registry_df = pd.read_csv(registry_path)
    shard_dir = bundle_root / "shards"
    shared_dir = bundle_root / "shared"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous shard artifacts before re-export.
    for path in shard_dir.glob("*.npz"):
        path.unlink()
    for path in shared_dir.glob("*.npz"):
        path.unlink()

    # Use the first model file as the canonical alignment index.
    first_model_path = ROOT / registry_df.iloc[0]["oof_file"]
    first_df = pd.read_csv(first_model_path)
    shared_index_path = shared_dir / "index_arrays.npz"
    np.savez_compressed(
        shared_index_path,
        customer_id=first_df["customer_id"].to_numpy(dtype=np.int64),
        target_monthly_spend=first_df["target_monthly_spend"].to_numpy(dtype=np.float32),
        fold_id=first_df["fold_id"].to_numpy(dtype=np.int16),
    )

    shard_rows = []
    for row in registry_df.to_dict(orient="records"):
        csv_path = ROOT / row["oof_file"]
        model_df = pd.read_csv(csv_path)
        shard_name = csv_path.stem + ".npz"
        shard_path = shard_dir / shard_name
        np.savez_compressed(
            shard_path,
            oof_prediction=model_df["oof_prediction"].to_numpy(dtype=np.float32),
        )
        shard_rows.append(
            {
                **row,
                "storage": "npz_compressed",
                "shared_index_file": str(shared_index_path.relative_to(ROOT)),
                "prediction_shard_file": str(shard_path.relative_to(ROOT)),
                "prediction_dtype": "float32",
                "sample_count": int(len(model_df)),
            }
        )

    shard_registry_path = bundle_root / "registry_shards.csv"
    pd.DataFrame(shard_rows).to_csv(shard_registry_path, index=False)

    # Remove bulky per-model CSVs and static assembled matrix, replacing them with shard storage.
    for csv_path in (bundle_root / "models").glob("*.csv"):
        csv_path.unlink()
    models_dir = bundle_root / "models"
    if models_dir.exists():
        try:
            models_dir.rmdir()
        except OSError:
            pass

    for matrix_path in bundle_root.glob("oof_matrix_*.csv"):
        matrix_path.unlink()

    metadata_path = bundle_root / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "storage_format": "shared_index_npz + per_model_npz_shards",
            "registry_shards_file": str(shard_registry_path.relative_to(ROOT)),
            "shared_index_file": str(shared_index_path.relative_to(ROOT)),
            "dynamic_assembly_script": "assemble_oof_bundle.py",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_path = bundle_root / "README.md"
    lines = []
    if readme_path.exists():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    if lines:
        lines.append("")
    lines.extend(
        [
            "## Lightweight Storage",
            "",
            "- Shared alignment arrays are stored once in `shared/index_arrays.npz`.",
            "- Each model OOF prediction is stored separately as a compressed `.npz` shard in `shards/`.",
            "- Use `assemble_oof_bundle.py` to rebuild an aligned OOF matrix on demand.",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Converted bundle: {bundle_root.relative_to(ROOT)}")
    print(f"  shared index -> {shared_index_path.relative_to(ROOT)}")
    print(f"  shard registry -> {shard_registry_path.relative_to(ROOT)}")


def main() -> None:
    for bundle_root in BUNDLES:
        convert_bundle(bundle_root)


if __name__ == "__main__":
    main()
