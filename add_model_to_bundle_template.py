from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template script for adding one new aligned OOF model shard into an existing bundle."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Bundle path relative to repo root, e.g. outputs/forward_selection_oof",
    )
    parser.add_argument("--category", required=True, help="Registry category, e.g. advanced_more or basic_fe")
    parser.add_argument("--model-name", required=True, help="Human readable model name for registry output")
    parser.add_argument(
        "--filename-tag",
        required=True,
        help="Short slug used in registry and shard file naming, e.g. lightgbm_fe",
    )
    parser.add_argument(
        "--important-params",
        required=True,
        help="Short parameter summary used in naming, e.g. leaves63_lr0p03_iter2000_seed42",
    )
    parser.add_argument(
        "--feature-engineering",
        default=None,
        help="Optional FE label. Use this when adding to the FE bundle registry.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Optional explicit rank. Default: append after the current max rank.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing an existing registry row and shard for the same filename_tag.",
    )
    return parser.parse_args()


def sanitize_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace("+", "plus")
    token = token.replace(".", "p")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        raise ValueError("Token sanitization produced an empty value.")
    return token


def load_bundle(bundle_rel: str) -> dict[str, Any]:
    bundle_root = ROOT / bundle_rel
    metadata_path = bundle_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json under {bundle_root}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / metadata["registry_shards_file"]
    shared_index_path = ROOT / metadata["shared_index_file"]

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing shard registry: {registry_path}")
    if not shared_index_path.exists():
        raise FileNotFoundError(f"Missing shared index arrays: {shared_index_path}")

    registry_df = pd.read_csv(registry_path)
    shared = np.load(shared_index_path)

    required_shared_keys = {"customer_id", "target_monthly_spend", "fold_id"}
    if set(shared.files) != required_shared_keys:
        raise ValueError(
            f"Unexpected shared index keys: {sorted(shared.files)}. Expected {sorted(required_shared_keys)}."
        )

    return {
        "bundle_root": bundle_root,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "registry_path": registry_path,
        "registry_df": registry_df,
        "shared_index_path": shared_index_path,
        "customer_id": shared["customer_id"],
        "target": shared["target_monthly_spend"].astype(np.float32),
        "fold_id": shared["fold_id"],
    }


def load_training_data() -> pd.DataFrame:
    data_path = ROOT / "analysis_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Missing analysis_data.csv at repo root. The template assumes bundle rows align to this file."
        )
    return pd.read_csv(data_path)


def validate_alignment(train_df: pd.DataFrame, customer_id: np.ndarray, target: np.ndarray) -> None:
    if len(train_df) != len(customer_id):
        raise ValueError(
            f"Row count mismatch between training data ({len(train_df)}) and bundle index ({len(customer_id)})."
        )
    if "customer_id" not in train_df.columns or "monthly_spend" not in train_df.columns:
        raise ValueError("analysis_data.csv must contain both customer_id and monthly_spend.")
    if not np.array_equal(train_df["customer_id"].to_numpy(), customer_id):
        raise ValueError(
            "customer_id order mismatch. New OOF models must reuse the exact bundle row order."
        )
    if not np.allclose(train_df["monthly_spend"].to_numpy(dtype=np.float32), target):
        raise ValueError(
            "monthly_spend mismatch between analysis_data.csv and shared bundle target."
        )


def build_oof_predictions_template(train_df: pd.DataFrame, fold_id: np.ndarray) -> np.ndarray:
    """
    Replace this function with real model training.

    Contract:
    - Input rows are already aligned to `shared/index_arrays.npz`.
    - `fold_id` contains the canonical 5-fold assignment used by the bundle.
    - Return a 1D NumPy array of OOF predictions with length == len(train_df).
    - Each row must be predicted only from a model that was not trained on that row.
    """

    _ = fold_id  # Keep the variable explicit in the template.

    # Example skeleton:
    # features = train_df.drop(columns=["monthly_spend"])
    # target = train_df["monthly_spend"].to_numpy()
    # oof_pred = np.zeros(len(train_df), dtype=np.float32)
    # for valid_fold in sorted(np.unique(fold_id)):
    #     train_mask = fold_id != valid_fold
    #     valid_mask = fold_id == valid_fold
    #     X_train = features.loc[train_mask]
    #     y_train = target[train_mask]
    #     X_valid = features.loc[valid_mask]
    #     model = ...
    #     model.fit(X_train, y_train)
    #     oof_pred[valid_mask] = model.predict(X_valid)
    # return oof_pred

    raise NotImplementedError(
        "Replace build_oof_predictions_template() with the actual fold training and OOF prediction logic."
    )


def validate_oof_vector(oof_pred: np.ndarray, expected_len: int) -> np.ndarray:
    oof_pred = np.asarray(oof_pred, dtype=np.float32)
    if oof_pred.ndim != 1:
        raise ValueError(f"OOF prediction must be 1D. Got shape {oof_pred.shape}.")
    if len(oof_pred) != expected_len:
        raise ValueError(f"OOF length mismatch. Expected {expected_len}, got {len(oof_pred)}.")
    if not np.isfinite(oof_pred).all():
        raise ValueError("OOF prediction contains NaN or infinite values.")
    return oof_pred


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def next_rank(registry_df: pd.DataFrame) -> int:
    if registry_df.empty:
        return 1
    return int(registry_df["rank"].max()) + 1


def build_shard_filename(rank: int, filename_tag: str, important_params: str, rmse: float) -> str:
    return f"{rank:02d}_{filename_tag}__{important_params}__rmse{rmse:.4f}.npz"


def upsert_registry_row(registry_df: pd.DataFrame, new_row: dict[str, Any], overwrite_existing: bool) -> pd.DataFrame:
    exists = registry_df["filename_tag"] == new_row["filename_tag"]
    if exists.any():
        if not overwrite_existing:
            raise ValueError(
                f"filename_tag={new_row['filename_tag']} already exists. "
                "Use --overwrite-existing to replace it."
            )
        registry_df = registry_df.loc[~exists].copy()

    registry_df = pd.concat([registry_df, pd.DataFrame([new_row])], ignore_index=True)
    return registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    filename_tag = sanitize_token(args.filename_tag)
    important_params = sanitize_token(args.important_params)

    bundle = load_bundle(args.bundle)
    train_df = load_training_data()
    validate_alignment(train_df, bundle["customer_id"], bundle["target"])

    print(f"Bundle: {Path(args.bundle)}")
    print(f"CV scheme: {bundle['metadata'].get('cv_scheme', 'unknown')}")
    print(f"Shared index: {Path(bundle['metadata']['shared_index_file'])}")
    print("Training template model against canonical fold_id alignment...")

    oof_pred = build_oof_predictions_template(train_df, bundle["fold_id"])
    oof_pred = validate_oof_vector(oof_pred, expected_len=len(train_df))
    rmse = compute_rmse(bundle["target"], oof_pred)

    rank = args.rank if args.rank is not None else next_rank(bundle["registry_df"])
    shard_name = build_shard_filename(rank, filename_tag, important_params, rmse)
    shard_path = bundle["bundle_root"] / "shards" / shard_name
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard_path, oof_prediction=oof_pred)

    new_row: dict[str, Any] = {
        "rank": int(rank),
        "category": args.category,
        "model_name": args.model_name,
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": rmse,
        "oof_file": "",
        "storage": "npz_compressed",
        "shared_index_file": str(Path(bundle["metadata"]["shared_index_file"])),
        "prediction_shard_file": str(shard_path.relative_to(ROOT)),
        "prediction_dtype": "float32",
        "sample_count": int(len(oof_pred)),
    }
    if "feature_engineering" in bundle["registry_df"].columns or args.feature_engineering is not None:
        new_row["feature_engineering"] = args.feature_engineering or ""

    updated_registry = upsert_registry_row(
        registry_df=bundle["registry_df"],
        new_row=new_row,
        overwrite_existing=args.overwrite_existing,
    )

    preferred_columns = [
        "rank",
        "category",
        "model_name",
        "filename_tag",
        "important_params",
        "oof_rmse_cv5_rs42",
        "feature_engineering",
        "oof_file",
        "storage",
        "shared_index_file",
        "prediction_shard_file",
        "prediction_dtype",
        "sample_count",
    ]
    final_columns = [col for col in preferred_columns if col in updated_registry.columns]
    final_columns.extend([col for col in updated_registry.columns if col not in final_columns])
    updated_registry = updated_registry[final_columns]
    updated_registry.to_csv(bundle["registry_path"], index=False)

    metadata = bundle["metadata"]
    metadata["model_count"] = int(len(updated_registry))
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Added shard successfully.")
    print(f"Model: {args.model_name}")
    print(f"filename_tag: {filename_tag}")
    print(f"important_params: {important_params}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Shard: {shard_path.relative_to(ROOT)}")
    print(f"Registry: {bundle['registry_path'].relative_to(ROOT)}")
    print("")
    print("Next step:")
    print("  Replace build_oof_predictions_template() with real fold training logic for the new model.")


if __name__ == "__main__":
    main()
