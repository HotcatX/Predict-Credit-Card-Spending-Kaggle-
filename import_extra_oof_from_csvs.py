from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "anotherOOPoutput"
BASE_BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof"
FE_BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof_feature_engineered"
FE_LABEL = "external_aggressive_or_alternate_fe"


EXTRA_SPECS = [
    {
        "source_file": "OOF_CatBoost_Pro_RMSE157.2800.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "CatBoost Pro (Extra OOF)",
        "filename_tag": "extra_catboost_pro",
        "prediction_col": "oof_catboost",
    },
    {
        "source_file": "OOF_FTTransformer_V2_RMSE133.1912.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "FTTransformer V2 (Extra OOF)",
        "filename_tag": "extra_fttransformer_v2",
        "prediction_col": "oof_ftt",
    },
    {
        "source_file": "OOF_MLP_FEAggressive_RMSE126.8311.csv",
        "bundle_root": FE_BUNDLE_ROOT,
        "category": "extra_oof_fe",
        "model_name": "MLP FE Aggressive (Extra OOF)",
        "filename_tag": "extra_mlp_fe_aggressive",
        "prediction_col": "oof_mlp",
        "feature_engineering": FE_LABEL,
    },
    {
        "source_file": "OOF_MLP_StableV2_RMSE125.6784.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "MLP Stable V2 (Extra OOF)",
        "filename_tag": "extra_mlp_stable_v2",
        "prediction_col": "oof_mlp",
    },
    {
        "source_file": "OOF_TabM_Baseline_RMSE131.4686.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "TabM Baseline (Extra OOF)",
        "filename_tag": "extra_tabm_baseline",
        "prediction_col": "oof_tabm",
    },
    {
        "source_file": "OOF_TabNet_FEAggressive_RMSE125.5492.csv",
        "bundle_root": FE_BUNDLE_ROOT,
        "category": "extra_oof_fe",
        "model_name": "TabNet FE Aggressive (Extra OOF)",
        "filename_tag": "extra_tabnet_fe_aggressive",
        "prediction_col": "oof_tabnet",
        "feature_engineering": FE_LABEL,
    },
    {
        "source_file": "OOF_TabNet_FE_RMSE124.3205.csv",
        "bundle_root": FE_BUNDLE_ROOT,
        "category": "extra_oof_fe",
        "model_name": "TabNet FE (Extra OOF)",
        "filename_tag": "extra_tabnet_fe",
        "prediction_col": "oof_tabnet",
        "feature_engineering": FE_LABEL,
    },
    {
        "source_file": "OOF_TabNet_Pro_RMSE124.6600.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "TabNet Pro (Extra OOF)",
        "filename_tag": "extra_tabnet_pro",
        "prediction_col": "oof_tabnet",
    },
    {
        "source_file": "OOF_TabNet_StableV2_RMSE124.9266.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "TabNet Stable V2 (Extra OOF)",
        "filename_tag": "extra_tabnet_stable_v2",
        "prediction_col": "oof_tabnet",
    },
    {
        "source_file": "OOF_TabPFN_RMSE128.7900.csv",
        "bundle_root": BASE_BUNDLE_ROOT,
        "category": "extra_oof",
        "model_name": "TabPFN (Extra OOF)",
        "filename_tag": "extra_tabpfn",
        "prediction_col": "oof_tabpfn",
    },
]


def sanitize_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace("+", "plus").replace(".", "p")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        raise ValueError("Token sanitization produced an empty value.")
    return token


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_bundle(bundle_root: Path) -> dict[str, object]:
    metadata_path = bundle_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / metadata["registry_shards_file"]
    legacy_registry_path = ROOT / metadata["registry_file"]
    shared_index_path = ROOT / metadata["shared_index_file"]
    shared = np.load(shared_index_path)
    return {
        "bundle_root": bundle_root,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "registry_path": registry_path,
        "legacy_registry_path": legacy_registry_path,
        "registry_df": pd.read_csv(registry_path),
        "legacy_registry_df": pd.read_csv(legacy_registry_path),
        "shared_index_file": metadata["shared_index_file"],
        "customer_id": shared["customer_id"],
        "target": shared["target_monthly_spend"].astype(np.float32),
    }


def important_params_from_name(filename: str) -> str:
    match = re.search(r"RMSE([0-9]+\.[0-9]+)", filename, flags=re.IGNORECASE)
    reported = match.group(1).replace(".", "p") if match else "unknown"
    stem = Path(filename).stem
    stem = stem.replace("OOF_", "")
    stem = re.sub(r"_RMSE[0-9]+\.[0-9]+$", "", stem, flags=re.IGNORECASE)
    return sanitize_token(f"source_{stem}_reported_{reported}")


def prepare_prediction_frame(spec: dict[str, object], bundle: dict[str, object]) -> tuple[pd.DataFrame, np.ndarray, float]:
    source_path = SOURCE_DIR / str(spec["source_file"])
    df = pd.read_csv(source_path)
    pred_col = str(spec["prediction_col"])
    if pred_col not in df.columns:
        raise ValueError(f"{source_path.name} is missing prediction column {pred_col}")
    if "customer_id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{source_path.name} must contain customer_id and target columns")
    if len(df) != len(bundle["customer_id"]):
        raise ValueError(f"{source_path.name} row count mismatch")
    if not np.array_equal(df["customer_id"].to_numpy(), bundle["customer_id"]):
        raise ValueError(f"{source_path.name} customer_id order mismatch")
    if not np.allclose(df["target"].to_numpy(dtype=np.float32), bundle["target"]):
        raise ValueError(f"{source_path.name} target mismatch")
    pred = df[pred_col].to_numpy(dtype=np.float32)
    if not np.isfinite(pred).all():
        raise ValueError(f"{source_path.name} prediction contains NaN/inf")
    score = compute_rmse(bundle["target"], pred)
    return df, pred, score


def remove_existing_artifact_if_any(registry_df: pd.DataFrame, filename_tag: str) -> None:
    rows = registry_df.loc[registry_df["filename_tag"] == filename_tag]
    for _, row in rows.iterrows():
        shard_rel = row.get("prediction_shard_file")
        if isinstance(shard_rel, str) and shard_rel:
            path = ROOT / shard_rel
            if path.exists():
                path.unlink()


def append_extra_oof(spec: dict[str, object]) -> None:
    bundle = load_bundle(Path(spec["bundle_root"]))
    _df, pred, score = prepare_prediction_frame(spec, bundle)

    filename_tag = sanitize_token(str(spec["filename_tag"]))
    remove_existing_artifact_if_any(bundle["registry_df"], filename_tag)
    registry_df = bundle["registry_df"].loc[bundle["registry_df"]["filename_tag"] != filename_tag].copy()
    legacy_registry_df = bundle["legacy_registry_df"].loc[
        bundle["legacy_registry_df"]["filename_tag"] != filename_tag
    ].copy()

    rank = int(registry_df["rank"].max()) + 1 if not registry_df.empty else 1
    important_params = important_params_from_name(str(spec["source_file"]))
    shard_name = f"{rank:02d}_{filename_tag}__{important_params}__rmse{score:.4f}.npz"
    shard_rel = Path(spec["bundle_root"]).relative_to(ROOT) / "shards" / shard_name
    np.savez_compressed(ROOT / shard_rel, oof_prediction=pred.astype(np.float32))

    shard_row = {
        "rank": rank,
        "category": spec["category"],
        "model_name": spec["model_name"],
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": score,
        "oof_file": str(SOURCE_DIR.relative_to(ROOT) / str(spec["source_file"])),
        "storage": "npz_compressed",
        "shared_index_file": bundle["shared_index_file"],
        "prediction_shard_file": str(shard_rel),
        "prediction_dtype": "float32",
        "sample_count": int(len(pred)),
    }
    legacy_row = {
        "rank": rank,
        "category": spec["category"],
        "model_name": spec["model_name"],
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": score,
        "oof_file": str(SOURCE_DIR.relative_to(ROOT) / str(spec["source_file"])),
    }

    if "feature_engineering" in registry_df.columns or "feature_engineering" in spec:
        shard_row["feature_engineering"] = str(spec.get("feature_engineering", ""))
        legacy_row["feature_engineering"] = str(spec.get("feature_engineering", ""))

    registry_df = pd.concat([registry_df, pd.DataFrame([shard_row])], ignore_index=True)
    legacy_registry_df = pd.concat([legacy_registry_df, pd.DataFrame([legacy_row])], ignore_index=True)
    registry_df = registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    legacy_registry_df = legacy_registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    registry_df.to_csv(bundle["registry_path"], index=False)
    legacy_registry_df.to_csv(bundle["legacy_registry_path"], index=False)

    metadata = bundle["metadata"]
    metadata["model_count"] = int(len(registry_df))
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Imported {spec['source_file']} -> {spec['model_name']} | rmse={score:.4f}")


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source dir: {SOURCE_DIR}")
    for spec in EXTRA_SPECS:
        append_extra_oof(spec)


if __name__ == "__main__":
    main()
