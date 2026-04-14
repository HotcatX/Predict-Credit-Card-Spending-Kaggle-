from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
BASE_BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof"
FE_BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof_feature_engineered"
FE_LABEL = "v1_interactions_ratios_category_crosses"
SEED = 123

ORIGINAL_CATEGORICAL = [
    "gender",
    "marital_status",
    "education_level",
    "region",
    "employment_status",
    "card_type",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        "fold_id": shared["fold_id"],
    }


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1.0
    out["fe_total_txn_value"] = out["num_transactions"] * out["avg_transaction_value"]
    out["fe_online_ratio"] = out["online_shopping_freq"] / (out["num_transactions"] + eps)
    out["fe_utility_ratio"] = out["utility_payment_count"] / (out["num_transactions"] + eps)
    out["fe_reward_per_txn"] = out["reward_points_balance"] / (out["num_transactions"] + eps)
    out["fe_income_per_card"] = out["annual_income"] / (out["num_credit_cards"] + eps)
    out["fe_limit_per_card"] = out["credit_limit"] / (out["num_credit_cards"] + eps)
    out["fe_income_to_limit"] = out["annual_income"] / (out["credit_limit"] + eps)
    out["fe_txn_value_to_limit"] = out["fe_total_txn_value"] / (out["credit_limit"] + eps)
    out["fe_tenure_to_age"] = out["tenure"] / (out["age"] + eps)
    out["fe_income_per_child"] = out["annual_income"] / (out["num_children"] + eps)
    out["fe_score_x_limit"] = out["credit_score"] * out["credit_limit"]
    out["fe_score_x_income"] = out["credit_score"] * out["annual_income"]
    out["fe_travel_x_txn"] = out["travel_frequency"] * out["num_transactions"]
    out["fe_home_loan_combo"] = (
        out["owns_home"].fillna(0).astype("Int64").astype(str)
        + "_"
        + out["has_auto_loan"].fillna(0).astype("Int64").astype(str)
    )
    out["fe_region_card_type"] = out["region"].fillna("missing") + "_" + out["card_type"].fillna("missing")
    out["fe_education_employment"] = (
        out["education_level"].fillna("missing") + "_" + out["employment_status"].fillna("missing")
    )
    child_band = pd.cut(
        out["num_children"].fillna(0),
        bins=[-1, 0, 2, 20],
        labels=["0", "1_2", "3_plus"],
    ).astype(str)
    out["fe_marital_children_band"] = out["marital_status"].fillna("missing") + "_" + child_band
    return out


def get_columns(train_df: pd.DataFrame, include_fe: bool) -> tuple[list[str], list[str]]:
    categorical_cols = ORIGINAL_CATEGORICAL.copy()
    if include_fe:
        categorical_cols += [
            "fe_home_loan_combo",
            "fe_region_card_type",
            "fe_education_employment",
            "fe_marital_children_band",
        ]
    numeric_cols = [
        c
        for c in train_df.columns
        if c not in set(categorical_cols).union({"customer_id", "monthly_spend"})
    ]
    return numeric_cols, categorical_cols


def encode_for_tabnet(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], StandardScaler]:
    X_train = train_df[numeric_cols].copy()
    X_valid = valid_df[numeric_cols].copy()

    for col in numeric_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_valid[col] = X_valid[col].fillna(median)

    scaler = StandardScaler()
    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols]).astype(np.float32)
    X_valid.loc[:, numeric_cols] = scaler.transform(X_valid[numeric_cols]).astype(np.float32)

    cat_dims: list[int] = []
    cat_idxs: list[int] = []
    offset = len(numeric_cols)

    for idx, col in enumerate(categorical_cols):
        train_values = train_df[col].fillna("missing").astype(str)
        valid_values = valid_df[col].fillna("missing").astype(str)
        uniques = sorted(train_values.unique().tolist())
        mapping = {value: code + 1 for code, value in enumerate(uniques)}
        X_train[col] = train_values.map(mapping).fillna(0).astype(np.int64)
        X_valid[col] = valid_values.map(mapping).fillna(0).astype(np.int64)
        cat_dims.append(len(mapping) + 1)
        cat_idxs.append(offset + idx)

    ordered_cols = numeric_cols + categorical_cols
    return (
        X_train[ordered_cols].to_numpy(dtype=np.float32),
        X_valid[ordered_cols].to_numpy(dtype=np.float32),
        cat_idxs,
        cat_dims,
        scaler,
    )


def fit_fold_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_idxs: list[int],
    cat_dims: list[int],
) -> np.ndarray:
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_valid_scaled = y_scaler.transform(y_valid.reshape(-1, 1)).astype(np.float32)

    model = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.1,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 1e-2},
        scheduler_params={"step_size": 10, "gamma": 0.8},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        seed=SEED,
        verbose=0,
    )
    model.fit(
        X_train=X_train,
        y_train=y_train_scaled,
        eval_set=[(X_valid, y_valid_scaled)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=20,
        patience=5,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )
    preds_scaled = model.predict(X_valid).reshape(-1, 1).astype(np.float32)
    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1).astype(np.float32)
    return preds


def append_model(
    bundle: dict[str, object],
    filename_tag: str,
    model_name: str,
    category: str,
    important_params: str,
    oof_pred: np.ndarray,
    feature_engineering: str | None,
) -> float:
    registry_df = bundle["registry_df"].copy()
    legacy_registry_df = bundle["legacy_registry_df"].copy()
    registry_df = registry_df.loc[registry_df["filename_tag"] != filename_tag].copy()
    legacy_registry_df = legacy_registry_df.loc[legacy_registry_df["filename_tag"] != filename_tag].copy()

    score = rmse(bundle["target"], oof_pred)
    rank = int(registry_df["rank"].max()) + 1 if not registry_df.empty else 1
    shard_name = f"{rank:02d}_{filename_tag}__{important_params}__rmse{score:.4f}.npz"
    shard_rel = bundle["bundle_root"].relative_to(ROOT) / "shards" / shard_name
    np.savez_compressed(ROOT / shard_rel, oof_prediction=oof_pred.astype(np.float32))

    shard_row = {
        "rank": rank,
        "category": category,
        "model_name": model_name,
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": score,
        "oof_file": "",
        "storage": "npz_compressed",
        "shared_index_file": bundle["shared_index_file"],
        "prediction_shard_file": str(shard_rel),
        "prediction_dtype": "float32",
        "sample_count": int(len(oof_pred)),
    }
    legacy_row = {
        "rank": rank,
        "category": category,
        "model_name": model_name,
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": score,
        "oof_file": "",
    }
    if feature_engineering is not None or "feature_engineering" in registry_df.columns:
        shard_row["feature_engineering"] = feature_engineering or ""
        legacy_row["feature_engineering"] = feature_engineering or ""

    registry_df = pd.concat([registry_df, pd.DataFrame([shard_row])], ignore_index=True)
    legacy_registry_df = pd.concat([legacy_registry_df, pd.DataFrame([legacy_row])], ignore_index=True)
    registry_df = registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    legacy_registry_df = legacy_registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    registry_df.to_csv(bundle["registry_path"], index=False)
    legacy_registry_df.to_csv(bundle["legacy_registry_path"], index=False)

    metadata = bundle["metadata"]
    metadata["model_count"] = int(len(registry_df))
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return score


def run_bundle(bundle_root: Path, include_fe: bool) -> float:
    bundle = load_bundle(bundle_root)
    train_df = pd.read_csv(ROOT / "analysis_data.csv")
    if not np.array_equal(train_df["customer_id"].to_numpy(), bundle["customer_id"]):
        raise ValueError(f"analysis_data.csv customer_id order does not match {bundle_root.name}.")
    if include_fe:
        train_df = add_feature_engineering(train_df)

    numeric_cols, categorical_cols = get_columns(train_df, include_fe=include_fe)
    fold_id = bundle["fold_id"]
    oof_pred = np.zeros(len(train_df), dtype=np.float32)

    for valid_fold in sorted(np.unique(fold_id)):
        train_mask = fold_id != valid_fold
        valid_mask = fold_id == valid_fold
        train_fold_df = train_df.loc[train_mask].reset_index(drop=True)
        valid_fold_df = train_df.loc[valid_mask].reset_index(drop=True)
        X_train, X_valid, cat_idxs, cat_dims, _num_scaler = encode_for_tabnet(
            train_fold_df, valid_fold_df, numeric_cols, categorical_cols
        )
        y_train = train_fold_df["monthly_spend"].to_numpy(dtype=np.float32)
        y_valid = valid_fold_df["monthly_spend"].to_numpy(dtype=np.float32)
        fold_pred = fit_fold_tabnet(X_train, y_train, X_valid, y_valid, cat_idxs, cat_dims)
        oof_pred[valid_mask] = fold_pred
        print(
            f"{'tabnet_fe' if include_fe else 'tabnet_base'}: "
            f"fold {valid_fold} rmse={rmse(y_valid, fold_pred):.6f}"
        )

    score = append_model(
        bundle=bundle,
        filename_tag="tabnet_fe" if include_fe else "tabnet_base",
        model_name="TabNet FE" if include_fe else "TabNet",
        category="advanced_more_fe" if include_fe else "advanced_more",
        important_params="fe_nd8_na8_steps3_gamma1p1_lr1em02_ep20_pat5_seed123"
        if include_fe
        else "nd8_na8_steps3_gamma1p1_lr1em02_ep20_pat5_seed123",
        oof_pred=oof_pred,
        feature_engineering=FE_LABEL if include_fe else None,
    )
    print(f"{'tabnet_fe' if include_fe else 'tabnet_base'}: overall rmse={score:.6f}")
    return score


def main() -> None:
    seed_everything(SEED)
    run_bundle(BASE_BUNDLE_ROOT, include_fe=False)
    run_bundle(FE_BUNDLE_ROOT, include_fe=True)


if __name__ == "__main__":
    main()
