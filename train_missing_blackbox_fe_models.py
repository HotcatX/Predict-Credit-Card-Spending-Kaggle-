from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof_feature_engineered"
FE_LABEL = "v1_interactions_ratios_category_crosses"

ORIGINAL_CATEGORICAL = [
    "gender",
    "marital_status",
    "education_level",
    "region",
    "employment_status",
    "card_type",
]


def load_bundle() -> dict[str, object]:
    metadata_path = BUNDLE_ROOT / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / metadata["registry_shards_file"]
    legacy_registry_path = ROOT / metadata["registry_file"]
    shared_index_path = ROOT / metadata["shared_index_file"]
    shared = np.load(shared_index_path)
    registry_df = pd.read_csv(registry_path)
    legacy_registry_df = pd.read_csv(legacy_registry_path)
    return {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "registry_path": registry_path,
        "legacy_registry_path": legacy_registry_path,
        "registry_df": registry_df,
        "legacy_registry_df": legacy_registry_df,
        "customer_id": shared["customer_id"],
        "target": shared["target_monthly_spend"].astype(np.float32),
        "fold_id": shared["fold_id"],
        "shared_index_file": metadata["shared_index_file"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the previously omitted black-box FE models and append them to the FE bundle.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["mlp_fe", "catboost_fe", "stacking_fe"],
        help="Subset of model keys to run. Choices: mlp_fe catboost_fe stacking_fe",
    )
    return parser.parse_args()


def load_train() -> pd.DataFrame:
    train_df = pd.read_csv(ROOT / "analysis_data.csv")
    bundle = load_bundle()
    if not np.array_equal(train_df["customer_id"].to_numpy(), bundle["customer_id"]):
        raise ValueError("analysis_data.csv is not aligned with FE bundle customer_id order.")
    if not np.allclose(train_df["monthly_spend"].to_numpy(dtype=np.float32), bundle["target"]):
        raise ValueError("analysis_data.csv monthly_spend does not match FE bundle target.")
    return train_df


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


def get_feature_lists(train_fe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    target_col = "monthly_spend"
    drop_cols = {"customer_id", target_col}
    categorical_cols = ORIGINAL_CATEGORICAL + [
        "fe_home_loan_combo",
        "fe_region_card_type",
        "fe_education_employment",
        "fe_marital_children_band",
    ]
    numeric_cols = [c for c in train_fe.columns if c not in drop_cols.union(categorical_cols)]
    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_mlp_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=5e-4,
        max_iter=360,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=123,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def make_stack_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> StackingRegressor:
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    estimators = [
        (
            "ridge",
            Pipeline([("prep", linear_preprocessor), ("model", Ridge(alpha=0.5))]),
        ),
        (
            "extra",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=250,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.03,
                            max_depth=8,
                            max_iter=700,
                            l2_regularization=0.5,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=5,
        passthrough=False,
        n_jobs=1,
    )


def fit_predict_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: list[str],
) -> np.ndarray:
    X_train_cb = X_train.copy()
    X_valid_cb = X_valid.copy()
    for col in categorical_cols:
        X_train_cb[col] = X_train_cb[col].fillna("missing").astype(str)
        X_valid_cb[col] = X_valid_cb[col].fillna("missing").astype(str)

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.03,
        iterations=4000,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
    )
    model.fit(
        X_train_cb,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_valid_cb, y_valid),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False,
    )
    return model.predict(X_valid_cb)


def run_oof_model(
    model_key: str,
    train_fe: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    fold_id: np.ndarray,
) -> np.ndarray:
    y = train_fe["monthly_spend"].to_numpy(dtype=np.float32)
    X = train_fe[feature_cols]
    oof_pred = np.zeros(len(train_fe), dtype=np.float32)

    for valid_fold in sorted(np.unique(fold_id)):
        train_mask = fold_id != valid_fold
        valid_mask = fold_id == valid_fold
        X_train = X.loc[train_mask]
        y_train = train_fe.loc[train_mask, "monthly_spend"]
        X_valid = X.loc[valid_mask]
        y_valid = train_fe.loc[valid_mask, "monthly_spend"]

        if model_key == "mlp_fe":
            model = make_mlp_pipeline(numeric_cols, categorical_cols)
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_valid)
        elif model_key == "catboost_fe":
            fold_pred = fit_predict_catboost(X_train, y_train, X_valid, y_valid, categorical_cols)
        elif model_key == "stacking_fe":
            model = make_stack_pipeline(numeric_cols, categorical_cols)
            model.fit(X_train, y_train)
            fold_pred = model.predict(X_valid)
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        oof_pred[valid_mask] = np.asarray(fold_pred, dtype=np.float32)
        print(f"{model_key}: fold {valid_fold} rmse={rmse(y_valid.to_numpy(), oof_pred[valid_mask]):.6f}")

    print(f"{model_key}: overall rmse={rmse(y, oof_pred):.6f}")
    return oof_pred


def remove_existing_files(registry_df: pd.DataFrame, legacy_registry_df: pd.DataFrame, filename_tag: str) -> None:
    for df in (registry_df, legacy_registry_df):
        rows = df.loc[df["filename_tag"] == filename_tag]
        for _, row in rows.iterrows():
            shard_path = row.get("prediction_shard_file")
            if isinstance(shard_path, str) and shard_path:
                path = ROOT / shard_path
                if path.exists():
                    path.unlink()


def append_model(
    bundle: dict[str, object],
    filename_tag: str,
    model_name: str,
    category: str,
    important_params: str,
    oof_pred: np.ndarray,
) -> None:
    registry_df = bundle["registry_df"].copy()
    legacy_registry_df = bundle["legacy_registry_df"].copy()
    remove_existing_files(registry_df, legacy_registry_df, filename_tag)
    registry_df = registry_df.loc[registry_df["filename_tag"] != filename_tag].copy()
    legacy_registry_df = legacy_registry_df.loc[legacy_registry_df["filename_tag"] != filename_tag].copy()

    rank = int(registry_df["rank"].max()) + 1 if not registry_df.empty else 1
    score = rmse(bundle["target"], oof_pred)
    shard_name = f"{rank:02d}_{filename_tag}__{important_params}__rmse{score:.4f}.npz"
    shard_rel = Path("outputs") / "forward_selection_oof_feature_engineered" / "shards" / shard_name
    shard_abs = ROOT / shard_rel
    np.savez_compressed(shard_abs, oof_prediction=oof_pred.astype(np.float32))

    shard_row = {
        "rank": rank,
        "category": category,
        "model_name": model_name,
        "filename_tag": filename_tag,
        "important_params": important_params,
        "oof_rmse_cv5_rs42": score,
        "feature_engineering": FE_LABEL,
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
        "feature_engineering": FE_LABEL,
        "oof_file": "",
    }

    registry_df = pd.concat([registry_df, pd.DataFrame([shard_row])], ignore_index=True)
    legacy_registry_df = pd.concat([legacy_registry_df, pd.DataFrame([legacy_row])], ignore_index=True)
    registry_df = registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    legacy_registry_df = legacy_registry_df.sort_values(["rank", "filename_tag"]).reset_index(drop=True)
    registry_df.to_csv(bundle["registry_path"], index=False)
    legacy_registry_df.to_csv(bundle["legacy_registry_path"], index=False)

    print(f"saved {filename_tag}: rmse={score:.6f} shard={shard_rel}")


def update_readme_and_metadata(bundle: dict[str, object]) -> None:
    metadata = bundle["metadata"]
    registry_df = pd.read_csv(bundle["registry_path"])
    metadata["model_count"] = int(len(registry_df))
    metadata["excluded_models"] = []
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_path = BUNDLE_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace(
        "- This bundle excludes `MLPRegressor`, `CatBoostRegressor`, and `StackingRegressor`.\n",
        "- This bundle now includes the previously omitted black-box FE models: `MLPRegressor`, `CatBoostRegressor`, and `StackingRegressor`.\n",
    )
    readme_path.write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle = load_bundle()
    train_df = pd.read_csv(ROOT / "analysis_data.csv")
    if not np.array_equal(train_df["customer_id"].to_numpy(), bundle["customer_id"]):
        raise ValueError("analysis_data.csv is not aligned with FE bundle customer_id order.")
    train_fe = add_feature_engineering(train_df)
    feature_cols, numeric_cols, categorical_cols = get_feature_lists(train_fe)

    configs = [
        (
            "mlp_fe",
            "MLPRegressor FE",
            "advanced_more_fe",
            "mlp_256_128_64_fe",
            "fe_h256x128x64_relu_bs256_lr0p0005_alpha1e_4_iter360_seed123",
        ),
        (
            "catboost_fe",
            "CatBoostRegressor FE",
            "advanced_more_fe",
            "catboost_depth8_fe",
            "fe_depth8_lr0p03_iter4000_seed42",
        ),
        (
            "stacking_fe",
            "StackingRegressor FE",
            "advanced_more_fe",
            "stack_ridge_extra_hist_fe",
            "fe_meta_ridge0p5_base_ridge_extra250_hist700",
        ),
    ]

    requested = set(args.models)
    valid_keys = {cfg[0] for cfg in configs}
    invalid = requested.difference(valid_keys)
    if invalid:
        raise ValueError(f"Unknown model keys requested: {sorted(invalid)}")

    for model_key, model_name, category, filename_tag, important_params in configs:
        if model_key not in requested:
            continue
        print("")
        print(f"=== running {model_name} ===")
        oof_pred = run_oof_model(
            model_key=model_key,
            train_fe=train_fe,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            fold_id=bundle["fold_id"],
        )
        append_model(
            bundle=bundle,
            filename_tag=filename_tag,
            model_name=model_name,
            category=category,
            important_params=important_params,
            oof_pred=oof_pred,
        )
        bundle = load_bundle()

    update_readme_and_metadata(bundle)
    print("")
    print("Updated FE bundle metadata and README.")


if __name__ == "__main__":
    main()
