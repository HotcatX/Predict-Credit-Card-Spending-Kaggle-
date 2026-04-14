from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT / "outputs" / "forward_selection_oof_feature_engineered"
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


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_bundle() -> dict[str, object]:
    metadata_path = BUNDLE_ROOT / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / metadata["registry_shards_file"]
    legacy_registry_path = ROOT / metadata["registry_file"]
    shared_index_path = ROOT / metadata["shared_index_file"]
    shared = np.load(shared_index_path)
    return {
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


def get_feature_lists(train_fe: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = ORIGINAL_CATEGORICAL + [
        "fe_home_loan_combo",
        "fe_region_card_type",
        "fe_education_employment",
        "fe_marital_children_band",
    ]
    numeric_cols = [
        c
        for c in train_fe.columns
        if c not in set(categorical_cols).union({"customer_id", "monthly_spend"})
    ]
    return numeric_cols, categorical_cols


def build_category_maps(train_df: pd.DataFrame, categorical_cols: list[str]) -> tuple[dict[str, dict[str, int]], list[int]]:
    cat_maps: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for col in categorical_cols:
        values = train_df[col].fillna("missing").astype(str)
        uniques = sorted(values.unique().tolist())
        mapping = {value: idx + 1 for idx, value in enumerate(uniques)}
        cat_maps[col] = mapping
        cardinalities.append(len(mapping) + 1)
    return cat_maps, cardinalities


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str], cat_maps: dict[str, dict[str, int]]) -> np.ndarray:
    encoded = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for idx, col in enumerate(categorical_cols):
        values = df[col].fillna("missing").astype(str)
        mapping = cat_maps[col]
        encoded[:, idx] = values.map(mapping).fillna(0).astype(np.int64).to_numpy()
    return encoded


class TabularDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray) -> None:
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class EmbeddingMLP(nn.Module):
    def __init__(self, num_numeric: int, cardinalities: list[int], hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        emb_total_dim = 0
        for cardinality in cardinalities:
            emb_dim = min(32, max(4, int(math.ceil(cardinality ** 0.25 * 2))))
            self.embeddings.append(nn.Embedding(cardinality, emb_dim))
            emb_total_dim += emb_dim

        input_dim = num_numeric + emb_total_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb_parts = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb_parts, dim=1)
        return self.mlp(x).squeeze(1)


def train_one_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numeric_cols].fillna(train_df[numeric_cols].median()))
    X_valid_num = scaler.transform(valid_df[numeric_cols].fillna(train_df[numeric_cols].median()))

    cat_maps, cardinalities = build_category_maps(train_df, categorical_cols)
    X_train_cat = encode_categoricals(train_df, categorical_cols, cat_maps)
    X_valid_cat = encode_categoricals(valid_df, categorical_cols, cat_maps)

    y_train = train_df["monthly_spend"].to_numpy(dtype=np.float32)
    y_valid = valid_df["monthly_spend"].to_numpy(dtype=np.float32)

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    valid_dataset = TabularDataset(X_valid_num, X_valid_cat, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)

    model = EmbeddingMLP(
        num_numeric=len(numeric_cols),
        cardinalities=cardinalities,
        hidden_dims=[256, 128, 64],
        dropout=0.10,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_state = None
    best_valid_rmse = float("inf")
    patience = 12
    no_improve = 0

    for _epoch in range(80):
        model.train()
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb_num, xb_cat)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_preds = []
        with torch.no_grad():
            for xb_num, xb_cat, _yb in valid_loader:
                xb_num = xb_num.to(device)
                xb_cat = xb_cat.to(device)
                batch_preds = model(xb_num, xb_cat).detach().cpu().numpy()
                valid_preds.append(batch_preds)
        valid_pred = np.concatenate(valid_preds).astype(np.float32)
        valid_score = rmse(y_valid, valid_pred)

        if valid_score < best_valid_rmse:
            best_valid_rmse = valid_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Embedding MLP failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    final_preds = []
    with torch.no_grad():
        for xb_num, xb_cat, _yb in valid_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            batch_preds = model(xb_num, xb_cat).detach().cpu().numpy()
            final_preds.append(batch_preds)
    return np.concatenate(final_preds).astype(np.float32)


def append_model(bundle: dict[str, object], filename_tag: str, model_name: str, category: str, important_params: str, oof_pred: np.ndarray) -> float:
    registry_df = bundle["registry_df"].copy()
    legacy_registry_df = bundle["legacy_registry_df"].copy()
    registry_df = registry_df.loc[registry_df["filename_tag"] != filename_tag].copy()
    legacy_registry_df = legacy_registry_df.loc[legacy_registry_df["filename_tag"] != filename_tag].copy()

    score = rmse(bundle["target"], oof_pred)
    rank = int(registry_df["rank"].max()) + 1 if not registry_df.empty else 1
    shard_name = f"{rank:02d}_{filename_tag}__{important_params}__rmse{score:.4f}.npz"
    shard_rel = Path("outputs") / "forward_selection_oof_feature_engineered" / "shards" / shard_name
    np.savez_compressed(ROOT / shard_rel, oof_prediction=oof_pred.astype(np.float32))

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
    registry_df = pd.concat([registry_df, pd.DataFrame([shard_row])], ignore_index=True).sort_values(["rank", "filename_tag"])
    legacy_registry_df = pd.concat([legacy_registry_df, pd.DataFrame([legacy_row])], ignore_index=True).sort_values(["rank", "filename_tag"])
    registry_df.to_csv(bundle["registry_path"], index=False)
    legacy_registry_df.to_csv(bundle["legacy_registry_path"], index=False)

    metadata = bundle["metadata"]
    metadata["model_count"] = int(len(registry_df))
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return score


def main() -> None:
    seed_everything(SEED)
    device = get_device()
    bundle = load_bundle()

    train_df = pd.read_csv(ROOT / "analysis_data.csv")
    if not np.array_equal(train_df["customer_id"].to_numpy(), bundle["customer_id"]):
        raise ValueError("analysis_data.csv customer_id order does not match FE bundle.")
    train_fe = add_feature_engineering(train_df)
    numeric_cols, categorical_cols = get_feature_lists(train_fe)

    fold_id = bundle["fold_id"]
    oof_pred = np.zeros(len(train_fe), dtype=np.float32)
    for valid_fold in sorted(np.unique(fold_id)):
        train_mask = fold_id != valid_fold
        valid_mask = fold_id == valid_fold
        fold_pred = train_one_fold(
            train_df=train_fe.loc[train_mask].reset_index(drop=True),
            valid_df=train_fe.loc[valid_mask].reset_index(drop=True),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            device=device,
        )
        oof_pred[valid_mask] = fold_pred
        fold_score = rmse(train_fe.loc[valid_mask, "monthly_spend"].to_numpy(dtype=np.float32), fold_pred)
        print(f"embedding_mlp_fe: fold {valid_fold} rmse={fold_score:.6f}")

    overall = append_model(
        bundle=bundle,
        filename_tag="embedding_mlp_fe",
        model_name="Embedding MLP FE",
        category="advanced_more_fe",
        important_params="fe_emb_tabmlp_h256x128x64_do0p10_lr8em04_bs512_ep80_pat12_seed123",
        oof_pred=oof_pred,
    )
    print(f"embedding_mlp_fe: overall rmse={overall:.6f}")


if __name__ == "__main__":
    main()
