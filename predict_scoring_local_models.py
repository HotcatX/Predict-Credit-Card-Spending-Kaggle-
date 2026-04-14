from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split

from train_fe_embedding_mlp import (
    EmbeddingMLP,
    add_feature_engineering as add_fe_embedding,
    build_category_maps,
    encode_categoricals,
    get_device,
    get_feature_lists as get_embedding_feature_lists,
    seed_everything as seed_embedding,
)
from train_missing_blackbox_fe_models import (
    add_feature_engineering as add_fe_blackbox,
    get_feature_lists as get_blackbox_feature_lists,
    make_mlp_pipeline,
    make_stack_pipeline,
)
from train_tabnet_base_and_fe import (
    add_feature_engineering as add_fe_tabnet,
    encode_for_tabnet,
    get_columns as get_tabnet_columns,
    seed_everything as seed_tabnet,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "outputs" / "submissions"


def save_submission(filename: str, customer_id: pd.Series, preds: np.ndarray) -> Path:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / filename
    pd.DataFrame({"customer_id": customer_id, "monthly_spend": preds}).to_csv(path, index=False)
    return path


def predict_mlp_fe(train_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    train_fe = add_fe_blackbox(train_df)
    score_fe = add_fe_blackbox(score_df)
    feature_cols, numeric_cols, categorical_cols = get_blackbox_feature_lists(train_fe)
    model = make_mlp_pipeline(numeric_cols, categorical_cols)
    model.fit(train_fe[feature_cols], train_fe["monthly_spend"])
    preds = model.predict(score_fe[feature_cols]).astype(np.float32)
    lower = float(train_fe["monthly_spend"].min())
    return np.clip(preds, lower, None)


def predict_catboost_fe(train_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    train_fe = add_fe_blackbox(train_df)
    score_fe = add_fe_blackbox(score_df)
    feature_cols, _numeric_cols, categorical_cols = get_blackbox_feature_lists(train_fe)
    X = train_fe[feature_cols].copy()
    X_score = score_fe[feature_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_score[col] = X_score[col].fillna("missing").astype(str)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, train_fe["monthly_spend"], test_size=0.1, random_state=42
    )
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
        X_tr,
        y_tr,
        cat_features=categorical_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=False,
    )
    preds = model.predict(X_score)
    lower = float(train_fe["monthly_spend"].min())
    return np.clip(np.asarray(preds, dtype=np.float32), lower, None)


def predict_stacking_fe(train_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    train_fe = add_fe_blackbox(train_df)
    score_fe = add_fe_blackbox(score_df)
    feature_cols, numeric_cols, categorical_cols = get_blackbox_feature_lists(train_fe)
    model = make_stack_pipeline(numeric_cols, categorical_cols)
    model.fit(train_fe[feature_cols], train_fe["monthly_spend"])
    preds = model.predict(score_fe[feature_cols]).astype(np.float32)
    lower = float(train_fe["monthly_spend"].min())
    return np.clip(preds, lower, None)


def predict_embedding_mlp_fe(train_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    seed_embedding(123)
    device = get_device()
    train_fe = add_fe_embedding(train_df)
    score_fe = add_fe_embedding(score_df)
    numeric_cols, categorical_cols = get_embedding_feature_lists(train_fe)

    X_train_num_raw = train_fe[numeric_cols].copy().fillna(train_fe[numeric_cols].median())
    X_score_num_raw = score_fe[numeric_cols].copy().fillna(train_fe[numeric_cols].median())
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num_raw).astype(np.float32)
    X_score_num = scaler.transform(X_score_num_raw).astype(np.float32)

    cat_maps, cardinalities = build_category_maps(train_fe, categorical_cols)
    X_train_cat = encode_categoricals(train_fe, categorical_cols, cat_maps)
    X_score_cat = encode_categoricals(score_fe, categorical_cols, cat_maps)
    y_train = train_fe["monthly_spend"].to_numpy(dtype=np.float32)

    from train_fe_embedding_mlp import TabularDataset

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = EmbeddingMLP(
        num_numeric=len(numeric_cols),
        cardinalities=cardinalities,
        hidden_dims=[256, 128, 64],
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    patience = 12
    no_improve = 0

    for _ in range(80):
        model.train()
        epoch_losses = []
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb_num, xb_cat)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score_num = torch.tensor(X_score_num, dtype=torch.float32).to(device)
        score_cat = torch.tensor(X_score_cat, dtype=torch.long).to(device)
        preds = model(score_num, score_cat).detach().cpu().numpy().astype(np.float32)
    lower = float(train_fe["monthly_spend"].min())
    return np.clip(preds, lower, None)


def predict_tabnet(train_df: pd.DataFrame, score_df: pd.DataFrame, include_fe: bool) -> np.ndarray:
    seed_tabnet(123)
    train_use = add_fe_tabnet(train_df) if include_fe else train_df.copy()
    score_use = add_fe_tabnet(score_df) if include_fe else score_df.copy()
    numeric_cols, categorical_cols = get_tabnet_columns(train_use, include_fe=include_fe)
    train_part, valid_part = train_test_split(train_use, test_size=0.1, random_state=42)
    X_tr, X_val, cat_idxs, cat_dims, _ = encode_for_tabnet(train_part, valid_part, numeric_cols, categorical_cols)
    X_full, X_score, _cat_idxs2, _cat_dims2, _ = encode_for_tabnet(train_use, score_use, numeric_cols, categorical_cols)
    y_tr = train_part["monthly_spend"].to_numpy(dtype=np.float32)
    y_val = valid_part["monthly_spend"].to_numpy(dtype=np.float32)

    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)

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
        seed=123,
        verbose=0,
    )
    model.fit(
        X_train=X_tr,
        y_train=y_tr_scaled,
        eval_set=[(X_val, y_val_scaled)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=20,
        patience=5,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )
    preds_scaled = model.predict(X_score).reshape(-1, 1).astype(np.float32)
    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1).astype(np.float32)
    lower = float(train_use["monthly_spend"].min())
    return np.clip(preds.astype(np.float32), lower, None)


def main() -> None:
    train_df = pd.read_csv(ROOT / "analysis_data.csv")
    score_df = pd.read_csv(ROOT / "scoring_data.csv")

    tasks = [
        (
            "12_mlp_256_128_64_fe__fe_h256x128x64_relu_bs256_lr0p0005_alpha1e_4_iter360_seed123__scoring.csv",
            "MLPRegressor FE",
            predict_mlp_fe,
        ),
        (
            "13_catboost_depth8_fe__fe_depth8_lr0p03_iter4000_seed42__scoring.csv",
            "CatBoostRegressor FE",
            predict_catboost_fe,
        ),
        (
            "14_stack_ridge_extra_hist_fe__fe_meta_ridge0p5_base_ridge_extra250_hist700__scoring.csv",
            "StackingRegressor FE",
            predict_stacking_fe,
        ),
        (
            "15_embedding_mlp_fe__fe_emb_tabmlp_h256x128x64_do0p10_lr8em04_bs512_ep80_pat12_seed123__scoring.csv",
            "Embedding MLP FE",
            predict_embedding_mlp_fe,
        ),
    ]

    for filename, label, fn in tasks:
        print(f"Running {label}...")
        preds = fn(train_df, score_df)
        path = save_submission(filename, score_df["customer_id"], preds)
        print(f"Saved {path.relative_to(ROOT)}")

    print("Running TabNet...")
    preds = predict_tabnet(train_df, score_df, include_fe=False)
    path = save_submission(
        "17_tabnet_base__nd8_na8_steps3_gamma1p1_lr1em02_ep20_pat5_seed123__scoring.csv",
        score_df["customer_id"],
        preds,
    )
    print(f"Saved {path.relative_to(ROOT)}")

    print("Running TabNet FE...")
    preds = predict_tabnet(train_df, score_df, include_fe=True)
    path = save_submission(
        "16_tabnet_fe__fe_nd8_na8_steps3_gamma1p1_lr1em02_ep20_pat5_seed123__scoring.csv",
        score_df["customer_id"],
        preds,
    )
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
