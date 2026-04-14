# Predict Credit Card Spending Kaggle

This repository currently contains a cleaned OOF bundle for preliminary forward selection.

Included:
- `outputs/forward_selection_oof/registry_shards.csv`
- `outputs/forward_selection_oof/shared/index_arrays.npz`
- `outputs/forward_selection_oof/shards/*.npz`
- `outputs/forward_selection_oof_feature_engineered/registry_shards.csv`
- `outputs/forward_selection_oof_feature_engineered/shared/index_arrays.npz`
- `outputs/forward_selection_oof_feature_engineered/shards/*.npz`
- `assemble_oof_bundle.py`
- `add_model_to_bundle_template.py`

Notes:
- The OOF bundle was rebuilt with a unified `5-Fold KFold(random_state=42)` scheme.
- Only the canonical 16 models were kept in the exported bundle.
- Old blend experiments and stale intermediate report artifacts were removed before packaging.
- A second bundle contains a feature-engineered version that excludes `MLP`, `CatBoost`, and `StackingRegressor`.
- OOF storage is now lightweight: one shared alignment file plus one compressed NumPy shard per model.
- Use `assemble_oof_bundle.py` to rebuild an aligned matrix dynamically for forward selection or hot updates.
- Use `add_model_to_bundle_template.py` as the canonical entry point for adding one new aligned model shard into an existing bundle.
