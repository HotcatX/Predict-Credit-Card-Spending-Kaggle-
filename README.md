# Predict Credit Card Spending Kaggle

This repository currently contains a cleaned OOF bundle for preliminary forward selection.

Included:
- `outputs/forward_selection_oof/registry.csv`
- `outputs/forward_selection_oof/oof_matrix_cv5_rs42.csv`
- `outputs/forward_selection_oof/models/*.csv`

Notes:
- The OOF bundle was rebuilt with a unified `5-Fold KFold(random_state=42)` scheme.
- Only the canonical 16 models were kept in the exported bundle.
- Old blend experiments and stale intermediate report artifacts were removed before packaging.
