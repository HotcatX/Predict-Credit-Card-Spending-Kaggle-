# Forward Selection OOF Bundle

- CV scheme: `KFold(n_splits=5, shuffle=True, random_state=42)`
- `registry.csv`: one row per canonical model with normalized file name and OOF RMSE
- `oof_matrix_cv3_rs42.csv`: aligned OOF matrix for direct forward selection / blending
- `models/`: per-model OOF files with columns `customer_id`, `target_monthly_spend`, `fold_id`, `oof_prediction`