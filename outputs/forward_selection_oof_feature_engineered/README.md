# Forward Selection OOF Bundle (Feature Engineered)

- CV scheme: `KFold(n_splits=5, shuffle=True, random_state=42)`
- This bundle excludes `MLPRegressor`, `CatBoostRegressor`, and `StackingRegressor`.
- It uses a shared feature-engineering layer with ratios, interactions, and category crosses.
- `registry.csv`: one row per canonical FE model
- `oof_matrix_cv5_rs42.csv`: aligned OOF matrix for direct forward selection
- `models/`: per-model OOF files with normalized names

## Lightweight Storage

- Shared alignment arrays are stored once in `shared/index_arrays.npz`.
- Each model OOF prediction is stored separately as a compressed `.npz` shard in `shards/`.
- Use `assemble_oof_bundle.py` to rebuild an aligned OOF matrix on demand.