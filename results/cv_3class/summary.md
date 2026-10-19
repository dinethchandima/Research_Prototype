# 5-fold grouped cross-validation (3 classes)

126 clips in 34 groups; every clip is tested once. Class counts: [61, 37, 28] (correct_form, knee_valgus, trunk_lean). fp32 models.

| Model | Accuracy mean ± SD over folds | Macro F1 mean ± SD | Pooled accuracy | Pooled macro F1 |
|---|---|---|---|---|
| A_ResNet50 | 0.439 ± 0.054 | 0.355 ± 0.069 | 0.437 | 0.389 |
| B_CNN_Transformer | 0.384 ± 0.140 | 0.299 ± 0.104 | 0.389 | 0.333 |
| C_XGBoost | 0.596 ± 0.111 | 0.529 ± 0.126 | 0.603 | 0.545 |
| R_angle_rule | 0.604 ± 0.144 | 0.493 ± 0.139 | 0.611 | 0.502 |

R_angle_rule is a reference only; its thresholds came from the pilot clips.

## McNemar's test on pooled predictions (exact)

| Pair | only 1st correct | only 2nd correct | p |
|---|---|---|---|
| A_ResNet50 vs B_CNN_Transformer | 21 | 15 | 0.405 |
| A_ResNet50 vs C_XGBoost | 15 | 36 | 0.005 |
| B_CNN_Transformer vs C_XGBoost | 15 | 42 | 0.000 |