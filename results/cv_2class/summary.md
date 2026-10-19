# 5-fold grouped cross-validation (2 classes)

126 clips in 34 groups; every clip is tested once. Class counts: [61, 65] (correct_form, fault). fp32 models.

| Model | Accuracy mean ± SD over folds | Macro F1 mean ± SD | Pooled accuracy | Pooled macro F1 |
|---|---|---|---|---|
| A_ResNet50 | 0.575 ± 0.075 | 0.541 ± 0.132 | 0.571 | 0.570 |
| B_CNN_Transformer | 0.646 ± 0.140 | 0.615 ± 0.165 | 0.643 | 0.640 |
| C_XGBoost | 0.659 ± 0.061 | 0.644 ± 0.070 | 0.659 | 0.656 |
| R_angle_rule | 0.655 ± 0.110 | 0.637 ± 0.112 | 0.659 | 0.649 |

R_angle_rule is a reference only; its thresholds came from the pilot clips.

## McNemar's test on pooled predictions (exact)

| Pair | only 1st correct | only 2nd correct | p |
|---|---|---|---|
| A_ResNet50 vs B_CNN_Transformer | 20 | 29 | 0.253 |
| A_ResNet50 vs C_XGBoost | 16 | 27 | 0.126 |
| B_CNN_Transformer vs C_XGBoost | 23 | 25 | 0.885 |