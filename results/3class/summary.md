# Model comparison (3 classes)

Test set: 17 clips from held-out groups (train 90, val 19). Class counts in test: [9, 5, 3] (correct_form, knee_valgus, trunk_lean).

| Model | Accuracy | Macro F1 | Size (MB) | Latency (ms, CPU 1 thread) |
|---|---|---|---|---|
| A_ResNet50 | 0.647 | 0.627 | 94.0 | 83 |
| B_CNN_Transformer | 0.176 | 0.162 | 96.1 | 659 |
| C_XGBoost | 0.647 | 0.492 | 95.3 | 83 |
| R_angle_rule | 0.588 | 0.351 | 0.0 | 0 |
| A_ResNet50_int8 | 0.588 | 0.575 | 23.7 | 101 |
| B_CNN_Transformer_int8 | 0.294 | 0.288 | 25.8 | 808 |
| C_XGBoost_int8 | 0.647 | 0.492 | 25.0 | 102 |

Targets: accuracy >= 0.9, macro F1 >= 0.85, latency <= 100 ms (in browser), size <= 50 MB.
Latency here is ONNX Runtime on this PC's CPU with 1 thread; the browser figure is measured in the web app.
_int8 rows are the quantised models the web app runs, evaluated on the same test clips.
R_angle_rule is a reference only; its thresholds came from the pilot clips, which are part of this data.

## McNemar's test (exact, test set)

| Pair | only 1st correct | only 2nd correct | p |
|---|---|---|---|
| A_ResNet50 vs B_CNN_Transformer | 8 | 0 | 0.008 |
| A_ResNet50 vs C_XGBoost | 4 | 4 | 1.000 |
| B_CNN_Transformer vs C_XGBoost | 1 | 9 | 0.021 |
| A_ResNet50_int8 vs B_CNN_Transformer_int8 | 6 | 1 | 0.125 |
| A_ResNet50_int8 vs C_XGBoost_int8 | 4 | 5 | 1.000 |
| B_CNN_Transformer_int8 vs C_XGBoost_int8 | 2 | 8 | 0.109 |