# Model comparison (2 classes)

Test set: 17 clips from held-out groups (train 90, val 19). Class counts in test: [9, 8] (correct_form, fault).

| Model | Accuracy | Macro F1 | Size (MB) | Latency (ms, CPU 1 thread) |
|---|---|---|---|---|
| A_ResNet50 | 0.588 | 0.582 | 94.0 | 83 |
| B_CNN_Transformer | 0.529 | 0.485 | 96.1 | 661 |
| C_XGBoost | 0.647 | 0.614 | 94.9 | 84 |
| R_angle_rule | 0.588 | 0.471 | 0.0 | 0 |
| A_ResNet50_int8 | 0.588 | 0.582 | 23.7 | 101 |
| B_CNN_Transformer_int8 | 0.471 | 0.395 | 25.8 | 813 |
| C_XGBoost_int8 | 0.706 | 0.664 | 24.7 | 103 |

Targets: accuracy >= 0.9, macro F1 >= 0.85, latency <= 100 ms (in browser), size <= 50 MB.
Latency here is ONNX Runtime on this PC's CPU with 1 thread; the browser figure is measured in the web app.
_int8 rows are the quantised models the web app runs, evaluated on the same test clips.
R_angle_rule is a reference only; its thresholds came from the pilot clips, which are part of this data.

## McNemar's test (exact, test set)

| Pair | only 1st correct | only 2nd correct | p |
|---|---|---|---|
| A_ResNet50 vs B_CNN_Transformer | 3 | 2 | 1.000 |
| A_ResNet50 vs C_XGBoost | 4 | 5 | 1.000 |
| B_CNN_Transformer vs C_XGBoost | 5 | 7 | 0.774 |
| A_ResNet50_int8 vs B_CNN_Transformer_int8 | 4 | 2 | 0.688 |
| A_ResNet50_int8 vs C_XGBoost_int8 | 4 | 6 | 0.754 |
| B_CNN_Transformer_int8 vs C_XGBoost_int8 | 5 | 9 | 0.424 |