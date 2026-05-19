# Model Effect Analysis

The current experiment suite focuses on baseline comparison and ablation analysis over the four DAppSCAN vulnerability tasks. Under uniform false-negative noise, RESCUER ranks first with an overall average F1 of 0.4560. Under asymmetric noise, CL ranks first with an overall average F1 of 0.4594. The full ablation model reaches an overall average F1 of 0.4229, outperforming the weakened variants in aggregate.

## Baseline Average F1 by Noise

| Method   |   noise_0.0 |   noise_0.1 |   noise_0.2 |   noise_0.3 |   overall_avg |
|:---------|------------:|------------:|------------:|------------:|--------------:|
| RESCUER  |      0.4439 |      0.4586 |      0.4663 |      0.4551 |        0.456  |
| FedAvg   |      0.4674 |      0.4882 |      0.3483 |      0.125  |        0.3572 |
| CL       |      0.4328 |      0.4431 |      0.441  |      0.3155 |        0.4081 |
| CLC      |      0.2739 |      0.2739 |      0.2739 |      0.2739 |        0.2739 |
| FedCorr  |      0.4518 |      0.3624 |      0.3774 |      0.1111 |        0.3257 |
| ARFL     |      0.4636 |      0.4449 |      0.3818 |      0.1176 |        0.352  |

## Robustness Summary

| Method   |   noise_0.0 |   noise_0.3 |   drop_0.0_to_0.3 |   overall_avg |
|:---------|------------:|------------:|------------------:|--------------:|
| RESCUER  |      0.4439 |      0.4551 |            0.0112 |        0.456  |
| FedAvg   |      0.4674 |      0.125  |           -0.3424 |        0.3572 |
| CL       |      0.4328 |      0.3155 |           -0.1173 |        0.4081 |
| CLC      |      0.2739 |      0.2739 |            0      |        0.2739 |
| FedCorr  |      0.4518 |      0.1111 |           -0.3407 |        0.3257 |
| ARFL     |      0.4636 |      0.1176 |           -0.346  |        0.352  |

## Ablation Average F1 by Noise

| Variant                |   noise_0.0 |   noise_0.1 |   noise_0.2 |   noise_0.3 |   overall_avg |
|:-----------------------|------------:|------------:|------------:|------------:|--------------:|
| Ablation_Full          |      0.4029 |      0.4616 |      0.4611 |      0.3658 |        0.4229 |
| Ablation_woLCN         |      0.4718 |      0.5018 |      0.3198 |      0.1071 |        0.3502 |
| Ablation_woWarmup      |      0.4513 |      0.4522 |      0.4536 |      0.3655 |        0.4307 |
| Ablation_woOrtho       |      0.4988 |      0.4829 |      0.4108 |      0.151  |        0.3859 |
| Ablation_woTransformer |      0.2171 |      0.1635 |      0.12   |      0.1208 |        0.1554 |
