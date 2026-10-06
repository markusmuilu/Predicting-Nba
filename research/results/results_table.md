
### Validation: 2024-25, 1209 games (home win rate 0.548)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.679 | 0.2104 | 0.6077 | 0.035 |
| Player model, actual roster (optimistic) | 0.690 | 0.2022 | 0.5892 | 0.033 |
| Ablation: logreg on player features, actual roster | 0.698 | 0.2013 | 0.5874 | 0.024 |
| Player model, previous-game roster | 0.684 | 0.2077 | 0.6024 | 0.029 |
| Ablation: logreg on player features, previous-game roster | 0.689 | 0.2073 | 0.6019 | 0.041 |
| Jev, real names, previous-game roster | 0.647 | 0.2288 | 0.6550 | 0.109 |
| Jev, anonymised, previous-game roster | 0.637 | 0.2355 | 0.6744 | 0.128 |
| Jev, real names, recalibrated on validation | 0.663 | 0.2110 | 0.6095 | 0.045 |
| Jev, anonymised, recalibrated on validation | 0.661 | 0.2117 | 0.6110 | 0.041 |
| Constant: home win rate | 0.548 | 0.2477 | 0.6886 | 0.000 |

### Test: 2025-26, 1209 games (home win rate 0.553)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.667 | 0.2093 | 0.6057 | 0.025 |
| Player model, actual roster (optimistic) | 0.693 | 0.1976 | 0.5782 | 0.029 |
| Ablation: logreg on player features, actual roster | 0.693 | 0.1988 | 0.5808 | 0.021 |
| Player model, previous-game roster | 0.676 | 0.2027 | 0.5895 | 0.026 |
| Ablation: logreg on player features, previous-game roster | 0.678 | 0.2045 | 0.5936 | 0.027 |
| Jev, real names, previous-game roster | 0.663 | 0.2217 | 0.6416 | 0.114 |
| Jev, anonymised, previous-game roster | 0.653 | 0.2277 | 0.6583 | 0.127 |
| Jev, real names, recalibrated on validation | 0.682 | 0.2048 | 0.5964 | 0.024 |
| Jev, anonymised, recalibrated on validation | 0.674 | 0.2058 | 0.5985 | 0.023 |
| Constant: home win rate | 0.553 | 0.2472 | 0.6874 | 0.000 |

**Games with recorded Pinnacle odds:** 680 (2026-01-07 to 2026-04-12)

| Model | Accuracy | Brier | Log loss | Flat ROI, back pick | +EV bets | Flat ROI, +EV | 1/4 Kelly final | 1/4 Kelly max DD | 1/2 Kelly final |
|---|---|---|---|---|---|---|---|---|---|
| Market (de-vigged) | 0.710 | 0.1829 | 0.5432 | | | | | | |
| Logistic regression (production features) | 0.696 | 0.1980 | 0.5805 | -0.002 | 595 | -0.122 | 0.037 | 0.979 | 0.000 |
| Player model, actual roster (optimistic) | 0.722 | 0.1825 | 0.5427 | +0.032 | 548 | -0.018 | 6.786 | 0.587 | 10.676 |
| Ablation: logreg on player features, actual roster | 0.724 | 0.1869 | 0.5532 | +0.045 | 586 | -0.051 | 1.648 | 0.664 | 0.495 |
| Player model, previous-game roster | 0.701 | 0.1888 | 0.5573 | -0.009 | 565 | -0.050 | 0.566 | 0.850 | 0.080 |
| Ablation: logreg on player features, previous-game roster | 0.699 | 0.1926 | 0.5663 | -0.004 | 592 | -0.115 | 0.242 | 0.918 | 0.010 |
| Jev, real names, previous-game roster | 0.679 | 0.2097 | 0.6096 | -0.017 | 615 | -0.108 | 0.020 | 0.985 | 0.000 |
| Jev, anonymised, previous-game roster | 0.665 | 0.2155 | 0.6244 | -0.035 | 614 | -0.101 | 0.008 | 0.994 | 0.000 |
| Jev, real names, recalibrated on validation | 0.699 | 0.1960 | 0.5772 | -0.001 | 620 | -0.148 | 0.040 | 0.974 | 0.000 |
| Jev, anonymised, recalibrated on validation | 0.691 | 0.1978 | 0.5813 | -0.014 | 615 | -0.178 | 0.020 | 0.988 | 0.000 |

**Validation: log loss improvement over production logreg, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Player model, actual roster (optimistic) | +0.0185 | [+0.0024, +0.0337] |
| Ablation: logreg on player features, actual roster | +0.0203 | [+0.0047, +0.0349] |
| Player model, previous-game roster | +0.0053 | [-0.0072, +0.0180] |
| Ablation: logreg on player features, previous-game roster | +0.0058 | [-0.0071, +0.0180] |
| Jev, real names, previous-game roster | -0.0473 | [-0.0710, -0.0248] |
| Jev, anonymised, previous-game roster | -0.0666 | [-0.0942, -0.0413] |
| Jev, real names, recalibrated on validation | -0.0018 | [-0.0119, +0.0083] |
| Jev, anonymised, recalibrated on validation | -0.0033 | [-0.0134, +0.0070] |

**Test: log loss improvement over production logreg, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Player model, actual roster (optimistic) | +0.0276 | [+0.0121, +0.0428] |
| Ablation: logreg on player features, actual roster | +0.0249 | [+0.0102, +0.0402] |
| Player model, previous-game roster | +0.0163 | [+0.0042, +0.0289] |
| Ablation: logreg on player features, previous-game roster | +0.0122 | [-0.0005, +0.0253] |
| Jev, real names, previous-game roster | -0.0359 | [-0.0598, -0.0122] |
| Jev, anonymised, previous-game roster | -0.0526 | [-0.0801, -0.0255] |
| Jev, real names, recalibrated on validation | +0.0093 | [-0.0016, +0.0196] |
| Jev, anonymised, recalibrated on validation | +0.0072 | [-0.0034, +0.0174] |

**Validation: Jev with real names vs anonymised, log loss gain from names, paired bootstrap 95% CI**

| Version | Mean | 95% CI |
|---|---|---|
| raw | +0.0193 | [+0.0147, +0.0245] |
| recalibrated | +0.0015 | [-0.0008, +0.0037] |

**Test: Jev with real names vs anonymised, log loss gain from names, paired bootstrap 95% CI**

| Version | Mean | 95% CI |
|---|---|---|
| raw | +0.0167 | [+0.0122, +0.0215] |
| recalibrated | +0.0021 | [-0.0002, +0.0044] |
