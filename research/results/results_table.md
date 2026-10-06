
### Validation: 2024-25, 1209 games (home win rate 0.548)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.679 | 0.2104 | 0.6077 | 0.035 |
| Player model, actual roster (optimistic) | 0.690 | 0.2022 | 0.5892 | 0.033 |
| Ablation: logreg on player features, actual roster | 0.698 | 0.2013 | 0.5874 | 0.024 |
| Player model, previous-game roster | 0.684 | 0.2077 | 0.6024 | 0.029 |
| Ablation: logreg on player features, previous-game roster | 0.689 | 0.2073 | 0.6019 | 0.041 |
| Constant: home win rate | 0.548 | 0.2477 | 0.6886 | 0.000 |

### Test: 2025-26, 1209 games (home win rate 0.553)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.667 | 0.2093 | 0.6057 | 0.025 |
| Player model, actual roster (optimistic) | 0.693 | 0.1976 | 0.5782 | 0.029 |
| Ablation: logreg on player features, actual roster | 0.693 | 0.1988 | 0.5808 | 0.021 |
| Player model, previous-game roster | 0.676 | 0.2027 | 0.5895 | 0.026 |
| Ablation: logreg on player features, previous-game roster | 0.678 | 0.2045 | 0.5936 | 0.027 |
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

**Validation: log loss improvement over production logreg, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Player model, actual roster (optimistic) | +0.0185 | [+0.0024, +0.0337] |
| Ablation: logreg on player features, actual roster | +0.0203 | [+0.0047, +0.0349] |
| Player model, previous-game roster | +0.0053 | [-0.0072, +0.0180] |
| Ablation: logreg on player features, previous-game roster | +0.0058 | [-0.0071, +0.0180] |

**Test: log loss improvement over production logreg, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Player model, actual roster (optimistic) | +0.0276 | [+0.0121, +0.0428] |
| Ablation: logreg on player features, actual roster | +0.0249 | [+0.0102, +0.0402] |
| Player model, previous-game roster | +0.0163 | [+0.0042, +0.0289] |
| Ablation: logreg on player features, previous-game roster | +0.0122 | [-0.0005, +0.0253] |
