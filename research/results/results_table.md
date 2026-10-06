
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
| Team Elo (settings tuned on validation) | 0.653 | 0.2098 | 0.6065 | 0.029 |
| Neural Elo: K, home bonus, carryover learned by gradient | 0.667 | 0.2098 | 0.6063 | 0.034 |
| Neural Elo with a learned update network | 0.670 | 0.2099 | 0.6065 | 0.027 |
| Player Elo, previous-game roster | 0.678 | 0.2087 | 0.6039 | 0.029 |
| Ridge plus-minus, previous-game roster | 0.680 | 0.2072 | 0.6003 | 0.013 |
| Player Elo, actual roster (optimistic) | 0.672 | 0.2056 | 0.5970 | 0.020 |
| Ridge plus-minus, actual roster (optimistic) | 0.681 | 0.2041 | 0.5933 | 0.021 |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | 0.687 | 0.2054 | 0.5962 | 0.028 |
| Production features + team Elo | 0.674 | 0.2086 | 0.6035 | 0.024 |
| Production features + player Elo and ridge | 0.681 | 0.2061 | 0.5978 | 0.020 |
| Production features + all rating scalars | 0.683 | 0.2060 | 0.5978 | 0.026 |
| Production features + rating scalars, actual roster (optimistic) | 0.686 | 0.2034 | 0.5914 | 0.019 |
| Deep Sets, hand features + raw sequences, 12 seasons | 0.686 | 0.2054 | 0.5977 | 0.022 |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | 0.701 | 0.2013 | 0.5878 | 0.031 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.698 | 0.2021 | 0.5892 | 0.025 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | 0.685 | 0.2023 | 0.5891 | 0.022 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | 0.687 | 0.2029 | 0.5910 | 0.018 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | 0.693 | 0.1987 | 0.5813 | 0.017 |
| Deep Sets, hand features + raw sequences, 4 seasons | 0.680 | 0.2084 | 0.6040 | 0.034 |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | 0.682 | 0.2047 | 0.5955 | 0.036 |
| Deep Sets, hand-built features, 12 seasons | 0.687 | 0.2054 | 0.5980 | 0.032 |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | 0.684 | 0.2023 | 0.5901 | 0.028 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | 0.690 | 0.2027 | 0.5909 | 0.033 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.689 | 0.2033 | 0.5921 | 0.036 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | 0.683 | 0.2037 | 0.5934 | 0.025 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | 0.694 | 0.1996 | 0.5825 | 0.026 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | 0.682 | 0.2035 | 0.5924 | 0.018 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | 0.693 | 0.1996 | 0.5824 | 0.027 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | 0.682 | 0.2038 | 0.5931 | 0.019 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | 0.689 | 0.2005 | 0.5849 | 0.022 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | 0.678 | 0.2039 | 0.5938 | 0.021 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | 0.688 | 0.1994 | 0.5821 | 0.036 |
| Deep Sets, hand-built features, 4 seasons | 0.674 | 0.2074 | 0.6018 | 0.028 |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | 0.685 | 0.2050 | 0.5960 | 0.039 |
| Deep Sets, hand-built features, 8 seasons | 0.684 | 0.2063 | 0.5999 | 0.029 |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | 0.685 | 0.2028 | 0.5912 | 0.031 |
| Deep Sets, raw sequences (GRU), 12 seasons | 0.687 | 0.2054 | 0.5971 | 0.029 |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | 0.693 | 0.2017 | 0.5878 | 0.035 |
| Deep Sets, raw sequences (GRU), 4 seasons | 0.683 | 0.2065 | 0.5996 | 0.036 |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | 0.687 | 0.2028 | 0.5910 | 0.044 |
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
| Team Elo (settings tuned on validation) | 0.691 | 0.2030 | 0.5925 | 0.033 |
| Neural Elo: K, home bonus, carryover learned by gradient | 0.687 | 0.2028 | 0.5918 | 0.032 |
| Neural Elo with a learned update network | 0.690 | 0.2032 | 0.5926 | 0.037 |
| Player Elo, previous-game roster | 0.687 | 0.2035 | 0.5930 | 0.027 |
| Ridge plus-minus, previous-game roster | 0.700 | 0.2020 | 0.5897 | 0.039 |
| Player Elo, actual roster (optimistic) | 0.695 | 0.2000 | 0.5855 | 0.036 |
| Ridge plus-minus, actual roster (optimistic) | 0.698 | 0.1987 | 0.5825 | 0.024 |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | 0.687 | 0.2024 | 0.5897 | 0.028 |
| Production features + team Elo | 0.682 | 0.2067 | 0.5999 | 0.022 |
| Production features + player Elo and ridge | 0.692 | 0.2050 | 0.5960 | 0.033 |
| Production features + all rating scalars | 0.691 | 0.2044 | 0.5946 | 0.034 |
| Production features + rating scalars, actual roster (optimistic) | 0.691 | 0.2026 | 0.5905 | 0.024 |
| Deep Sets, hand features + raw sequences, 12 seasons | 0.682 | 0.2035 | 0.5910 | 0.022 |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | 0.691 | 0.2000 | 0.5831 | 0.019 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.692 | 0.1995 | 0.5821 | 0.016 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | 0.695 | 0.2008 | 0.5853 | 0.028 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | 0.688 | 0.1999 | 0.5829 | 0.022 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | 0.705 | 0.1964 | 0.5751 | 0.026 |
| Deep Sets, hand features + raw sequences, 4 seasons | 0.677 | 0.2061 | 0.5973 | 0.034 |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | 0.682 | 0.2022 | 0.5885 | 0.040 |
| Deep Sets, hand-built features, 12 seasons | 0.672 | 0.2059 | 0.5963 | 0.033 |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | 0.686 | 0.2002 | 0.5833 | 0.022 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | 0.691 | 0.2017 | 0.5872 | 0.027 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.685 | 0.2013 | 0.5865 | 0.017 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | 0.691 | 0.2016 | 0.5868 | 0.020 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | 0.691 | 0.1966 | 0.5754 | 0.028 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | 0.675 | 0.2016 | 0.5866 | 0.023 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | 0.694 | 0.1966 | 0.5752 | 0.027 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | 0.684 | 0.2025 | 0.5890 | 0.034 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | 0.691 | 0.1974 | 0.5773 | 0.024 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | 0.682 | 0.2020 | 0.5877 | 0.025 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | 0.694 | 0.1972 | 0.5769 | 0.021 |
| Deep Sets, hand-built features, 4 seasons | 0.674 | 0.2081 | 0.6019 | 0.046 |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | 0.677 | 0.2024 | 0.5887 | 0.033 |
| Deep Sets, hand-built features, 8 seasons | 0.677 | 0.2063 | 0.5976 | 0.036 |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | 0.687 | 0.2010 | 0.5852 | 0.021 |
| Deep Sets, raw sequences (GRU), 12 seasons | 0.685 | 0.2035 | 0.5913 | 0.027 |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | 0.696 | 0.1998 | 0.5829 | 0.023 |
| Deep Sets, raw sequences (GRU), 4 seasons | 0.680 | 0.2050 | 0.5955 | 0.037 |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | 0.677 | 0.2021 | 0.5885 | 0.028 |
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
| Team Elo (settings tuned on validation) | 0.718 | 0.1926 | 0.5693 | +0.034 | 598 | -0.174 | 0.118 | 0.956 | 0.003 |
| Neural Elo: K, home bonus, carryover learned by gradient | 0.715 | 0.1919 | 0.5672 | +0.028 | 583 | -0.152 | 0.166 | 0.933 | 0.007 |
| Neural Elo with a learned update network | 0.715 | 0.1922 | 0.5682 | +0.028 | 599 | -0.173 | 0.131 | 0.947 | 0.004 |
| Player Elo, previous-game roster | 0.706 | 0.1930 | 0.5691 | +0.005 | 579 | -0.123 | 0.080 | 0.963 | 0.002 |
| Ridge plus-minus, previous-game roster | 0.722 | 0.1924 | 0.5679 | +0.033 | 581 | -0.169 | 0.127 | 0.969 | 0.004 |
| Player Elo, actual roster (optimistic) | 0.715 | 0.1887 | 0.5600 | +0.013 | 568 | -0.173 | 0.276 | 0.908 | 0.029 |
| Ridge plus-minus, actual roster (optimistic) | 0.724 | 0.1878 | 0.5574 | +0.028 | 570 | -0.172 | 0.540 | 0.912 | 0.099 |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | 0.710 | 0.1922 | 0.5664 | +0.005 | 555 | -0.153 | 0.120 | 0.947 | 0.005 |
| Production features + team Elo | 0.709 | 0.1970 | 0.5786 | +0.016 | 599 | -0.149 | 0.040 | 0.979 | 0.000 |
| Production features + player Elo and ridge | 0.710 | 0.1950 | 0.5727 | +0.010 | 571 | -0.151 | 0.063 | 0.965 | 0.001 |
| Production features + all rating scalars | 0.712 | 0.1948 | 0.5727 | +0.015 | 571 | -0.165 | 0.051 | 0.970 | 0.001 |
| Production features + rating scalars, actual roster (optimistic) | 0.715 | 0.1915 | 0.5650 | +0.002 | 575 | -0.113 | 0.168 | 0.930 | 0.008 |
| Deep Sets, hand features + raw sequences, 12 seasons | 0.703 | 0.1913 | 0.5634 | +0.010 | 591 | -0.058 | 0.261 | 0.899 | 0.014 |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | 0.712 | 0.1873 | 0.5531 | +0.012 | 576 | -0.033 | 0.916 | 0.806 | 0.225 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.713 | 0.1879 | 0.5549 | +0.011 | 548 | -0.058 | 0.484 | 0.866 | 0.079 |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | 0.728 | 0.1887 | 0.5570 | +0.045 | 547 | -0.054 | 0.527 | 0.888 | 0.073 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | 0.703 | 0.1878 | 0.5548 | -0.006 | 530 | -0.089 | 0.584 | 0.865 | 0.108 |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | 0.719 | 0.1832 | 0.5435 | +0.014 | 530 | -0.020 | 2.136 | 0.681 | 1.971 |
| Deep Sets, hand features + raw sequences, 4 seasons | 0.701 | 0.1936 | 0.5694 | +0.014 | 613 | -0.106 | 0.185 | 0.954 | 0.005 |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | 0.712 | 0.1881 | 0.5564 | +0.016 | 581 | -0.085 | 1.008 | 0.875 | 0.213 |
| Deep Sets, hand-built features, 12 seasons | 0.691 | 0.1949 | 0.5713 | -0.014 | 596 | -0.111 | 0.091 | 0.957 | 0.002 |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | 0.706 | 0.1878 | 0.5542 | +0.005 | 579 | -0.038 | 0.741 | 0.780 | 0.161 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | 0.718 | 0.1900 | 0.5600 | +0.022 | 560 | -0.067 | 0.214 | 0.921 | 0.015 |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | 0.703 | 0.1894 | 0.5587 | -0.010 | 566 | -0.095 | 0.272 | 0.902 | 0.025 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | 0.715 | 0.1896 | 0.5590 | +0.020 | 558 | -0.075 | 0.268 | 0.899 | 0.025 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | 0.718 | 0.1833 | 0.5441 | +0.010 | 527 | -0.031 | 1.839 | 0.568 | 1.641 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | 0.701 | 0.1887 | 0.5569 | -0.014 | 564 | -0.075 | 0.362 | 0.868 | 0.049 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | 0.724 | 0.1830 | 0.5432 | +0.021 | 532 | -0.037 | 2.382 | 0.603 | 2.626 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | 0.704 | 0.1901 | 0.5603 | +0.002 | 570 | -0.111 | 0.259 | 0.903 | 0.020 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | 0.718 | 0.1839 | 0.5452 | +0.012 | 550 | -0.015 | 1.869 | 0.633 | 1.407 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | 0.712 | 0.1890 | 0.5577 | +0.009 | 578 | -0.082 | 0.333 | 0.887 | 0.038 |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | 0.725 | 0.1825 | 0.5422 | +0.026 | 537 | +0.014 | 3.024 | 0.618 | 3.815 |
| Deep Sets, hand-built features, 4 seasons | 0.700 | 0.1963 | 0.5754 | +0.008 | 613 | -0.133 | 0.080 | 0.968 | 0.001 |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | 0.704 | 0.1893 | 0.5587 | -0.001 | 581 | -0.091 | 0.566 | 0.879 | 0.075 |
| Deep Sets, hand-built features, 8 seasons | 0.701 | 0.1954 | 0.5730 | +0.006 | 605 | -0.101 | 0.066 | 0.972 | 0.001 |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | 0.709 | 0.1883 | 0.5555 | +0.008 | 569 | -0.070 | 0.587 | 0.836 | 0.096 |
| Deep Sets, raw sequences (GRU), 12 seasons | 0.709 | 0.1919 | 0.5648 | +0.024 | 584 | -0.082 | 0.194 | 0.922 | 0.009 |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | 0.716 | 0.1869 | 0.5523 | +0.024 | 561 | -0.002 | 0.763 | 0.820 | 0.178 |
| Deep Sets, raw sequences (GRU), 4 seasons | 0.703 | 0.1908 | 0.5628 | +0.015 | 594 | -0.133 | 0.339 | 0.909 | 0.023 |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | 0.706 | 0.1873 | 0.5540 | +0.002 | 572 | -0.072 | 0.896 | 0.829 | 0.215 |
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
| Team Elo (settings tuned on validation) | +0.0012 | [-0.0081, +0.0111] |
| Neural Elo: K, home bonus, carryover learned by gradient | +0.0014 | [-0.0091, +0.0118] |
| Neural Elo with a learned update network | +0.0012 | [-0.0087, +0.0113] |
| Player Elo, previous-game roster | +0.0038 | [-0.0067, +0.0146] |
| Ridge plus-minus, previous-game roster | +0.0074 | [-0.0050, +0.0196] |
| Player Elo, actual roster (optimistic) | +0.0108 | [-0.0003, +0.0223] |
| Ridge plus-minus, actual roster (optimistic) | +0.0144 | [+0.0016, +0.0277] |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | +0.0115 | [+0.0008, +0.0216] |
| Production features + team Elo | +0.0042 | [-0.0012, +0.0095] |
| Production features + player Elo and ridge | +0.0099 | [+0.0012, +0.0180] |
| Production features + all rating scalars | +0.0099 | [+0.0017, +0.0172] |
| Production features + rating scalars, actual roster (optimistic) | +0.0163 | [+0.0057, +0.0265] |
| Deep Sets, hand features + raw sequences, 12 seasons | +0.0100 | [-0.0032, +0.0236] |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | +0.0199 | [+0.0048, +0.0357] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0185 | [+0.0058, +0.0311] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | +0.0186 | [+0.0052, +0.0324] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | +0.0167 | [+0.0042, +0.0298] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0264 | [+0.0117, +0.0415] |
| Deep Sets, hand features + raw sequences, 4 seasons | +0.0037 | [-0.0104, +0.0174] |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | +0.0122 | [-0.0037, +0.0278] |
| Deep Sets, hand-built features, 12 seasons | +0.0097 | [-0.0032, +0.0223] |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | +0.0176 | [+0.0033, +0.0321] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | +0.0168 | [+0.0043, +0.0292] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0156 | [+0.0031, +0.0275] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | +0.0143 | [+0.0025, +0.0265] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0252 | [+0.0117, +0.0391] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | +0.0153 | [+0.0031, +0.0274] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | +0.0253 | [+0.0116, +0.0398] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | +0.0146 | [+0.0029, +0.0265] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | +0.0228 | [+0.0094, +0.0372] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | +0.0139 | [+0.0016, +0.0266] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | +0.0256 | [+0.0111, +0.0404] |
| Deep Sets, hand-built features, 4 seasons | +0.0059 | [-0.0076, +0.0192] |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | +0.0117 | [-0.0038, +0.0271] |
| Deep Sets, hand-built features, 8 seasons | +0.0078 | [-0.0048, +0.0200] |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | +0.0165 | [+0.0020, +0.0310] |
| Deep Sets, raw sequences (GRU), 12 seasons | +0.0106 | [-0.0028, +0.0237] |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | +0.0199 | [+0.0046, +0.0351] |
| Deep Sets, raw sequences (GRU), 4 seasons | +0.0081 | [-0.0062, +0.0222] |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | +0.0167 | [+0.0007, +0.0324] |
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
| Team Elo (settings tuned on validation) | +0.0132 | [+0.0031, +0.0238] |
| Neural Elo: K, home bonus, carryover learned by gradient | +0.0140 | [+0.0030, +0.0250] |
| Neural Elo with a learned update network | +0.0131 | [+0.0025, +0.0239] |
| Player Elo, previous-game roster | +0.0127 | [+0.0013, +0.0236] |
| Ridge plus-minus, previous-game roster | +0.0160 | [+0.0027, +0.0291] |
| Player Elo, actual roster (optimistic) | +0.0203 | [+0.0089, +0.0316] |
| Ridge plus-minus, actual roster (optimistic) | +0.0233 | [+0.0091, +0.0370] |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | +0.0160 | [+0.0046, +0.0266] |
| Production features + team Elo | +0.0058 | [-0.0002, +0.0116] |
| Production features + player Elo and ridge | +0.0097 | [+0.0004, +0.0190] |
| Production features + all rating scalars | +0.0111 | [+0.0024, +0.0195] |
| Production features + rating scalars, actual roster (optimistic) | +0.0152 | [+0.0036, +0.0261] |
| Deep Sets, hand features + raw sequences, 12 seasons | +0.0147 | [-0.0001, +0.0285] |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | +0.0226 | [+0.0067, +0.0380] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0236 | [+0.0097, +0.0372] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | +0.0205 | [+0.0056, +0.0348] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | +0.0228 | [+0.0091, +0.0362] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0306 | [+0.0160, +0.0454] |
| Deep Sets, hand features + raw sequences, 4 seasons | +0.0085 | [-0.0055, +0.0225] |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | +0.0172 | [+0.0020, +0.0321] |
| Deep Sets, hand-built features, 12 seasons | +0.0095 | [-0.0051, +0.0231] |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | +0.0225 | [+0.0079, +0.0370] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | +0.0185 | [+0.0052, +0.0318] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0192 | [+0.0059, +0.0321] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | +0.0189 | [+0.0054, +0.0318] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0303 | [+0.0164, +0.0440] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | +0.0192 | [+0.0063, +0.0315] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | +0.0306 | [+0.0161, +0.0449] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | +0.0167 | [+0.0034, +0.0291] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | +0.0284 | [+0.0143, +0.0422] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | +0.0180 | [+0.0046, +0.0312] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | +0.0288 | [+0.0137, +0.0440] |
| Deep Sets, hand-built features, 4 seasons | +0.0038 | [-0.0109, +0.0179] |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | +0.0170 | [+0.0019, +0.0318] |
| Deep Sets, hand-built features, 8 seasons | +0.0081 | [-0.0061, +0.0214] |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | +0.0205 | [+0.0057, +0.0348] |
| Deep Sets, raw sequences (GRU), 12 seasons | +0.0145 | [-0.0005, +0.0286] |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | +0.0229 | [+0.0071, +0.0382] |
| Deep Sets, raw sequences (GRU), 4 seasons | +0.0103 | [-0.0042, +0.0243] |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | +0.0172 | [+0.0009, +0.0327] |
| Jev, real names, recalibrated on validation | +0.0093 | [-0.0016, +0.0196] |
| Jev, anonymised, recalibrated on validation | +0.0072 | [-0.0034, +0.0174] |

**Validation: log loss improvement over the 7-number logistic regression, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Logistic regression (production features) | -0.0115 | [-0.0216, -0.0008] |
| Player model, actual roster (optimistic) | +0.0070 | [-0.0070, +0.0207] |
| Ablation: logreg on player features, actual roster | +0.0088 | [-0.0041, +0.0223] |
| Player model, previous-game roster | -0.0062 | [-0.0159, +0.0036] |
| Ablation: logreg on player features, previous-game roster | -0.0057 | [-0.0156, +0.0038] |
| Jev, real names, previous-game roster | -0.0588 | [-0.0801, -0.0384] |
| Jev, anonymised, previous-game roster | -0.0782 | [-0.1028, -0.0545] |
| Team Elo (settings tuned on validation) | -0.0104 | [-0.0170, -0.0032] |
| Neural Elo: K, home bonus, carryover learned by gradient | -0.0102 | [-0.0171, -0.0028] |
| Neural Elo with a learned update network | -0.0103 | [-0.0173, -0.0030] |
| Player Elo, previous-game roster | -0.0078 | [-0.0127, -0.0029] |
| Ridge plus-minus, previous-game roster | -0.0042 | [-0.0092, +0.0007] |
| Player Elo, actual roster (optimistic) | -0.0008 | [-0.0075, +0.0061] |
| Ridge plus-minus, actual roster (optimistic) | +0.0029 | [-0.0049, +0.0105] |
| Production features + team Elo | -0.0073 | [-0.0150, +0.0005] |
| Production features + player Elo and ridge | -0.0016 | [-0.0068, +0.0032] |
| Production features + all rating scalars | -0.0017 | [-0.0064, +0.0031] |
| Production features + rating scalars, actual roster (optimistic) | +0.0047 | [-0.0031, +0.0123] |
| Deep Sets, hand features + raw sequences, 12 seasons | -0.0016 | [-0.0112, +0.0083] |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | +0.0083 | [-0.0038, +0.0212] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0070 | [+0.0005, +0.0131] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | +0.0071 | [-0.0007, +0.0146] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | +0.0052 | [-0.0019, +0.0120] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0149 | [+0.0042, +0.0260] |
| Deep Sets, hand features + raw sequences, 4 seasons | -0.0078 | [-0.0184, +0.0024] |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | +0.0007 | [-0.0128, +0.0144] |
| Deep Sets, hand-built features, 12 seasons | -0.0019 | [-0.0112, +0.0070] |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | +0.0061 | [-0.0059, +0.0184] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | +0.0053 | [-0.0006, +0.0113] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0041 | [-0.0019, +0.0096] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | +0.0028 | [-0.0034, +0.0084] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0137 | [+0.0039, +0.0238] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | +0.0037 | [-0.0026, +0.0097] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | +0.0138 | [+0.0038, +0.0239] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | +0.0031 | [-0.0036, +0.0094] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | +0.0113 | [+0.0010, +0.0219] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | +0.0024 | [-0.0044, +0.0089] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | +0.0141 | [+0.0033, +0.0250] |
| Deep Sets, hand-built features, 4 seasons | -0.0057 | [-0.0159, +0.0042] |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | +0.0002 | [-0.0127, +0.0133] |
| Deep Sets, hand-built features, 8 seasons | -0.0038 | [-0.0127, +0.0047] |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | +0.0050 | [-0.0071, +0.0173] |
| Deep Sets, raw sequences (GRU), 12 seasons | -0.0010 | [-0.0102, +0.0084] |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | +0.0084 | [-0.0036, +0.0207] |
| Deep Sets, raw sequences (GRU), 4 seasons | -0.0034 | [-0.0138, +0.0071] |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | +0.0052 | [-0.0077, +0.0182] |
| Jev, real names, recalibrated on validation | -0.0134 | [-0.0226, -0.0040] |
| Jev, anonymised, recalibrated on validation | -0.0148 | [-0.0244, -0.0053] |

**Test: log loss improvement over the 7-number logistic regression, paired bootstrap 95% CI** (positive = better)

| Model | Mean | 95% CI |
|---|---|---|
| Logistic regression (production features) | -0.0160 | [-0.0266, -0.0046] |
| Player model, actual roster (optimistic) | +0.0116 | [-0.0020, +0.0244] |
| Ablation: logreg on player features, actual roster | +0.0089 | [-0.0046, +0.0221] |
| Player model, previous-game roster | +0.0003 | [-0.0094, +0.0093] |
| Ablation: logreg on player features, previous-game roster | -0.0038 | [-0.0141, +0.0060] |
| Jev, real names, previous-game roster | -0.0519 | [-0.0736, -0.0316] |
| Jev, anonymised, previous-game roster | -0.0686 | [-0.0934, -0.0447] |
| Team Elo (settings tuned on validation) | -0.0027 | [-0.0098, +0.0044] |
| Neural Elo: K, home bonus, carryover learned by gradient | -0.0020 | [-0.0094, +0.0052] |
| Neural Elo with a learned update network | -0.0029 | [-0.0102, +0.0042] |
| Player Elo, previous-game roster | -0.0033 | [-0.0090, +0.0020] |
| Ridge plus-minus, previous-game roster | +0.0000 | [-0.0052, +0.0054] |
| Player Elo, actual roster (optimistic) | +0.0043 | [-0.0027, +0.0109] |
| Ridge plus-minus, actual roster (optimistic) | +0.0073 | [-0.0005, +0.0148] |
| Production features + team Elo | -0.0102 | [-0.0178, -0.0025] |
| Production features + player Elo and ridge | -0.0063 | [-0.0119, -0.0012] |
| Production features + all rating scalars | -0.0049 | [-0.0098, -0.0002] |
| Production features + rating scalars, actual roster (optimistic) | -0.0008 | [-0.0090, +0.0067] |
| Deep Sets, hand features + raw sequences, 12 seasons | -0.0012 | [-0.0111, +0.0077] |
| Deep Sets, hand features + raw sequences, 12 seasons, actual roster (optimistic) | +0.0066 | [-0.0061, +0.0181] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0076 | [+0.0010, +0.0138] |
| Deep Sets, hand features + raw sequences, 12 seasons, + per-player Elo/ridge ratings, + rating scalars, + league context | +0.0045 | [-0.0032, +0.0121] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars | +0.0068 | [-0.0003, +0.0129] |
| Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0146 | [+0.0039, +0.0245] |
| Deep Sets, hand features + raw sequences, 4 seasons | -0.0075 | [-0.0170, +0.0021] |
| Deep Sets, hand features + raw sequences, 4 seasons, actual roster (optimistic) | +0.0012 | [-0.0111, +0.0128] |
| Deep Sets, hand-built features, 12 seasons | -0.0065 | [-0.0162, +0.0024] |
| Deep Sets, hand-built features, 12 seasons, actual roster (optimistic) | +0.0065 | [-0.0050, +0.0171] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings | +0.0025 | [-0.0041, +0.0083] |
| Deep Sets, hand-built features, 12 seasons, + per-player Elo/ridge ratings, + rating scalars | +0.0032 | [-0.0028, +0.0086] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars | +0.0029 | [-0.0031, +0.0083] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, actual roster (optimistic) | +0.0143 | [+0.0044, +0.0232] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context | +0.0032 | [-0.0027, +0.0088] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + league context, actual roster (optimistic) | +0.0146 | [+0.0048, +0.0236] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights | +0.0008 | [-0.0057, +0.0068] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + recency weights, actual roster (optimistic) | +0.0124 | [+0.0025, +0.0218] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year | +0.0020 | [-0.0043, +0.0081] |
| Deep Sets, hand-built features, 12 seasons, + rating scalars, + season year, actual roster (optimistic) | +0.0128 | [+0.0021, +0.0224] |
| Deep Sets, hand-built features, 4 seasons | -0.0122 | [-0.0219, -0.0023] |
| Deep Sets, hand-built features, 4 seasons, actual roster (optimistic) | +0.0011 | [-0.0110, +0.0124] |
| Deep Sets, hand-built features, 8 seasons | -0.0078 | [-0.0168, +0.0008] |
| Deep Sets, hand-built features, 8 seasons, actual roster (optimistic) | +0.0045 | [-0.0071, +0.0153] |
| Deep Sets, raw sequences (GRU), 12 seasons | -0.0015 | [-0.0112, +0.0075] |
| Deep Sets, raw sequences (GRU), 12 seasons, actual roster (optimistic) | +0.0069 | [-0.0057, +0.0179] |
| Deep Sets, raw sequences (GRU), 4 seasons | -0.0057 | [-0.0158, +0.0038] |
| Deep Sets, raw sequences (GRU), 4 seasons, actual roster (optimistic) | +0.0012 | [-0.0110, +0.0128] |
| Jev, real names, recalibrated on validation | -0.0067 | [-0.0168, +0.0031] |
| Jev, anonymised, recalibrated on validation | -0.0088 | [-0.0189, +0.0014] |

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
