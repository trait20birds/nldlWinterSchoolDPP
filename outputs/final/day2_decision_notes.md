# Day 2 Decision Notes

- Selected primary model: **tvae**
- Real baseline (TRTR) macro-F1: 0.9623

## Decision Criteria
- 1) Lowest validation rule violation rate
- 2) Highest TSTR macro-F1 (utility on held-out real test)
- 3) Lower exact-overlap and suspiciously-close privacy risk
- 4) Smaller fairness gap by group

## Ranked Models
- tvae: violation=0.0000, TSTR_F1=0.5319, overlap=0.0000, fairness_gap=0.4286
- ctgan: violation=0.0000, TSTR_F1=0.4757, overlap=0.0000, fairness_gap=0.2864
- privbayes: violation=0.0000, TSTR_F1=0.3736, overlap=0.0000, fairness_gap=0.2634
