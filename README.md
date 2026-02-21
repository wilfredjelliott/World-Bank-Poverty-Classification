# World-Bank-Poverty-Classification
Top 2.3% solution (30th/1,322) for the World Bank poverty classification competition, with monotonicity constraints and isotonic calibration across 19 classifiers.
# World Bank Poverty Imputation — 30th / 1,322 teams

Top 2.3% solution for the [World Bank Survey-to-Survey Imputation Challenge](https://www.drivendata.org/competitions/305/competition-worldbank-poverty/) on DrivenData.

The competition asks you to predict both per-capita household consumption and the share of households below 19 poverty thresholds, given anonymised survey data with no consumption information. The two tasks are scored jointly.

## Why monotonicity matters

The poverty distribution predictions must satisfy a structural constraint: the share of households below a higher poverty line must be at least as high as the share below any lower line. This isn't a modelling preference — it's a logical necessity. A prediction that says 40% of households live below $5/day but only 35% live below $7/day is not merely inaccurate, it's impossible.

Most approaches treat the 19 thresholds as independent classification problems. This is fast but produces predictions that routinely violate monotonicity, especially at neighbouring thresholds where the true rates are close. My approach enforces monotonicity as a hard constraint rather than hoping the models learn it implicitly.

## Approach

### Poverty distribution (19 classifiers)

For each of the 19 poverty thresholds:

1. **Binary classifier**: LightGBM trained to predict P(consumption < threshold), with class weights adjusted for imbalance.
2. **Out-of-fold isotonic calibration**: 3-fold stratified CV produces out-of-fold probability estimates. Isotonic regression maps these to calibrated probabilities — correcting for LightGBM's tendency to produce overconfident or poorly calibrated outputs.
3. **Shrinkage**: Calibrated probabilities are mixed with the survey-level base rate on a per-threshold grid search, regularising toward the empirical prior. This stabilises predictions at extreme thresholds where data is sparse.
4. **Monotonicity enforcement**: After all 19 classifiers produce their calibrated, shrunk probabilities, a cumulative maximum (`np.maximum.accumulate`) is applied across thresholds to guarantee that P(below $t_i$) ≤ P(below $t_j$) whenever $t_i < t_j$.

The monotonicity step is not a post-hoc fix — it's the reason the pipeline works. Without it, the 19 classifiers disagree with each other in ways that are structurally impossible, and the scoring metric penalises this heavily.

### Household consumption (regression ensemble)

1. Two LightGBM regressors trained in log-space: one with Huber loss (robust to outliers), one with median quantile loss.
2. The best single model or geometric mean is selected on a held-out tune set using weighted MAPE.
3. Isotonic regression in log-space calibrates the final predictions, followed by a multiplicative scaling factor optimised on the tune set.

### Feature engineering

- Cross features between log utility expenditure per person and log household size.
- Aggregated binary consumption indicators (food items consumed) into per-person counts.
- Hashed categorical encoding (2^17 buckets, dual hash) to handle high-cardinality survey variables without target leakage.

## Result

**30th out of 1,322 teams** on the final leaderboard. The score was higher because the predictions were correct — not just in the sense of minimising a loss function, but in the sense of producing outputs that respect the known structure of the problem.

## Relevance to AI safety

This competition shaped how I think about model constraints. The insight — that outputs must satisfy known invariants, and that models which violate them are producing impossible results regardless of their loss — generalises beyond poverty measurement. In alignment evaluation, a model that claims inability on a dangerous task but performs structurally identical reasoning on a benign framing is violating a consistency constraint. Detecting and enforcing these kinds of structural guarantees is, I think, an underexplored part of the alignment toolbox.

## Requirements

```
numpy
pandas
lightgbm
scikit-learn
joblib
```
