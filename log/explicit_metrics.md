# Theorem Importance Analysis Report

*Generated: 2025-08-15 18:26*

## Executive Summary

**Dataset**: 2121 theorems and lemmas analyzed from number theory domain

**Methodology**: Three explicit, interpretable importance metrics:
1. **Direct Usage**: Number of theorems that directly call this theorem
2. **Transitive Dependencies**: Number of theorems that depend on this one (full closure)
3. **Combined Score**: `(direct × transitive) / (existential_quantifiers + 1)`

### Top Theorems by Each Metric

**Most Directly Used**: `Pell.matiyasevic`
- 18 direct uses

**Most Dependencies**: `Nat.modCore_eq_mod`
- 569 theorems depend on it

**Highest Combined Score**: `Nat.gcd_rec`
- Score: 1200.0

### Key Statistics

| Metric | Mean | Median | Max | Zero Count |
|--------|------|--------|-----|------------|
| Direct Usage | 1.64 | 1 | 18 | 312 (14.7%) |
| Transitive (Full) | 11.44 | 1 | 569 | 761 (35.9%) |
| Combined Score | 13.76 | 0 | 1200 | - |

## Metric 1: Direct Usage Count

Number of theorems that directly call this theorem. This measures immediate utility.

### Top 10 Theorems by Direct Usage

| Rank | Uses | Theorem Name |
|------|------|--------------|
| 1 | 18 | `Pell.matiyasevic` |
| 2 | 15 | `ArithmeticFunction.moebius_mul_coe_zeta` |
| 3 | 14 | `ArithmeticFunction.IsMultiplicative.mul` |
| 4 | 13 | `ZMod.eq_unit_mul_divisor` |
| 5 | 12 | `PrimeSpectrum.existsUnique_idempotent_basicOpen_eq_of_isClopen` |
| 6 | 11 | `ZMod.Ico_map_valMinAbs_natAbs_eq_Ico_map_id` |
| 7 | 11 | `Fermat42.not_minimal` |
| 8 | 11 | `Pell.eq_pow_of_pell` |
| 9 | 10 | `ArithmeticFunction.IsMultiplicative.lcm_apply_mul_gcd_apply` |
| 10 | 10 | `Pell.y_dvd_iff` |

### Usage Distribution

```
0      uses: █████████████  312 ( 14.7%)
1      uses: ████████████████████████████████████████  947 ( 44.6%)
2      uses: █████████████████████  503 ( 23.7%)
3-4    uses: ██████████  259 ( 12.2%)
5-9    uses: ███   88 (  4.1%)
10-19  uses:    12 (  0.6%)
20+    uses:     0 (  0.0%)
```

## Metric 2: Transitive Dependencies (Full Closure)

Number of theorems that depend on this one, directly or indirectly through dependency chains.

### Full Closure vs 3-Hop Approximation

| Statistic | 3-Hop | Full Closure | Ratio |
|-----------|-------|--------------|-------|
| Mean | 6.56 | 11.44 | 1.74x |
| Median | 1 | 1 | - |
| Max | 216 | 569 | 2.63x |
| Correlation | - | - | 0.811 |

### Top 10 Theorems by Transitive Dependencies

| Rank | Full Deps | 3-Hop | Diff | Theorem Name |
|------|-----------|-------|------|--------------|
| 1 | 569 | 71 | +498 | `Nat.modCore_eq_mod` |
| 2 | 568 | 216 | +352 | `Nat.mod_eq` |
| 3 | 506 | 144 | +362 | `Nat.mod_eq_of_lt` |
| 4 | 483 | 71 | +412 | `Nat.mod_eq_sub_mod` |
| 5 | 465 | 85 | +380 | `Nat.mod_lt` |
| 6 | 457 | 103 | +354 | `Nat.mod_zero` |
| 7 | 429 | 135 | +294 | `Nat.gcd_zero_left` |
| 8 | 426 | 82 | +344 | `Nat.gcd_succ` |
| 9 | 423 | 66 | +357 | `Nat.gcd_zero_right` |
| 10 | 400 | 96 | +304 | `Nat.gcd_rec` |

### Largest Underestimations by 3-Hop

These theorems have much larger transitive impact than 3-hop analysis suggests:

| Theorem | 3-Hop | Full | Ratio |
|---------|-------|------|-------|
| `Nat.modCore_eq_mod` | 71 | 569 | 8.0x |
| `Nat.mod_eq_sub_mod` | 71 | 483 | 6.8x |
| `Nat.mod_lt` | 85 | 465 | 5.5x |
| `Nat.mod_eq_of_lt` | 144 | 506 | 3.5x |
| `Nat.gcd_zero_right` | 66 | 423 | 6.4x |
| `Nat.mod_zero` | 103 | 457 | 4.4x |
| `Nat.mod_eq` | 216 | 568 | 2.6x |
| `Nat.gcd_succ` | 82 | 426 | 5.2x |
| `Nat.gcd.induction` | 67 | 393 | 5.9x |
| `Nat.gcd_rec` | 96 | 400 | 4.2x |

## Metric 3: Combined Weighted Score

**Formula**: `(direct_usage × transitive_deps) / (existential_quantifiers + 1)`

This metric balances immediate utility with foundational importance while penalizing complex statements.

### Top 10 Theorems by Combined Score

| Rank | Score | Direct | Transitive | ∃ | Theorem Name |
|------|-------|--------|------------|---|--------------|
| 1 | 1200.0 | 3 | 400 | - | `Nat.gcd_rec` |
| 2 | 930.0 | 2 | 465 | - | `Nat.mod_lt` |
| 3 | 846.0 | 2 | 423 | - | `Nat.gcd_zero_right` |
| 4 | 810.0 | 3 | 270 | - | `Nat.gcd_dvd` |
| 5 | 765.0 | 3 | 255 | - | `Nat.gcd_mul_left` |
| 6 | 672.0 | 8 | 84 | - | `Nat.Coprime.gcd_mul_left_cancel` |
| 7 | 568.0 | 1 | 568 | - | `Nat.mod_eq` |
| 8 | 528.0 | 3 | 176 | - | `Nat.Prime.coprime_iff_not_dvd` |
| 9 | 506.0 | 1 | 506 | - | `Nat.mod_eq_of_lt` |
| 10 | 483.0 | 1 | 483 | - | `Nat.mod_eq_sub_mod` |

### Ranking Stability Analysis

How rankings change when using full transitive closure vs 3-hop:

| Stability Level | Rank Change | % of Theorems |
|-----------------|-------------|---------------|
| Stable | ≤5 | 64.6% |
| Moderate | 6-20 | 11.1% |
| Volatile | >20 | 24.3% |

#### Biggest Rank Improvements (with full transitive)

| Theorem | 3-Hop Rank | Full Rank | Change |
|---------|------------|-----------|--------|
| `IsPrimePow.not_unit` | #553 | #164 | ↑389 |
| `Nat.gcd_dvd_gcd_mul_left_right` | #553 | #204 | ↑349 |
| `Nat.gcd_dvd_gcd_mul_right_right` | #553 | #204 | ↑349 |
| `Nat.Prime.factorization_pos_of_dvd` | #587 | #269 | ↑318 |
| `Nat.gcd_mul_right_left` | #481 | #204 | ↑277 |

#### Biggest Rank Drops (with full transitive)

| Theorem | 3-Hop Rank | Full Rank | Change |
|---------|------------|-----------|--------|
| `Pell.xy_coprime` | #457 | #699 | ↓242 |
| `Int.gcd_mul_right` | #339 | #575 | ↓236 |
| `PythagoreanTriple.gcd_dvd` | #235 | #446 | ↓211 |
| `Pell.IsFundamental.d_pos` | #182 | #391 | ↓209 |
| `Pell.pell_eq` | #386 | #575 | ↓189 |

## Existential Quantifier Analysis

How existential quantifiers (∃) affect theorem importance metrics.

### Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Theorems with ∃ | 107 | 5.0% |
| Theorems without ∃ | 2014 | 95.0% |
| Mean ∃ count (when present) | 1.42 | - |
| Max ∃ count | 6 | - |

### Impact on Metrics

| Metric | With ∃ | Without ∃ | Ratio |
|--------|--------|-----------|-------|
| Avg Direct Usage | 2.56 | 1.59 | 1.61x |
| Avg Transitive Deps | 4.58 | 11.80 | 0.39x |
| Avg Combined Score | 3.26 | 14.32 | 0.23x |

**Interpretation**: Theorems with existential quantifiers tend to be more specialized (higher direct usage) but less foundational (lower transitive reach).

## Correlation Analysis

How different metrics relate to each other.

### Correlation Matrix

| Metric | Direct Usage | Trans-3hop | Trans-Full | Combined | ∃ Count | ∀ Count |
|------|------------|----------|----------|--------|-------|-------|
| Direct Usage | 1.00 | -0.10 | -0.07 | 0.15 | 0.21 | 0.02 |
| Trans-3hop | -0.10 | 1.00 | 0.81 | 0.54 | -0.03 | -0.10 |
| Trans-Full | -0.07 | 0.81 | 1.00 | 0.70 | -0.03 | -0.11 |
| Combined | 0.15 | 0.54 | 0.70 | 1.00 | -0.03 | -0.08 |
| ∃ Count | 0.21 | -0.03 | -0.03 | -0.03 | 1.00 | 0.01 |
| ∀ Count | 0.02 | -0.10 | -0.11 | -0.08 | 0.01 | 1.00 |

### Key Insights

- **Direct vs Transitive correlation**: -0.072
  - Negative correlation suggests foundational theorems are rarely used directly

- **3-Hop vs Full Transitive correlation**: 0.811
  - High correlation but significant differences in absolute values

- **Existential quantifiers correlations**:
  - With direct usage: 0.208
  - With transitive deps: -0.034

## Conclusions and Recommendations

### Key Findings

1. **Full transitive closure is essential**: 3-hop analysis misses significant dependency chains, underestimating importance by up to 8x for some theorems.

2. **Different usage patterns**: Foundational theorems (high transitive) and utility theorems (high direct) serve different roles in the library.

3. **Sparse direct usage**: 14.7% of theorems have zero direct usage, suggesting many intermediate lemmas.

4. **Existential quantifiers indicate specialization**: Theorems with ∃ have higher direct usage but lower transitive reach.

### Recommended Metric

**Use Metric 3 (Combined Score with Full Transitive Closure)** for overall importance ranking because it:
- Balances immediate utility with foundational importance
- Accounts for statement complexity
- Provides fully explainable scores

### Applications

These metrics can be used for:
- **Library maintenance**: Identify unused theorems for potential removal
- **Documentation priority**: Focus on high-impact theorems
- **Teaching order**: Start with foundational theorems
- **ML training data**: Use as ground truth for importance prediction models

---

*Report generated using explicit, interpretable metrics as suggested by domain expert.*
*All scores can be manually verified by counting dependencies in the theorem library.*