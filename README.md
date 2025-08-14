# `lean-rank`

A pipeline that learns to **rank useful premises** for a target declaration using structural analysis of Lean types and dependency graphs.

We use [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data) (included as submodule):
- `premises.txt` — from `lake exe premises Mathlib` (declaration dependencies)
- `declaration_types.txt` — from `lake exe declaration_types Mathlib` (names + types)
- `declaration_structures.jsonl` — our custom Lean metaprogram that extracts structural features, from `lake exe declaration_structures Mathlib`

## Key Features

- **Structural type analysis**: Extracts explicit premise counts, nesting depth, typeclass constraints, and more via Lean metaprogramming
- **Use-cost calculation**: Based on actual structure (explicit premises, nesting depth, etc.) not string heuristics
- **Cold-start ranking**: Suggests premises for new statements given only their type
- **Productivity scoring**: Estimates how often a new statement would be adopted by existing theorems
- **Graph-aware what-if analysis**: Simulates adding a new statement to the dependency graph

## Setup

Clone the repository with `--recursive` to ensure submodules are included.

Use a Python environment.
```bash
# core (always)
pip install -r requirements.txt

# optional: GNN (choose a combo that matches your Torch build; see comments inside)
pip install -r requirements-gnn.txt
```

## Data

Generated from the `lean-training data` submodule.

- `data/premises.txt` (output of `lake exe premises Mathlib`)
- `data/declaration_types.txt` (output of `lake exe declaration_types Mathlib`)
- `data/declaration_structures.jsonl` (output of `lake exe declaration_structures Mathlib`)

## Run

```bash
# Standard run (builds everything needed)
./run_walkthrough.sh

# Force rebuild everything from scratch
FORCE=1 ./run_walkthrough.sh
```

The pipeline automatically:
1. Extracts structural features
2. Combines structural + text features
3. Trains model
4. Calculates use-cost for what-if analysis

## Documentation

- [DESIGN.md](DESIGN.md) - System architecture and rationale
- [STEPS.md](STEPS.md) - Detailed step-by-step walkthrough
- [lean-training-data/README.md](lean-training-data/README.md#declaration_structures) - Includes documentation for our `declaration_structures` tool

## Analysis Tools

Beyond the main pipeline, we provide analysis scripts to understand feature-usage relationships:

### Feature-Usage Analysis

Analyzes how structural features relate to actual usage patterns in Mathlib:

```bash
# Run after the main pipeline completes
python3 -m src.tasks.analyze_feature_usage \
  --structures data/processed/structures.parquet \
  --graph_metrics data/processed/graph_metrics.parquet
```

This reveals insights like:
- Theorems with 1-2 premises are most frequently used (the "sweet spot")
- Deep nesting (`max_nesting_depth`) strongly predicts usage (+7.5 coefficient)
- Many universal quantifiers (`num_forall`) strongly predict non-usage (-7.9 coefficient)
- Only 58% of theorems are ever used, while 100% of constructors are used
- Structural features can predict usage with ~74% accuracy

### Compare Learned vs Formula-based Use-Cost

To see what the use-cost model learned:

```bash
python3 -c "
import pandas as pd
structures = pd.read_parquet('data/processed/structures.parquet')
learned = pd.read_parquet('data/processed/structures_with_learned_cost.parquet')
df = structures.merge(learned[['id', 'learned_use_cost']], on='id')
print(f'Formula cost: {df[\"use_cost\"].mean():.2f} ± {df[\"use_cost\"].std():.2f}')
print(f'Learned cost: {df[\"learned_use_cost\"].mean():.2f} ± {df[\"learned_use_cost\"].std():.2f}')
print(f'Correlation: {df[\"use_cost\"].corr(df[\"learned_use_cost\"]):.2f}')
"
```

The learned model often produces different costs than the formula, revealing that usage patterns don't follow simple structural rules.

### Productivity Prediction

Learn to predict how many downstream theorems a new statement will enable:

```bash
python3 -m src.tasks.learn_productivity_model \
  --structures data/processed/structures.parquet \
  --graph_metrics data/processed/graph_metrics.parquet
```

Key insights from this analysis:
- Identifies a "productivity sweet spot": theorems with 1-3 premises, depth 3-8, ≤3 foralls are **2.72x more productive**
- Shows that simple, fundamental theorems (Eq, id, congrArg) enable the most downstream work
- Reveals that structural complexity alone poorly predicts productivity (R² ≈ 0.02)
- High-productivity theorems have MORE arrows but FEWER universal quantifiers


