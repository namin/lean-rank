# `lean-rank`

A pipeline that learns to **rank useful premises** for a target declaration using structural analysis of Lean types and dependency graphs.

We use [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data) (included as submodule):
- `premises.txt` — from `lake exe premises Mathlib` (declaration dependencies)
- `declaration_types.txt` — from `lake exe declaration_types Mathlib` (names + types)
- `declaration_structures` — our custom Lean metaprogram that extracts structural features, from `lake exe declaration_structures Mathlib`

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
- `data/declaration_structures.txt` (output of `lake exe declaration_structures Mathlib`)

## Run

```bash
# Standard run (builds everything needed)
./run_walkthrough.sh

# Force rebuild everything from scratch
FORCE=1 ./run_walkthrough.sh
```

The pipeline automatically:
1. Extracts structural features via Lean metaprogramming (if needed)
2. Combines structural + text features (313 dimensions)
3. Trains model with enhanced features
4. Calculates use-cost for what-if analysis

No configuration needed - it just works with the best available features!

## Documentation

- [DESIGN.md](DESIGN.md) - System architecture and rationale
- [STEPS.md](STEPS.md) - Detailed step-by-step walkthrough
- [lean-training-data/README.md](lean-training-data/README.md#declaration_structures) - Includes documentation for our `declaration_structures` tool

