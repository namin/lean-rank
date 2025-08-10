# `lean-rank`

A pipeline that learns to **rank useful premises** for a target declaration using just two inputs, premises and declaration types.

We use [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data):
- `premises.txt` — from `lake exe premises Mathlib` (declaration dependencies)
- `declaration_types.txt` — from `lake exe declaration_types Mathlib` (names + types)

The pipeline handles **cold-start**: if you know a new target’s type, it can suggest premises.
If can also suggest a productivity score for a new target by calculating how it would be adopted by the current model.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# core (always)
pip install -r requirements.txt

# optional: GNN (choose a combo that matches your Torch build; see comments inside)
pip install -r requirements-gnn.txt
```

## Data

- `data/premises.txt` (output of `lake exe premises Mathlib`)
- `data/declaration_types.txt` (output of `lake exe declaration_types Mathlib`)

## Run

Try `./run_walkthrough.sh`.

## Docs

- Design doc in [DESIGN.md](DESIGN.md).
- Detailed step walkthroughts in [STEPS.md](STEPS.md).

