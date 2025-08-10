# Lean Rank

This is a single-path, text-only pipeline that learns to **rank useful premises**
for a target declaration using just two inputs:

We use two computed files using [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data):
- `premises.txt` — from `lake exe premises Mathlib` (declaration dependencies)
- `declaration_types.txt` — from `lake exe declaration_types Mathlib` (names + types)

The pipeline handles **cold-start**: if you know a new target’s type, it can suggest premises.
If can also suggest a productivity score for a new target by calculating how it would be adopted by the current model.

## Quick start

### 0) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Place your inputs
```
data/premises.txt
data/declaration_types.txt
```

### 2) Build dataset (streaming)
```bash
python -m src.build_dataset --premises data/premises.txt --out_dir data/processed
```
Outputs:
- `data/processed/nodes.parquet` (id, name)
- `data/processed/contexts.jsonl` (one JSON per target with its positives)
- `data/processed/meta.json`

### 3) Parse declaration types and build features
```bash
python -m src.build_decltypes   --decltypes data/declaration_types.txt   --nodes data/processed/nodes.parquet   --out_dir data/processed

python -m src.tasks.build_type_features   --decltypes data/processed/decltypes.parquet   --nodes data/processed/nodes.parquet   --out data/processed/type_features.npz   --buckets 128
```

### 4) Train the text ranker (CPU, fast)
```bash
python -m src.tasks.train_text_ranker   --features data/processed/type_features.npz   --contexts data/processed/contexts.jsonl   --out_ckpt outputs/text_ranker.pt   --emb_dim 64 --batch 512 --neg_per_pos 8 --epochs 1
```

### 5) Score rankings for all targets
```bash
python -m src.tasks.score_text_rankings   --features data/processed/type_features.npz   --contexts data/processed/contexts.jsonl   --nodes data/processed/nodes.parquet   --ckpt outputs/text_ranker.pt   --out data/processed/rankings.parquet   --topk 50 --batch 512 --chunk 128000
```

### 6) Explain results (human-readable)
```bash
python -m src.tasks.explain_rankings   --rankings data/processed/rankings.parquet   --nodes data/processed/nodes.parquet   --contexts data/processed/contexts.jsonl   --out data/processed/rankings_explained.csv   --format csv --sort_by recall --topk 50 --limit 500
```

### 7) Cold start
```bash
python -m src.tasks.score_on_text \
  --features data/processed/type_features.npz \
  --nodes data/processed/nodes.parquet \
  --ckpt outputs/text_ranker.pt \
    --type_string "∀ {X : TopCat} (x : ↑X) (U : TopologicalSpace.OpenNhds (↑(CategoryTheory.CategoryStruct.id X) x)), (TopologicalSpace.OpenNhds.map (CategoryTheory.CategoryStruct.id X) x).obj U = U" \
  --out data/processed/new_target_rankings.parquet \
  --topk 50 --buckets 128
```

### 8) Score productivity
```bash
python -m src.tasks.score_productivity \
  --type_string "∀ {X : TopCat} (x : ↑X) (U : TopologicalSpace.OpenNhds (↑(CategoryTheory.CategoryStruct.id X) x)), (TopologicalSpace.OpenNhds.map (CategoryTheory.CategoryStruct.id X) x).obj U = U" \
  --features data/processed/type_features.npz \
  --contexts data/processed/contexts.jsonl \
  --rankings data/processed/rankings.parquet \
  --nodes data/processed/nodes.parquet \
  --decltypes data/processed/decltypes.parquet \
  --ckpt outputs/text_ranker.pt \
  --buckets 128 --k_list 10,20,50 \
  --show_hits 50 --hits_limit 25 \
  --target_kinds theorem,lemma \
  --target_prefixes "TopologicalSpace."
```

## Notes
- All builders are streaming and memory-light (8M+ line premises OK).
- The ranker uses a **shared MLP encoder** for target and premise type features.
- The productivity score calculates the model’s predicted adoption@K for the new statement:  if we treated every current target as a proxy for future ones, your statement would land in the Top-K suggestions.
