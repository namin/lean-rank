# Lean Rank

A pipeline that learns to **rank useful premises** for a target declaration using just two inputs, premises and declaration types.

We use [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data):
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
- The productivity score calculates the model’s predicted adoption@K for the new statement:  if we treated every current target as a proxy for future ones, your statement would land in the Top-K suggestions. `lift = adoption@K / (K/N)`, where `N` is the total candidate premises.

# Principled Graph-Based Productivity for Mathlib

This add-on computes **graph-aware productivity** for Lean Mathlib using the **premise → target** graph.
It provides:
- A **transparent analytic score** combining direct uses, discounted transitive reach (Katz), and a cost penalty.
- An optional **GraphSAGE** GNN regressor that learns to predict that productivity from graph structure (+ optional text features).

## Install
```bash
pip install numpy pandas pyarrow scipy tqdm
# (optional, for the GNN)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3 torch-scatter torch-sparse torch-cluster   --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

## 1) Build graph + analytic productivity
```bash
python -m src.tasks.build_graph   --contexts data/processed/contexts.jsonl   --nodes data/processed/nodes.parquet   --out_edges data/processed/edge_index.npz   --out_metrics data/processed/graph_metrics.parquet   --katz_k 8 --katz_beta 0.2   --reach_h 3   --alpha 0.2 --gamma 0.5
```
Writes:
- `edge_index.npz` with arrays `src`, `dst` (premise→target)
- `graph_metrics.parquet` with: `in_deg, out_deg, katz_k8_b0.2, reach_h3, pagerank_d0.85` and analytic productivity
  - Katz-based: `prod_katz_a0.2_b0.2_g0.5 = log1p((out_deg + 0.2*katz) / (1+in_deg)^0.5)`
  - Reach-based: `prod_reach_a0.2_h3_g0.5 = log1p((out_deg + 0.2*reach) / (1+in_deg)^0.5)`

## 2) (Optional) Train GraphSAGE to predict productivity
```bash
python -m src.tasks.train_gnn_productivity \
  --edge_index data/processed/edge_index.npz \
  --graph_metrics data/processed/graph_metrics.parquet \
  --type_features data/processed/type_features.npz \
  --out_ckpt outputs/gnn_prod.pt \
  --label_col prod_katz_a0.2_b0.2_g0.5 \
  --epochs 3 --batch_nodes 65536 --fanout 15,15 --device cpu
```

## 3) Score all nodes with the GNN
```bash
python -m src.tasks.score_gnn_productivity \
  --edge_index data/processed/edge_index.npz \
  --graph_metrics data/processed/graph_metrics.parquet \
  --type_features data/processed/type_features.npz \
  --nodes data/processed/nodes.parquet \
  --ckpt outputs/gnn_prod.pt \
  --out data/processed/gnn_productivity.parquet \
  --fanout 15,15 --batch_nodes 65536 --device cpu
```
