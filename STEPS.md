# `lean-rank`

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
# core (always)
pip install -r requirements.txt

# optional: GNN (choose a combo that matches your Torch build; see comments inside)
pip install -r requirements-gnn.txt

# optional: FAISS
pip install -r requirements-faiss.txt
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
# Basic type features (char-3grams, etc.)
python -m src.build_decltypes   --decltypes data/declaration_types.txt   --nodes data/processed/nodes.parquet   --out_dir data/processed

python -m src.tasks.build_type_features   --decltypes data/processed/decltypes.parquet   --nodes data/processed/nodes.parquet   --out data/processed/type_features.npz   --buckets 128

# NEW: Structural features via Lean metaprogramming
# First, extract structures (this runs Lean and can take ~30 min for full Mathlib)
cd lean-training-data && lake exe declaration_structures Mathlib > ../data/declaration_structures.jsonl

# Then build combined structural + text features
python src/build_declaration_structures.py   --structures-jsonl data/declaration_structures.jsonl   --nodes data/processed/nodes.parquet   --out-structures data/processed/structures.parquet   --out-features data/processed/structure_features.npz   --combine-with-text   --text-features data/processed/type_features.npz   --buckets 128
```

The structural features include:
- Explicit premise counts (key for usability!)
- Nesting depth and complexity metrics
- Sophisticated use-cost calculation
- Combined 313-dimensional representation

### 4) Train the text ranker (CPU, fast)
```bash
# Use combined structural features for better performance
python -m src.tasks.train_text_ranker   --features data/processed/structure_features.npz   --contexts data/processed/contexts.jsonl   --out_ckpt outputs/text_ranker.pt   --emb_dim 64 --batch 512 --neg_per_pos 8 --epochs 1
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

# Attaching the pipelines

```bash
python -m src.tasks.reweight_rankings \
  --rankings_in data/processed/rankings.parquet \
  --prior_parquet data/processed/graph_metrics.parquet \
  --prior_score_col prod_katz_a0.2_b0.2_g0.5 \
  --out data/processed/rankings_reweighted.parquet \
  --alpha 0.2 --mix linear --zscore_prior
```

```bash
python -m src.tasks.explain_rankings \
  --rankings data/processed/rankings_reweighted.parquet \
  --nodes data/processed/nodes.parquet \
  --contexts data/processed/contexts.jsonl \
  --out data/processed/rankings_reweighted_explained.csv \
  --format csv --topk 50 --limit 500
```

```bash
python -m src.tasks.distill_gnn_to_text \
  --type_features data/processed/type_features.npz \
  --labels_parquet data/processed/graph_metrics.parquet \
  --label_col prod_katz_a0.2_b0.2_g0.5 \
  --out_ckpt outputs/text_prod.pt \
  --epochs 3 --batch 8192 --lr 2e-3 --hidden 256 --device cpu
```

```
python -m src.tasks.score_text_prod_on_text \
  --ckpt outputs/text_prod.pt \
  --type_string "∀ {X : TopCat} (x : ↑X) (U : TopologicalSpace.OpenNhds (↑(CategoryTheory.CategoryStruct.id X) x)), (TopologicalSpace.OpenNhds.map (CategoryTheory.CategoryStruct.id X) x).obj U = U" \
  --buckets 128
```

# End‑to‑End Walkthrough

This guide shows a **single, coherent path** from raw Mathlib dumps to usable outputs:

- **Premise ranking** for every target (and human‑readable explanations)
- **Cold‑start ranking** for a brand‑new statement (by type string)
- **Productivity (adoption@K)** for a brand‑new statement
- *(Optional)* **Graph‑based productivity** (analytic + GNN) and how to attach it

It assumes you use data generated by [`kim-em/lean-training-data`](https://github.com/kim-em/lean-training-data).

---

## 0) Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Inputs** (place these files):
```
data/premises.txt             # from: lake exe premises Mathlib
data/declaration_types.txt    # from: lake exe declaration_types Mathlib
```

> All builders stream the inputs. An 8M+ line `premises.txt` is fine on a laptop.

---

## 1) Build the learning dataset (from premises)

Convert `premises.txt` into compact artifacts used for training and scoring.

```bash
python -m src.build_dataset   --premises data/premises.txt   --out_dir data/processed
```

**Outputs**
- `data/processed/nodes.parquet` — `(id:int, name:str)` mapping
- `data/processed/contexts.jsonl` — one JSON per **target**, with its positive **premises**
- `data/processed/meta.json` — counts and sanity stats

**Schema: `contexts.jsonl`**
```json
{"target_id": 123, "positives": [7, 42, 88, ...]}
```

---

## 2) Parse declaration types and build text features

First, turn `declaration_types.txt` into a table aligned with `nodes.parquet`:

```bash
python -m src.build_decltypes   --decltypes data/declaration_types.txt   --nodes data/processed/nodes.parquet   --out_dir data/processed
# -> data/processed/decltypes.parquet
```

Then featurize each declaration’s **type string**:

```bash
python -m src.tasks.build_type_features   --decltypes data/processed/decltypes.parquet   --nodes data/processed/nodes.parquet   --out data/processed/type_features.npz   --buckets 128
```

**Features include:**
- basic length & symbol counts,
- hashed **char trigrams**,
- hashed one‑hot of the **head symbol**.

Stored as `X` inside `type_features.npz` (shape: `N × D`).

---

## 3) Train the **text ranker** (fast, CPU)

Learns to place targets and premises near each other given positive edges from the graph.

```bash
python -m src.tasks.train_text_ranker   --features data/processed/type_features.npz   --contexts data/processed/contexts.jsonl   --out_ckpt outputs/text_ranker.pt   --emb_dim 64 --batch 512 --neg_per_pos 8 --epochs 1
```

> This uses a **shared MLP encoder** for both sides. Negatives are sampled per target.

---

## 4) Score **premise rankings** for all targets

```bash
python -m src.tasks.score_text_rankings   --features data/processed/type_features.npz   --contexts data/processed/contexts.jsonl   --nodes data/processed/nodes.parquet   --ckpt outputs/text_ranker.pt   --out data/processed/rankings.parquet   --topk 50 --batch 512 --chunk 128000
```

**Output:** `rankings.parquet` — for each target, the Top‑K candidate premises with scores.

---

## 5) Explain the rankings (human‑readable)

```bash
python -m src.tasks.explain_rankings   --rankings data/processed/rankings.parquet   --nodes data/processed/nodes.parquet   --contexts data/processed/contexts.jsonl   --out data/processed/rankings_explained.csv   --format csv --topk 50 --limit 500
```

This produces a CSV with names, hits vs. gold premises, recall@K, etc.

---

## 6) **Cold‑start**: rank premises for a brand‑new type

Give a new Lean type string; we embed it and score against all targets’ Top‑K thresholds.

```bash
python -m src.tasks.score_on_text   --features data/processed/type_features.npz   --nodes data/processed/nodes.parquet   --ckpt outputs/text_ranker.pt   --type_string "∀ {X : TopCat} (x : ↑X) (U : TopologicalSpace.OpenNhds (↑(CategoryTheory.CategoryStruct.id X) x)), (TopologicalSpace.OpenNhds.map (CategoryTheory.CategoryStruct.id X) x).obj U = U"   --out data/processed/new_target_rankings.parquet   --topk 50 --buckets 128
```

**Output:** `new_target_rankings.parquet` — Top‑K premises for your new statement.

---

## 7) **Productivity** of a new statement (adoption@K + lift)

Estimate how broadly useful a brand‑new statement would be, *if tomorrow looked like today*.
We say a target “adopts” the statement if the new premise would have landed in its Top‑K.

```bash
python -m src.tasks.score_productivity   --type_string "∀ {X : TopCat} (x : ↑X) ..."   --features data/processed/type_features.npz   --contexts data/processed/contexts.jsonl   --rankings data/processed/rankings.parquet   --nodes data/processed/nodes.parquet   --decltypes data/processed/decltypes.parquet   --ckpt outputs/text_ranker.pt   --buckets 128 --k_list 10,20,50   --show_hits 50 --hits_limit 25   --target_kinds theorem,lemma   --target_prefixes "TopologicalSpace."
```

You’ll see lines like:
```
adoption@10: 0.020%  (~39/N);  random=0.015%  lift=1.34
```
- **adoption@K:** fraction of targets that would have selected the new premise (Top‑K cutoff).  
- **random:** baseline `K / N` (with `N` candidates).  
- **lift = adoption@K / (K/N):** use this to interpret:  
  ~1.0 random; **2–3×** promising; **5×+** broadly useful.  
- `--show_hits K` prints a few representative targets that adopt the premise.

> This is *model‑based* productivity. For a truly future‑proof evaluation, train on an older snapshot and measure against a newer one (temporal split).

---

# (Optional) Principled **Graph‑based** productivity

If you also want a graph‑theoretic notion of productivity (“enables many things, directly and transitively, but is cheap”), use the add‑on scripts.

### 8) Build the **premise → target** graph & analytic scores

```bash
python -m src.tasks.build_graph   --contexts data/processed/contexts.jsonl   --nodes data/processed/nodes.parquet   --out_edges data/processed/edge_index.npz   --out_metrics data/processed/graph_metrics.parquet   --katz_k 8 --katz_beta 0.2   --reach_h 3   --alpha 0.2 --gamma 0.5
```

This writes:
- `edge_index.npz` — arrays `src, dst`
- `graph_metrics.parquet` — columns:
  - degrees: `in_deg, out_deg`
  - discounted reach: `katz_k8_b0.2`, finite‑hop `reach_h3`, `pagerank_d0.85`
  - **analytic productivity** (Katz):  
    
\(\text{prod} = \log\!\left(1 + \frac{\text{out\_deg} + \alpha\cdot\text{Katz}}{(1+\text{in\_deg})^{\gamma}}\right)\)

  - (and a reach‑based variant)

### 9) (Optional) Train a **GraphSAGE** regressor

```bash
python -m src.tasks.train_gnn_productivity   --edge_index data/processed/edge_index.npz   --graph_metrics data/processed/graph_metrics.parquet   --type_features data/processed/type_features.npz   --out_ckpt outputs/gnn_prod.pt   --label_col prod_katz_a0.2_b0.2_g0.5   --epochs 3 --batch_nodes 65536 --fanout 15,15 --device cpu
```

### 10) Score **all nodes** with the GNN

```bash
python -m src.tasks.score_gnn_productivity   --edge_index data/processed/edge_index.npz   --graph_metrics data/processed/graph_metrics.parquet   --type_features data/processed/type_features.npz   --nodes data/processed/nodes.parquet   --ckpt outputs/gnn_prod.pt   --out data/processed/gnn_productivity.parquet   --fanout 15,15 --batch_nodes 65536 --device cpu
```

**Output:** `gnn_productivity.parquet` with `y_pred` per declaration (graph‑aware productivity).

---

## (Optional) Attach productivity to premise ranking

**A) Re‑weight premise rankings** with a productivity prior

```bash
python -m src.tasks.reweight_rankings   --rankings_in data/processed/rankings.parquet   --prior_parquet data/processed/graph_metrics.parquet   --prior_score_col prod_katz_a0.2_b0.2_g0.5   --out data/processed/rankings_reweighted.parquet   --alpha 0.2 --mix linear --zscore_prior
```

Explain after reweighting:
```bash
python -m src.tasks.explain_rankings   --rankings data/processed/rankings_reweighted.parquet   --nodes data/processed/nodes.parquet   --contexts data/processed/contexts.jsonl   --out data/processed/rankings_reweighted_explained.csv   --format csv --topk 50 --limit 500
```

**B) Cold‑start productivity (text‑only) via distillation**

```bash
python -m src.tasks.distill_gnn_to_text   --type_features data/processed/type_features.npz   --labels_parquet data/processed/graph_metrics.parquet   --label_col prod_katz_a0.2_b0.2_g0.5   --out_ckpt outputs/text_prod.pt   --epochs 3 --batch 8192 --lr 2e-3 --hidden 256 --device cpu

python -m src.tasks.score_text_prod_on_text   --ckpt outputs/text_prod.pt   --type_string "∀ {X : TopCat} (x : ↑X) ..."   --buckets 128
# -> prints: predicted_productivity (text-only) = ...
```

---

## Interpreting outputs (cheat‑sheet)

- **`rankings.parquet`**: Top‑K candidate premises per target (IDs + scores).  
  Use `explain_rankings` to view names and recall@K.

- **Cold‑start** (`score_on_text`): Top‑K premises for a brand‑new statement.

- **Productivity** (`score_productivity`):  
  - **adoption@K** — fraction of targets that would select the new premise.  
  - **lift** — how much better than random (key number).  
  - Use `--target_kinds`/`--target_prefixes` to scope domains and avoid noise.

- **Graph productivity**: `graph_metrics.parquet` (analytic) and `gnn_productivity.parquet` (learned).  
  You can **reweight** premise rankings with either one.

---

## Troubleshooting

- **Large inputs**: All steps stream; if you hit memory limits, lower `--batch`/`--chunk` in scoring.
- **OMP duplicate lib (FAISS)**: If you use FAISS elsewhere and hit `KMP_DUPLICATE_LIB_OK` errors, set:
  ```bash
  export KMP_DUPLICATE_LIB_OK=TRUE
  ```
- **Mac GPU (MPS)**: Text ranker is CPU‑fast. For GNN, keep all tensors & model on the same device.
- **PyG NeighborLoader quirks**: Our scripts rely on PyG’s invariant that **the first `batch_size` nodes are the seeds**; this avoids `input_id` differences across versions.

---

## Rationale (why this is coherent)

- **Premise ranking** teaches an embedding that captures *what goes with what* by supervision from the dependency graph.  
- **Cold‑start ranking** uses the same encoder on a new type string to suggest its most plausible premises.  
- **Productivity (adoption@K)** asks: *how often would a future target pick this as a premise?* — using today’s Top‑K thresholds as a proxy. Report **lift**.  
- **Graph‑based productivity** gives a principled, transparent baseline (direct + discounted transitive reach, penalized by cost), and a GNN adds message‑passing structure.
