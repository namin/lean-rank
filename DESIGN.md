# Design of `lean-rank`

A high-level map of the system: what each component does, the data it reads/writes, and how the pieces fit. This is meant to be read top-to-bottom once; after that, the per-command walkthroughs will make more intuitive sense.

---

## 1) Goal

Learn from Mathlib’s existing **premise → target** dependencies to:
1. **Rank useful premises** for each target (retrieval).
2. **Cold-start**: rank premises for a brand-new statement given **only its type string**.
3. **Productivity** of a new statement: estimate how often future targets would adopt it (adoption@K + lift).
4. *(Optional)* Provide a **graph-theoretic productivity** signal (Katz/Reach/PageRank) and a **GNN** to learn a stronger prior; let you **reweight** rankings and **distill** a text-only productivity model.

Everything is streaming / CPU-friendly by default. GNN is optional.

---

## 2) Bird’s‑eye view

```
kim-em/lean-training-data
   ├─ premises.txt                # declaration dependencies
   └─ declaration_types.txt       # name + kind + type for every decl
          │
          ▼
[build_dataset]  ─────────┐
  nodes.parquet           │    contexts.jsonl
  (id, name)              │    (target_id, positives[])
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
[build_decltypes]                  [build_type_features]
  decltypes.parquet                 type_features.npz (X)
  (id, name, kind, type)           └─ basic stats + char-3grams + head symbol hash
          │
          ▼
[train_text_ranker]  (shared MLP encoder for type features)
  outputs/text_ranker.pt
          │
          ├─────────────► [score_text_rankings] → rankings.parquet
          │                        └─ Top-K candidate premises per target
          │
          ├─────────────► [score_on_text] → new_target_rankings.parquet
          │                        └─ Cold-start premise ranking for a new type string
          │
          └─────────────► [score_productivity] → prints adoption@K + lift (+ hits)
                                   └─ “Would the new statement be adopted by today’s targets?”
```

**Optional graph add-on**

```
[build_graph] → edge_index.npz + graph_metrics.parquet (Katz/Reach/PR + analytic productivity)
      │
      ├─► [train_gnn_productivity] + [score_gnn_productivity] → gnn_productivity.parquet (y_pred)
      │
      ├─► [reweight_rankings]  (use analytic or GNN prior) → rankings_reweighted.parquet
      │
      └─► [distill_gnn_to_text] → outputs/text_prod.pt
            └─► [score_text_prod_on_text]  (cold-start productivity from type string)
```

---

## 3) Data artifacts & schemas

- **`nodes.parquet`** — the canonical ID/name mapping
  - `id : int` (0..N-1), `name : str`
- **`contexts.jsonl`** — one JSON per target with its positive premises
  - `{ "target_id": int, "positives": [int, ...] }`
- **`decltypes.parquet`**
  - `id : int`, `name : str`, `kind : str` (theorem/lemma/definition/…),
  - `type : str` (multi-line Lean type, normalized to a single string)
- **`type_features.npz`**
  - `X : float32 [N × D]` — features per declaration
    - base statistics (lengths, symbol counts)
    - hashed **char trigrams**
    - hashed one-hot of the **head symbol** (e.g., `Iff`, `Eq`, …)
- **`rankings.parquet`**
  - One row per target with `candidate_ids : list[int]`, `scores : list[float]` (Top-K; K is configurable)
- **`rankings_explained.csv`**
  - Human-readable names, hits vs. gold premises, recall@K per target.
- *(Optional)* **Graph files**
  - `edge_index.npz` — `src, dst : int64[]` for edges (premise→target)
  - `graph_metrics.parquet` — `in_deg, out_deg, katz_k*, reach_h*, pagerank_d*, prod_*`
  - `gnn_productivity.parquet` — `id, name, y_pred` (graph-aware productivity)

---

## 4) Components

### A) Dataset builders

- **`src/build_dataset.py`**  
  Parses `premises.txt` and writes:
  - `nodes.parquet` — all declarations referenced
  - `contexts.jsonl` — for each target, the list of premise IDs used in its proof/definition
  - The builder is streaming, so 8M+ line files are fine.
  - *Note:* We currently treat any dependency line as a positive. Filtering by markers (`*` explicit argument, `s` simplifier) can be added if you want that distinction.

- **`src/build_decltypes.py`**  
  Parses `declaration_types.txt` → `decltypes.parquet`; aligns by name with `nodes.parquet` (unseen names are dropped, optional stats logged).

- **`src/tasks/build_type_features.py`**  
  Converts types into a numeric `X` matrix (`npz`). Hash sizes (`--buckets`) must be consistent with later text-only steps.

### B) Text ranker (core model)

- **Model:** shared MLP encoder for both targets and premises. Given features `x`, we produce an embedding `e = MLP(x)`; the score is a dot product:  
  `s(t, p) = ⟨e_t, e_p⟩`.
- **Training:** for each target, contrast positive premises vs. sampled negatives (`--neg_per_pos`). Loss is a standard “one positive among negatives” softmax (InfoNCE-like) over dot-products. It’s CPU-fast.
- **Files:**
  - `src/tasks/train_text_ranker.py`
  - `src/tasks/score_text_rankings.py` (embeds everything, retrieves Top-K per target; chunked to bound memory)
  - `src/tasks/explain_rankings.py`

### C) Cold-start ranking

- **`src/tasks/score_on_text.py`**  
  Featurize a **new type string** → embed with the same MLP → score against all premises → return Top-K.
  - Output: `new_target_rankings.parquet` (same schema as `rankings.parquet`, but for a single “virtual target”).

### D) Productivity of a new statement (model-based adoption@K)

- **`src/tasks/score_productivity.py`**
  - Re-embeds the **new statement** on the **premise side**.
  - For each target, compare its score `s(new, target)` to that target’s **Top-K cutoff** (the K‑th best score in `rankings.parquet`). If ≥ cutoff, we say **the target adopts the new statement**.
  - Aggregate across a filtered set of targets (e.g., `--target_kinds theorem,lemma` and `--target_prefixes TopologicalSpace.`).
  - Report:
    - `adoption@K = (# adopters) / (# targets considered)`
    - `random = K / M` where `M` is the global number of candidate premises
    - `lift = adoption@K / random`
  - Optionally print a few **hits** with score, cutoff, and margin.

> Why this is reasonable: it answers “If tomorrow looked like today, how often would this be Top‑K for real targets?” A cleaner eval is to train on an earlier snapshot and evaluate on a newer one; the code is already time-agnostic, so you can do that with two dumps.

### E) Graph productivity (optional)

- **`src/tasks/build_graph.py`**  
  Builds `edge_index` from `contexts.jsonl` and computes analytic signals:
  - degrees (`in/out`), **truncated Katz**, **finite-hop reach**, **PageRank**
  - **analytic productivity** (Katz variant):  
    \[
    \text{prod} = \log\!\left(1 + \frac{\text{out\_deg} + \alpha \cdot \text{Katz}}{(1 + \text{in\_deg})^{\gamma}}\right)
    \]
  - This captures “enables many things (directly + transitively) but is cheap to use.”

- **`src/tasks/train_gnn_productivity.py`**, **`src/tasks/score_gnn_productivity.py`**  
  A small **GraphSAGE** regressor learns to predict the analytic productivity (or any label column). It uses neighbor sampling and supports optional concatenation of `type_features` to `graph_metrics` as node inputs.

  Why use a GNN?
  - Better prior on existing nodes (multi-hop message passing smooths noise).
  - Can serve as a **teacher** for a **text-only** cold-start model via distillation.

### F) Attaching productivity to ranking

- **`src/tasks/reweight_rankings.py`**  
  Mix a **productivity prior** (analytic or GNN `y_pred`) into `rankings.parquet`:
  - `linear` (recommended): `s' = s + α · prior_z`
  - `mult`: `s' = s · (1 + α · prior_z)`
  - Re-run `explain_rankings` on the reweighted file to see metric deltas.

- **`src/tasks/distill_gnn_to_text.py`**, **`src/tasks/score_text_prod_on_text.py`**  
  Train a tiny MLP on `type_features` to match a graph productivity label (analytic or GNN `y_pred`). The resulting `outputs/text_prod.pt` gives a **cold-start productivity score** for any new type string.

---

## 5) How the pieces interact (sequence)

**Training & indexing**
1. `build_dataset` → `contexts.jsonl`, `nodes.parquet`
2. `build_decltypes` → `decltypes.parquet`
3. `build_type_features` → `type_features.npz`
4. `train_text_ranker` → `text_ranker.pt`
5. `score_text_rankings` → `rankings.parquet` (Target→Top‑K Premises)

**Using the system**
- *Cold-start ranking*: `score_on_text` using `text_ranker.pt` and `type_features.npz` (no graph needed).
- *Productivity*: `score_productivity` uses **only** the MLP encoder results and `rankings.parquet`’s per-target cutoffs.
- *Optional graph*: `build_graph` (+ `train_gnn_productivity`/`score_gnn_productivity`) → reweight rankings or distill to text.

---

## 6) Interpretation & trade-offs

- **Text ranker**:
  - + Fast, simple, cold-start ready.
  - – Ignores explicit graph structure beyond what types imply.

- **Model-based productivity (adoption@K)**:
  - + Directly tied to “would real targets take this as a premise?”
  - – Uses *today’s* landscape as a proxy for *tomorrow* (consider temporal eval for research-grade claims).

- **Graph productivity (analytic + GNN)**:
  - + Transparent (analytic) and powerful (GNN).
  - + Works as a **prior** to bias premise selection and as a **teacher** for text-only cold-start productivity.
  - – GNN doesn’t help truly cold-start **by itself** (no edges), unless you do *what‑if insertion* (attach the new node to its top‑L predicted neighbors and run the GNN).

---

## 7) Extensibility

- **Type features**: replace char-3grams with a Lean-aware tokenizer; add symbol embeddings; subword BPE.
- **Losses**: experiment with margin/Triplet/BPR vs. InfoNCE.
- **Temporal splits**: train on snapshot _t_, evaluate productivity on snapshot _t+Δ_.
- **Domain conditioning**: compute adoption baselines within namespaces (e.g., only `TopologicalSpace.*` candidates).

---

## 8) Common pitfalls

- **Mismatched `--buckets`**: keep it consistent between `build_type_features` and any text‑only scoring.
- **Interpreter mismatch**: ensure the same Python/venv is used for all steps (use the v3 runner).
- **FAISS / OpenMP** (if you enable FAISS): set `export KMP_DUPLICATE_LIB_OK=TRUE` on macOS if you see the duplicate lib error.
- **PyG NeighborLoader** quirks: our GNN scripts rely on the invariant that the first `batch_size` nodes in a batch are seeds (robust across versions).

---

## 9) TL;DR

- Train a **text ranker** once → get premise rankings for all targets.
- For a **new statement**, you can:
  - **Rank** its likely premises (cold-start).
  - Estimate **productivity** (adoption@K + lift).
- Optionally compute **graph productivity** to:
  - **Reweight** rankings (better global prior).
  - **Distill** to a **text-only** productivity model for cold‑start scoring.
