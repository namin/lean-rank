# Design of `lean-rank`

A high-level map of the system: what each component does, the data it reads/writes, and how the pieces fit. This is meant to be read top-to-bottom once; after that, the per-command walkthroughs will make more intuitive sense.

---

## 1) Goal

Learn from Mathlib’s existing **premise → target** dependencies to:
1. **Rank useful premises** for each target (retrieval).
2. **Cold-start**: rank premises for a new statement given **only its type string**.
3. **Productivity** of a new statement: estimate how often future targets would adopt it (**adoption@K + lift**).
4. *(Optional)* Provide a **graph-theoretic productivity** signal (Katz/Reach/PageRank) and a **GNN** to learn a stronger prior; let you **reweight** rankings and **distill** a text-only productivity model.
5. Provide a **What‑if Graph Productivity** score that simulates inserting a new statement into the dependency graph and measures its **direct + transitive** influence.

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

What‑if Graph Productivity
      └─► [score_graph_whatif] → whatif.json / whatif.md
            └─ Simulate adding a new statement → attach to top‑L likely adopters →
               compute truncated Katz mass + use‑cost penalty → one scalar “productivity-like” score
```

---

## 3) Data artifacts & schemas

- **`nodes.parquet`** — the canonical ID/name mapping  
  `id : int` (0..N-1), `name : str`

- **`contexts.jsonl`** — one JSON per target with its positive premises  
  `{ "target_id": int, "positives": [int, ...] }`

- **`decltypes.parquet`**  
  `id : int`, `name : str`, `kind : str` (theorem/lemma/definition/…), `type : str` (multi-line Lean type, normalized)

- **`type_features.npz`**  
  `X : float32 [N × D]` — features per declaration (base stats, hashed **char trigrams**, hashed **head symbol**)

- **`rankings.parquet`**  
  One row per target with `candidate_ids : list[int]`, `scores : list[float]` (Top-K; K is configurable)

- **`rankings_explained.csv`**  
  Human-readable names, hits vs. gold premises, recall@K per target.

- *(Optional)* **Graph files**  
  `edge_index.npz` — `src, dst : int64[]` for edges (premise→target)  
  `graph_metrics.parquet` — `in_deg, out_deg, katz_k*, reach_h*, pagerank_d*, prod_*`  
  `gnn_productivity.parquet` — `id, name, y_pred` (graph-aware productivity)

- **What‑if outputs**  
  `whatif.json` — structured details for a single run (attachments, per‑hop mass, productivity, percentile)  
  `whatif.md` — a Markdown report for sharing (produced by `--report_md` or implied by `--out_json`)

---

## 4) Components

### A) Dataset builders

**`src/build_dataset.py`**  
Parses `premises.txt` and writes:
- `nodes.parquet` — all declarations referenced
- `contexts.jsonl` — for each target, the list of premise IDs used in its proof/definition

Streaming: 8M+ line files are fine. *Note:* We currently treat any dependency line as a positive. Filtering by markers (`*` explicit argument, `s` simplifier) can be added if you want that distinction.

**`src/build_decltypes.py`**  
Parses `declaration_types.txt` → `decltypes.parquet`; aligns by name with `nodes.parquet` (unseen names dropped).

**`src/tasks/build_type_features.py`**  
Converts types into a numeric `X` matrix (`npz`). Hash sizes (`--buckets`) must be consistent with later text-only steps.

---

### B) Text ranker (core model)

- **Model:** shared MLP encoder for both targets and premises. Given features `x`, we produce an embedding `e = MLP(x)`; the score is a dot product: `s(t, p) = ⟨e_t, e_p⟩`.
- **Training:** for each target, contrast positive premises vs. sampled negatives (`--neg_per_pos`). Loss is a standard “one positive among negatives” softmax (InfoNCE-like) over dot-products. It’s CPU-fast.

**Files:**  
`src/tasks/train_text_ranker.py`, `src/tasks/score_text_rankings.py` (embeds everything, retrieves Top-K per target; chunked), `src/tasks/explain_rankings.py`

---

### C) Cold-start ranking

**`src/tasks/score_on_text.py`**  
Featurize a **new type string** → embed with the same MLP → score against all premises → return Top-K.  
Output: `new_target_rankings.parquet` (same schema as `rankings.parquet`, but for a single “virtual target”).

---

### D) Productivity of a new statement (model-based adoption@K)

**`src/tasks/score_productivity.py`**  
- Re-embeds the **new statement** on the **premise side**.  
- For each target, compare its score `s(new, target)` to that target’s **Top-K cutoff** (the K‑th best score in `rankings.parquet`). If ≥ cutoff, we say **the target adopts the new statement**.  
- Aggregate across a filtered target set (e.g., `--target_kinds theorem,lemma` and `--target_prefixes TopologicalSpace.`).

Reports:
- `adoption@K = (# adopters) / (# targets considered)`
- `random = K / M` where `M` is the total candidate premises
- `lift = adoption@K / random`  
Optionally prints a few **hits** with score, cutoff, and margin.

> **Why this is reasonable.** It answers: “If tomorrow looked like today, how often would this be Top‑K for real targets?” A more rigorous eval is to train on an earlier snapshot and evaluate on a newer one; the code is time-agnostic, so you can do this with two dumps.

---

### E) Graph productivity (optional)

**`src/tasks/build_graph.py`**  
Builds `edge_index` from `contexts.jsonl` and computes analytic signals:
- degrees (`in/out`), **truncated Katz**, **finite-hop reach**, **PageRank**
- **analytic productivity** (Katz variant):  
  \[\displaystyle
    \text{prod} = \log\!\left(1 + \frac{\text{out\_deg} + \alpha \cdot \text{Katz}}{(1 + \text{in\_deg})^{\gamma}}\right)
  \]
This captures “enables many things (directly + transitively) but is cheap to use.”

**`src/tasks/train_gnn_productivity.py`**, **`src/tasks/score_gnn_productivity.py`**  
A small **GraphSAGE** regressor learns to predict analytic productivity (or any label column). It uses neighbor sampling and can concatenate `type_features` and `graph_metrics` as inputs.

Why use a GNN?
- Better prior on existing nodes (multi-hop message passing smooths noise).
- Can serve as a **teacher** for a **text-only** cold-start model via distillation.

---

### F) Attaching productivity to ranking

**`src/tasks/reweight_rankings.py`**  
Mix a **productivity prior** (analytic or GNN `y_pred`) into `rankings.parquet`:
- `linear` (recommended): `s' = s + α · prior_z`
- `mult`: `s' = s · (1 + α · prior_z)`

Re-run `explain_rankings` on the reweighted file to see metric deltas.

**`src/tasks/distill_gnn_to_text.py`**, **`src/tasks/score_text_prod_on_text.py`**  
Train a tiny MLP on `type_features` to match a graph productivity label (analytic or GNN `y_pred`). The resulting `outputs/text_prod.pt` gives a **cold-start productivity score** for any new type string.

---

### G) What‑if Graph Productivity (Personalized Katz)

**File:** `src/tasks/score_graph_whatif.py`

**Goal.** Estimate how productive a **new** statement would be **if it existed in Mathlib**, accounting for **direct and transitive** effects (Melanie’s criterion).

**Idea.** We *simulate* inserting a virtual node for the new statement and attach it to the **top‑L targets** most likely to adopt it (ranked by text similarity or by your learned encoder). From those attachments we measure discounted multi‑hop influence with a **truncated Katz** walk:

\[\displaystyle
\text{KatzMass} \;=\; \sum_{k=1}^K \beta^k \cdot \big\lVert A^k \, \mathbf{s}\big\rVert_1
\quad\text{where}\quad \mathbf{s}=\text{one‑hot mass on the L attached targets,}
\]

and convert to a single “productivity‑like” score with a simple **use‑cost** penalty (fewer hypotheses ⇒ easier to reuse):

\[\displaystyle
\boxed{\;\textbf{prod\_whatif}\;=\;\log\!\Big(1+\frac{L + \alpha\cdot\text{KatzMass}}{(1+\text{use\_cost})^\gamma}\Big)\;}
\]

- \(L\): number of target attachments (how many places would likely adopt it right away).  
- \(K\): walk truncation (how far influence can propagate).  
- \(\beta\): hop decay (down‑weights long chains).  
- \(\alpha\): how much to trust transitive mass vs direct attachments.  
- \(\gamma\): strength of the use‑cost penalty.  
- **use_cost**: rough proxy from the type string (counts of `→` and `∀` for now; swap in a richer parser later).

This is a **personalized** centrality score: it’s computed **relative to your statement** and the region of the graph it would connect to.

**CLI**

```bash
python -m src.tasks.score_graph_whatif \
  --edge_index data/processed/edge_index.npz \
  --nodes data/processed/nodes.parquet \
  --decltypes data/processed/decltypes.parquet \
  --type_features data/processed/type_features.npz \
  --type_string "∀ {X : TopCat} (x : ↑X) (U : ...)" \
  --target_kinds theorem,lemma \
  --target_prefixes TopologicalSpace. \
  --L 50 --K 6 --beta 0.2 --alpha 0.2 --gamma 0.5 \
  --graph_metrics data/processed/graph_metrics.parquet \
  --out_json data/processed/whatif.json \
  --report_md data/processed/whatif.md
```

Add `--ckpt outputs/text_ranker.pt --use_model` to pick top‑L targets using the **learned encoder** instead of raw cosine on features.

**Inputs & alignment**

- `edge_index.npz` — directed edges `premise -> target`.  
- `nodes.parquet` — must align 1:1 with feature rows; column `id` preferred; if missing, row order is treated as id.  
- `decltypes.parquet` — provides `kind` for filtering; merged by `id` if available, else by `name`.  
- `type_features.npz` — feature matrix `X` for all nodes (same order/ids as `nodes.parquet`).  
- `graph_metrics.parquet` (optional) — lets the script report a **percentile** vs an analytic column like `prod_katz_a0.2_b0.2_g0.5`.

**Outputs**

- **Console summary**: attached targets (name + similarity), per‑hop mass, total Katz mass, use‑cost, `prod_whatif`, optional percentile.  
- **JSON** (`--out_json`) with full details (attachments, per‑hop curve, `prod_whatif`, percentile).  
- **Markdown** (`--report_md` or implied from `--out_json`) for easy sharing.

**How it differs from *adoption@K***

- **adoption@K** (existing pipeline): “Across all *current* targets, how often would my statement have made the Top‑K list?” — pointwise thresholding per target, then averaged.  
- **what‑if prod** (this section): “If I *actually add it* and connect it to top‑L likely adopters, **how much influence propagates** through Mathlib?” — a **graph‑aware**, transitive score in one scalar.

They complement each other: adoption@K is a *ranking* sanity check; `prod_whatif` is a localized **personalized centrality**.

**Parameter guide (defaults in parentheses)**

- `L (50)`: Larger L models “more immediate uptake.” Start with 10–100; too large just adds noise.  
- `K (6)`: 4–8 is typical; higher K increases runtime.  
- `β (0.2)`: Smaller means strong locality; 0.1–0.3 tends to behave well.  
- `α (0.2)`: Balance between direct (`L`) and transitive (`KatzMass`).  
- `γ (0.5)`: Heaviness of the use‑cost penalty. Try 0.3–1.0.

**Complexity & performance**

Frontier expansion uses CSR‑like arrays and coalesces duplicates. Complexity is roughly:
\[\displaystyle
\mathcal{O}\Big(\sum_{k=1}^{K}\, \sum_{u \in \text{frontier}_k}\deg(u)\Big)
\]
It only touches the **small neighborhood** around your L attachments (fast even on large graphs).

**Runner integration**

Enable a one‑shot report via the runner:

```bash
DO_WHATIF=1 WHATIF_USE_MODEL=1 ./run_walkthrough_v4.sh
# writes: data/processed/whatif.json and data/processed/whatif.md
```
If `edge_index.npz` is missing, the runner builds it automatically.

**Interpretation checklist**

- **High `prod_whatif` + low use‑cost** → broadly applicable, likely to be reused.  
- **High `prod_whatif` but high use‑cost** → influential but specialized; consider refactoring into lemmas with fewer hypotheses.  
- **Percentile vs graph metrics** → situate your statement against Mathlib’s distribution (“top 10% among existing lemmas” is a good sign).  
- **Per‑hop mass curve** → if it decays too quickly, influence is very local; consider different target domain or a different formulation.

**Edge cases & robustness**

- If `decltypes.parquet` lacks `id`, the script merges by `name`. If `kind` is missing, the kind filter is ignored with a warning.  
- If domain filters remove all candidates, the script exits with a friendly message.  
- The current **use‑cost** is intentionally simple (counts `→`, `∀`). You can later replace it with a richer parser (binder counts, explicit vs implicit args, head‑symbol class, etc.) without touching the graph logic.

**Future upgrades**

- **Use‑cost 2.0**: move from glyph counts to proper hypothesis structure features (outer `→` count, ∀/∃ counts, head symbol class, typeclass constraints).  
- **Bi‑directional what‑if**: also add `premise → new` edges to simulate proof dependencies if you have a predicted proof skeleton.  
- **Temporal validation**: time‑split Mathlib (T₀→T₁) and compare `prod_whatif` vs real adoption growth for new T₁ statements.  
- **Teacher → student**: learn a small text model to predict `prod_whatif` directly from the type string (distillation), for instant cold‑start productivity without any graph access.

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
- *What‑if productivity*: `score_graph_whatif` to get a personalized, graph‑aware scalar for a new statement.

---

## 6) Interpretation & trade-offs

- **Text ranker**
  - + Fast, simple, cold-start ready.
  - – Ignores explicit graph structure beyond what types imply.

- **Model-based productivity (adoption@K)**
  - + Directly tied to “would real targets take this as a premise?”
  - – Uses *today’s* landscape as a proxy for *tomorrow* (consider temporal eval for research-grade claims).

- **Graph productivity (analytic + GNN)**
  - + Transparent (analytic) and powerful (GNN).
  - + Works as a **prior** to bias premise selection and as a **teacher** for text-only cold-start productivity.
  - – GNN doesn’t help truly cold-start **by itself** (no edges), unless you do *what‑if insertion* (attach the new node to its top‑L predicted neighbors and run the GNN).

- **What‑if productivity**
  - + Gives a single, interpretable scalar for a new statement’s *graph‑aware* potential impact.
  - + Localized and fast: only explores the neighborhood around predicted attachments.
  - – Still depends on a quality **target‑attachment** heuristic (text similarity or learned encoder).

---

## 7) Extensibility

- **Type features**: replace char-3grams with a Lean-aware tokenizer; add symbol embeddings; subword BPE.
- **Losses**: experiment with margin/Triplet/BPR vs. InfoNCE.
- **Temporal splits**: train on snapshot _t_, evaluate productivity on snapshot _t+Δ_.
- **Domain conditioning**: compute adoption baselines within namespaces (e.g., only `TopologicalSpace.*` candidates).
- **What‑if variants**: richer use‑cost; attach both as premise and as target; GNN propagation on the augmented neighborhood.

---

## 8) Common pitfalls

- **Mismatched `--buckets`**: keep it consistent between `build_type_features` and any text‑only scoring.
- **Interpreter mismatch**: ensure the same Python/venv is used for all steps (see the runner).
- **FAISS / OpenMP** (if you enable FAISS): set `export KMP_DUPLICATE_LIB_OK=TRUE` on macOS if you see the duplicate lib error.
- **PyG NeighborLoader** quirks: our GNN scripts rely on the invariant that the first `batch_size` nodes in a batch are seeds (robust across versions).
- **What‑if scoping**: forgetting `--target_prefixes` can attach to the wrong domain and inflate mass in irrelevant areas.

---

## 9) TL;DR

- Train a **text ranker** once → get premise rankings for all targets.
- For a **new statement**, you can:
  - **Rank** its likely premises (cold-start).
  - Estimate **productivity** (adoption@K + lift).
  - Simulate **graph‑aware productivity** via What‑if (personalized Katz + use‑cost).
- Optionally compute **graph productivity** to:
  - **Reweight** rankings (better global prior).
  - **Distill** to a **text-only** productivity model for cold‑start scoring.
