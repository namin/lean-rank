#!/usr/bin/env bash
set -euo pipefail
set -f

# -----------------------------------------------------------------------------
# Lean Rank — End-to-End Walkthrough Runner
# -----------------------------------------------------------------------------

# ------------------------ Config (overridable via env) ------------------------
DATA_DIR="${DATA_DIR:-data}"
PROC_DIR="${PROC_DIR:-$DATA_DIR/processed}"
OUT_DIR="${OUT_DIR:-outputs}"

# Use the active interpreter by default (resolves venv/conda correctly)
PYTHON_BIN="${PYTHON_BIN:-$(python -c 'import sys; print(sys.executable)')}"

# Model / features
BUCKETS="${BUCKETS:-128}"
EMB_DIM="${EMB_DIM:-64}"
NEG_PER_POS="${NEG_PER_POS:-8}"
BATCH="${BATCH:-512}"
EPOCHS="${EPOCHS:-1}"
TOPK="${TOPK:-50}"
CHUNK="${CHUNK:-128000}"   # contexts chunk for scoring

# Productivity evaluation
K_LIST="${K_LIST:-10,20,50}"
TARGET_KINDS="${TARGET_KINDS:-theorem,lemma}"
TARGET_PREFIXES="${TARGET_PREFIXES:-TopologicalSpace.}"

# Optional graph pipeline toggles
DO_GRAPH="${DO_GRAPH:-0}"     # 1 to build graph + metrics + (optional) GNN
DO_ATTACH="${DO_ATTACH:-0}"   # 1 to reweight rankings and distill text productivity

# New: optional What-if productivity report (Markdown)
DO_WHATIF="${DO_WHATIF:-1}"
WHATIF_L="${WHATIF_L:-50}"
WHATIF_K="${WHATIF_K:-6}"
WHATIF_BETA="${WHATIF_BETA:-0.2}"
WHATIF_ALPHA="${WHATIF_ALPHA:-0.2}"
WHATIF_GAMMA="${WHATIF_GAMMA:-0.5}"
WHATIF_USE_MODEL="${WHATIF_USE_MODEL:-0}"  # 1 to use learned encoder (needs TEXT_CKPT)

# Force rebuilds
FORCE="${FORCE:-0}"

# New type string for cold-start + productivity (edit as needed)
TYPE_STRING="${TYPE_STRING:-∀ {X : TopCat} (x : ↑X) (U : TopologicalSpace.OpenNhds (↑(CategoryTheory.CategoryStruct.id X) x)), (TopologicalSpace.OpenNhds.map (CategoryTheory.CategoryStruct.id X) x).obj U = U}"

# ------------------------------- Sanity checks --------------------------------
need_file() { [[ -f "$1" ]] || { echo "ERROR: missing $1"; exit 1; }; }
say() { printf "==> %s\n" "$*"; }

mkdir -p "$PROC_DIR" "$OUT_DIR"

need_file "$DATA_DIR/premises.txt"
need_file "$DATA_DIR/declaration_types.txt"

say "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('  exe:', sys.executable)"
# Torch smoke test (informative only)
if "$PYTHON_BIN" -c "import torch"; then
  "$PYTHON_BIN" - <<'PY'
import torch, sys
print(f"  torch: {torch.__version__}  cuda={torch.cuda.is_available()}  mps={getattr(getattr(torch.backends,'mps',None),'is_available',lambda:False)()}")
PY
else
  echo "  torch: NOT FOUND in this interpreter (that's OK unless you use graph/GNN tasks)"
fi

# ------------------------------- Helper runner --------------------------------
run_step() {
  local label="$1"; shift
  say "$label"
  printf "    %q " "$@"
  printf "\n"
  "$@"
}

maybe_run() {
  # maybe_run "<label>" "<output_path_to_check>" <cmd...>
  local label="$1"; shift
  local out="$1"; shift
  if [[ "$FORCE" == "1" || ! -e "$out" ]]; then
    run_step "$label" "$@"
  else
    say "$label [skip] $out exists (use FORCE=1 to rebuild)"
  fi
}

# --------------------------------- Paths --------------------------------------
NODES="$PROC_DIR/nodes.parquet"
CONTEXTS="$PROC_DIR/contexts.jsonl"
META="$PROC_DIR/meta.json"
DECLTYPES_TXT="$DATA_DIR/declaration_types.txt"
DECLTYPES="$PROC_DIR/decltypes.parquet"
TYPE_FEATS="$PROC_DIR/type_features.npz"
TEXT_CKPT="$OUT_DIR/text_ranker.pt"
RANKINGS="$PROC_DIR/rankings.parquet"
RANKINGS_EXPLAINED="$PROC_DIR/rankings_explained.csv"
NEW_RANKINGS="$PROC_DIR/new_target_rankings.parquet"

# Graph outputs (optional)
EDGE_INDEX="$PROC_DIR/edge_index.npz"
GRAPH_METRICS="$PROC_DIR/graph_metrics.parquet"
GNN_CKPT="$OUT_DIR/gnn_prod.pt"
GNN_SCORES="$PROC_DIR/gnn_productivity.parquet"
RANKINGS_REWEIGHTED="$PROC_DIR/rankings_reweighted.parquet"
RANKINGS_REWEIGHTED_EXPLAINED="$PROC_DIR/rankings_reweighted_explained.csv"
TEXT_PROD_CKPT="$OUT_DIR/text_prod.pt"

# What-if outputs
WHATIF_JSON="$PROC_DIR/whatif.json"
WHATIF_MD="$PROC_DIR/whatif.md"

# --------------------------------- Steps --------------------------------------

# 1) Build dataset from premises
maybe_run "1) Build dataset" "$NODES" \
  "$PYTHON_BIN" -m src.build_dataset \
    --premises "$DATA_DIR/premises.txt" \
    --out_dir "$PROC_DIR"

# 2) Parse decl types & build features
maybe_run "2a) Parse decl types" "$DECLTYPES" \
  "$PYTHON_BIN" -m src.build_decltypes \
    --decltypes "$DECLTYPES_TXT" \
    --nodes "$NODES" \
    --out_dir "$PROC_DIR"

maybe_run "2b) Build type features" "$TYPE_FEATS" \
  "$PYTHON_BIN" -m src.tasks.build_type_features \
    --decltypes "$DECLTYPES" \
    --nodes "$NODES" \
    --out "$TYPE_FEATS" \
    --buckets "$BUCKETS"

# 3) Train text ranker
maybe_run "3) Train text ranker" "$TEXT_CKPT" \
  "$PYTHON_BIN" -m src.tasks.train_text_ranker \
    --features "$TYPE_FEATS" \
    --contexts "$CONTEXTS" \
    --out_ckpt "$TEXT_CKPT" \
    --emb_dim "$EMB_DIM" --batch "$BATCH" --neg_per_pos "$NEG_PER_POS" --epochs "$EPOCHS"

# 4) Score rankings
maybe_run "4) Score rankings (Top-$TOPK)" "$RANKINGS" \
  "$PYTHON_BIN" -m src.tasks.score_text_rankings \
    --features "$TYPE_FEATS" \
    --contexts "$CONTEXTS" \
    --nodes "$NODES" \
    --ckpt "$TEXT_CKPT" \
    --out "$RANKINGS" \
    --topk "$TOPK" --batch "$BATCH" --chunk "$CHUNK"

# 5) Explain rankings
maybe_run "5) Explain rankings (CSV)" "$RANKINGS_EXPLAINED" \
  "$PYTHON_BIN" -m src.tasks.explain_rankings \
    --rankings "$RANKINGS" \
    --nodes "$NODES" \
    --contexts "$CONTEXTS" \
    --out "$RANKINGS_EXPLAINED" \
    --format csv --topk "$TOPK" --limit 500

# 6) Cold-start ranking
maybe_run "6) Cold-start ranking for new statement" "$NEW_RANKINGS" \
  "$PYTHON_BIN" -m src.tasks.score_on_text \
    --features "$TYPE_FEATS" \
    --nodes "$NODES" \
    --ckpt "$TEXT_CKPT" \
    --type_string "$TYPE_STRING" \
    --out "$NEW_RANKINGS" \
    --topk "$TOPK" --buckets "$BUCKETS"

# 7) Productivity (adoption@K + lift)
run_step "7) Productivity for new statement" \
  "$PYTHON_BIN" -m src.tasks.score_productivity \
    --type_string "$TYPE_STRING" \
    --features "$TYPE_FEATS" \
    --contexts "$CONTEXTS" \
    --rankings "$RANKINGS" \
    --nodes "$NODES" \
    --decltypes "$DECLTYPES" \
    --ckpt "$TEXT_CKPT" \
    --buckets "$BUCKETS" --k_list "$K_LIST" \
    --show_hits "$TOPK" --hits_limit 25 \
    --target_kinds "$TARGET_KINDS" \
    --target_prefixes "$TARGET_PREFIXES"

# --------------------------- Optional: Graph pipeline --------------------------
if [[ "$DO_GRAPH" == "1" ]]; then
  say "---- Optional graph-based productivity ----"

  maybe_run "8) Build graph + analytic productivity" "$GRAPH_METRICS" \
    "$PYTHON_BIN" -m src.tasks.build_graph \
      --contexts "$CONTEXTS" \
      --nodes "$NODES" \
      --out_edges "$EDGE_INDEX" \
      --out_metrics "$GRAPH_METRICS" \
      --katz_k 8 --katz_beta 0.2 \
      --reach_h 3 \
      --alpha 0.2 --gamma 0.5

  maybe_run "9) Train GNN productivity (GraphSAGE)" "$GNN_CKPT" \
    "$PYTHON_BIN" -m src.tasks.train_gnn_productivity \
      --edge_index "$EDGE_INDEX" \
      --graph_metrics "$GRAPH_METRICS" \
      --type_features "$TYPE_FEATS" \
      --out_ckpt "$GNN_CKPT" \
      --label_col prod_katz_a0.2_b0.2_g0.5 \
      --epochs 3 --batch_nodes 65536 --fanout 15,15 --device cpu

  maybe_run "10) Score all nodes with GNN" "$GNN_SCORES" \
    "$PYTHON_BIN" -m src.tasks.score_gnn_productivity \
      --edge_index "$EDGE_INDEX" \
      --graph_metrics "$GRAPH_METRICS" \
      --type_features "$TYPE_FEATS" \
      --nodes "$NODES" \
      --ckpt "$GNN_CKPT" \
      --out "$GNN_SCORES" \
      --fanout 15,15 --batch_nodes 65536 --device cpu
fi

# --------------------------- Optional: What-if report --------------------------
if [[ "$DO_WHATIF" == "1" ]]; then
  say "---- What-if productivity (Markdown report) ----"
  # Ensure edge_index exists; if not, build minimal graph artifacts
  if [[ ! -f "$EDGE_INDEX" ]]; then
    say "edge_index not found; building graph artifacts needed for what-if"
    run_step "8*) Build graph edges for what-if" \
      "$PYTHON_BIN" -m src.tasks.build_graph \
        --contexts "$CONTEXTS" \
        --nodes "$NODES" \
        --out_edges "$EDGE_INDEX" \
        --out_metrics "$GRAPH_METRICS" \
        --katz_k 8 --katz_beta "$WHATIF_BETA" \
        --reach_h 3 \
        --alpha "$WHATIF_ALPHA" --gamma "$WHATIF_GAMMA"
  fi

  # Assemble optional use_model args
  WHATIF_ARGS=()
  if [[ "$WHATIF_USE_MODEL" == "1" ]]; then
    WHATIF_ARGS+=(--ckpt "$TEXT_CKPT" --use_model)
  fi
  # Graph metrics are optional (for percentile)
  if [[ -f "$GRAPH_METRICS" ]]; then
    WHATIF_ARGS+=(--graph_metrics "$GRAPH_METRICS")
  fi

  run_step "W) What-if report" \
    "$PYTHON_BIN" -m src.tasks.score_graph_whatif \
      --edge_index "$EDGE_INDEX" \
      --nodes "$NODES" \
      --decltypes "$DECLTYPES" \
      --type_features "$TYPE_FEATS" \
      --type_string "$TYPE_STRING" \
      --target_kinds "$TARGET_KINDS" \
      --target_prefixes "$TARGET_PREFIXES" \
      --L "$WHATIF_L" --K "$WHATIF_K" --beta "$WHATIF_BETA" --alpha "$WHATIF_ALPHA" --gamma "$WHATIF_GAMMA" \
      --out_json "$WHATIF_JSON" \
      --report_md "$WHATIF_MD" \
      "${WHATIF_ARGS[@]}"
fi

# --------------------------- Optional: Attach signals --------------------------
if [[ "$DO_ATTACH" == "1" ]]; then
  say "---- Optional attachment of productivity signals ----"

  maybe_run "A) Reweight rankings with analytic prod (Katz)" "$RANKINGS_REWEIGHTED" \
    "$PYTHON_BIN" -m src.tasks.reweight_rankings \
      --rankings_in "$RANKINGS" \
      --prior_parquet "$GRAPH_METRICS" \
      --prior_score_col prod_katz_a0.2_b0.2_g0.5 \
      --out "$RANKINGS_REWEIGHTED" \
      --alpha 0.2 --mix linear --zscore_prior

  maybe_run "A2) Explain reweighted rankings" "$RANKINGS_REWEIGHTED_EXPLAINED" \
    "$PYTHON_BIN" -m src.tasks.explain_rankings \
      --rankings "$RANKINGS_REWEIGHTED" \
      --nodes "$NODES" \
      --contexts "$CONTEXTS" \
      --out "$RANKINGS_REWEIGHTED_EXPLAINED" \
      --format csv --topk "$TOPK" --limit 500

  maybe_run "B) Distill analytic prod to text-only" "$TEXT_PROD_CKPT" \
    "$PYTHON_BIN" -m src.tasks.distill_gnn_to_text \
      --type_features "$TYPE_FEATS" \
      --labels_parquet "$GRAPH_METRICS" \
      --label_col prod_katz_a0.2_b0.2_g0.5 \
      --out_ckpt "$TEXT_PROD_CKPT" \
      --epochs 3 --batch 8192 --lr 2e-3 --hidden 256 --device cpu
fi

echo ""
echo "All done."
