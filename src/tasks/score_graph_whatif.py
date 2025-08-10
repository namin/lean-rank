from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

try:
    import torch  # optional, only if --use_model
except Exception:
    torch = None

# Reuse your existing featurizer
try:
    from src.utils.type_features import featurize_type
except Exception as e:
    print("[whatif] ERROR: could not import featurize_type from src.utils.type_features", file=sys.stderr)
    raise

def _load_edge_index(edge_path: Path, num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ed = np.load(edge_path)
    src = ed["src"].astype(np.int64)
    dst = ed["dst"].astype(np.int64)
    order = np.argsort(src, kind="mergesort")
    src = src[order]; dst = dst[order]
    out_deg = np.bincount(src, minlength=num_nodes)
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(out_deg, out=row_ptr[1:])
    col_idx = dst
    return row_ptr, col_idx, out_deg

def _count_use_cost(type_string: str) -> int:
    # Simple "use-cost": count implication arrows and foralls (rough hypothesis proxy)
    return type_string.count("→") + type_string.count("∀")

def _featurize_new_type(type_string: str, buckets: int) -> np.ndarray:
    base, tri, head = featurize_type(type_string, buckets=buckets)
    return np.concatenate([base, tri, head]).astype(np.float32)

def _build_target_mask(nodes_df: pd.DataFrame,
                       decl_df: pd.DataFrame,
                       kinds: List[str],
                       prefixes: List[str]) -> np.ndarray:
    # Merge kinds onto nodes by id if available, else by name
    merged = nodes_df.copy()
    if "kind" in decl_df.columns:
        if "id" in decl_df.columns:
            merged = merged.merge(decl_df[["id", "kind"]], on="id", how="left")
        elif "name" in decl_df.columns:
            merged = merged.merge(decl_df[["name", "kind"]], on="name", how="left")
        else:
            print("[whatif] decltypes has 'kind' but neither 'id' nor 'name'; ignoring kind filter.", file=sys.stderr)
            merged["kind"] = np.nan
    else:
        print("[whatif] decltypes missing 'kind' column; ignoring kind filter.", file=sys.stderr)
        merged["kind"] = np.nan

    if kinds:
        if merged["kind"].notna().any():
            kind_mask = merged["kind"].isin(kinds)
        else:
            print("[whatif] no kind information available after merge; not filtering by kind.", file=sys.stderr)
            kind_mask = np.ones(len(merged), dtype=bool)
    else:
        kind_mask = np.ones(len(merged), dtype=bool)

    if prefixes:
        name_mask = merged["name"].str.startswith(tuple(prefixes))
    else:
        name_mask = np.ones(len(merged), dtype=bool)

    return (kind_mask & name_mask).to_numpy()

class _SharedMLP(torch.nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 0, bias: bool = True):
        super().__init__()
        layers = []
        if hidden and hidden > 0:
            layers += [torch.nn.Linear(in_dim, hidden, bias=bias), torch.nn.ReLU()]
            layers += [torch.nn.Linear(hidden, emb_dim, bias=bias)]
        else:
            layers += [torch.nn.Linear(in_dim, emb_dim, bias=bias)]
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def _load_text_encoder(ckpt_path: Path, in_dim: int):
    if torch is None:
        raise RuntimeError("PyTorch is not available; cannot use --use_model.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    linear_keys = [k for k in state.keys() if k.endswith("weight") and getattr(state[k], "ndim", 0) == 2]
    if not linear_keys:
        raise RuntimeError("Could not find linear weights in checkpoint.")
    linear_ws = [state[k] for k in linear_keys]
    hidden = linear_ws[0].shape[0] if len(linear_ws) >= 2 else 0
    emb_dim = linear_ws[-1].shape[0]
    model = _SharedMLP(in_dim, emb_dim, hidden=hidden, bias=True)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        stripped = { (k.split(".",1)[1] if "." in k else k): v for k,v in state.items() }
        model.load_state_dict(stripped, strict=False)
    model.eval()
    return model

def katz_mass_frontier(row_ptr: np.ndarray, col_idx: np.ndarray, seeds: np.ndarray,
                       K: int = 6, beta: float = 0.2, max_nodes: int = 1_000_000):
    front_ids = seeds.astype(np.int64)
    front_vals = np.ones_like(front_ids, dtype=np.float64)
    total_mass = 0.0
    per_hop = []

    y_total_ids = []
    y_total_vals = []

    for k in range(1, K+1):
        if front_ids.size == 0:
            break
        mass_k = float(front_vals.sum())
        total_mass += (beta ** k) * mass_k
        per_hop.append((k, mass_k))

        y_total_ids.append(front_ids.copy())
        y_total_vals.append((beta ** k) * front_vals.copy())

        degs = row_ptr[front_ids + 1] - row_ptr[front_ids]
        total_len = int(degs.sum())
        if total_len == 0:
            front_ids = np.empty(0, dtype=np.int64)
            front_vals = np.empty(0, dtype=np.float64)
            continue
        if total_len > max_nodes:
            order = np.argsort(-front_vals)
            chosen = []
            acc = 0
            for idx in order:
                d = int(degs[idx])
                if d == 0:
                    continue
                if acc + d > max_nodes:
                    break
                chosen.append(idx)
                acc += d
            if not chosen:
                break
            front_ids = front_ids[chosen]
            front_vals = front_vals[chosen]
            degs = row_ptr[front_ids + 1] - row_ptr[front_ids]
            total_len = int(degs.sum())

        # Expand neighbors
        new_ids_list = []
        new_vals_list = []
        for i, u in enumerate(front_ids):
            s = row_ptr[u]; e = row_ptr[u+1]
            if e <= s:
                continue
            nbrs = col_idx[s:e]
            if nbrs.size == 0:
                continue
            new_ids_list.append(nbrs)
            new_vals_list.append(np.full_like(nbrs, front_vals[i], dtype=np.float64))

        if not new_ids_list:
            front_ids = np.empty(0, dtype=np.int64)
            front_vals = np.empty(0, dtype=np.float64)
            continue

        new_ids = np.concatenate(new_ids_list)
        new_vals = np.concatenate(new_vals_list)

        order = np.argsort(new_ids, kind="mergesort")
        new_ids = new_ids[order]; new_vals = new_vals[order]
        uniq, start = np.unique(new_ids, return_index=True)
        sums = np.add.reduceat(new_vals, start)
        front_ids = uniq
        front_vals = sums

    if y_total_ids:
        all_ids = np.concatenate(y_total_ids)
        all_vals = np.concatenate(y_total_vals)
        order = np.argsort(all_ids, kind="mergesort")
        all_ids = all_ids[order]; all_vals = all_vals[order]
        uniq, start = np.unique(all_ids, return_index=True)
        sums = np.add.reduceat(all_vals, start)
        y_total = (uniq, sums.astype(np.float32))
    else:
        y_total = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    return float(total_mass), y_total, per_hop

def main():
    ap = argparse.ArgumentParser(description="What-if graph productivity for a new statement.")
    ap.add_argument("--edge_index", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--decltypes", required=True)
    ap.add_argument("--type_features", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--type_string", required=True)
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--target_kinds", default="theorem,lemma")
    ap.add_argument("--target_prefixes", default="")
    ap.add_argument("--use_model", action="store_true")
    ap.add_argument("--graph_metrics", default="")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    nodes = pd.read_parquet(args.nodes)
    if "id" in nodes.columns:
        nodes_indexed = nodes.set_index("id", drop=False)
        N = int(nodes["id"].max()) + 1
    else:
        # Assume row index corresponds to id order
        nodes_indexed = nodes.copy()
        nodes_indexed["id"] = np.arange(len(nodes_indexed), dtype=np.int64)
        nodes_indexed = nodes_indexed.set_index("id", drop=False)
        N = len(nodes_indexed)

    decl = pd.read_parquet(args.decltypes)
    X = np.load(args.type_features)["X"].astype(np.float32)
    assert X.shape[0] == N, "type_features.npz rows must match nodes.parquet"
    row_ptr, col_idx, out_deg = _load_edge_index(Path(args.edge_index), N)

    kinds = [k.strip() for k in args.target_kinds.split(",") if k.strip()]
    prefixes = [p.strip() for p in args.target_prefixes.split(",") if p.strip()]
    mask = _build_target_mask(nodes_indexed.reset_index(drop=True)[["id","name"]], decl, kinds, prefixes)
    target_ids = np.nonzero(mask)[0]
    if target_ids.size == 0:
        print("[whatif] No candidate targets after filtering. Relax --target_kinds/--target_prefixes.", file=sys.stderr)
        sys.exit(1)

    x_new = _featurize_new_type(args.type_string, buckets=args.buckets)

    if args.use_model:
        if not args.ckpt:
            raise SystemExit("--use_model requires --ckpt outputs/text_ranker.pt")
        if torch is None:
            raise SystemExit("torch is not available; install it or drop --use_model")
        enc = _load_text_encoder(Path(args.ckpt), in_dim=X.shape[1])
        with torch.no_grad():
            e_new = enc(torch.from_numpy(x_new).unsqueeze(0)).squeeze(0).numpy()
            e_tgt = enc(torch.from_numpy(X[target_ids])).numpy()
            sims = (e_tgt @ e_new)
    else:
        x = X[target_ids]
        x_norm = np.linalg.norm(x, axis=1) + 1e-8
        new_norm = float(np.linalg.norm(x_new) + 1e-8)
        sims = (x @ x_new) / (x_norm * new_norm)

    L = min(args.L, target_ids.size)
    top_idx = np.argpartition(-sims, L-1)[:L]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    attach_targets = target_ids[top_sorted]
    attach_scores = sims[top_sorted]

    total_mass, (infl_ids, infl_vals), per_hop = katz_mass_frontier(row_ptr, col_idx, attach_targets, K=args.K, beta=args.beta)

    use_cost = _count_use_cost(args.type_string)
    prod_whatif = math.log1p((L + args.alpha * total_mass) / ((1 + use_cost) ** args.gamma))

    print(f"[whatif] candidates after filtering: {target_ids.size}  | attached L={L}")
    print("[whatif] top-L targets (name, score):")
    for i in range(min(L, 15)):
        j = int(attach_targets[i])
        nm = nodes_indexed.at[j, "name"]
        print(f"  - {nm}  {attach_scores[i]:.6f}")
    print("[whatif] per-hop mass: " + ", ".join([f"k={k}: {m:.1f}" for (k, m) in per_hop]))
    print(f"[whatif] total Katz mass (K={args.K}, beta={args.beta}): {total_mass:.2f}")
    print(f"[whatif] use-cost (rough hyp_count+forall): {use_cost}")
    print(f"[whatif] prod_whatif (alpha={args.alpha}, gamma={args.gamma}): {prod_whatif:.6f}")

    pct = None
    matched_col = None
    if args.graph_metrics:
        gm = pd.read_parquet(args.graph_metrics)
        wanted = f"prod_katz_a{args.alpha}_b{args.beta}_g{args.gamma}"
        if wanted in gm.columns:
            col = gm[wanted].to_numpy(np.float32)
            matched_col = wanted
            pct = float((col < prod_whatif).mean() * 100.0)
            print(f"[whatif] percentile of prod_whatif vs {matched_col}: {pct:.2f}")
        else:
            wanted2 = f"katz_k{args.K}_b{args.beta}"
            if wanted2 in gm.columns:
                col = gm[wanted2].to_numpy(np.float32)
                matched_col = wanted2
                pct = float((col < total_mass).mean() * 100.0)
                print(f"[whatif] percentile of katz_mass vs {matched_col}: {pct:.2f}")

    if infl_ids.size > 0:
        take = min(15, infl_ids.size)
        order = np.argsort(-infl_vals)[:take]
        print("[whatif] top influenced nodes:")
        for idx in order:
            nid = int(infl_ids[idx]); w = float(infl_vals[idx])
            nm = nodes_indexed.at[nid, "name"]
            print(f"  - {nm}  weight={w:.4f}")

    if args.out_json:
        out = {
            "type_string": args.type_string,
            "L": int(L), "K": int(args.K), "beta": float(args.beta),
            "alpha": float(args.alpha), "gamma": float(args.gamma),
            "attach_targets": [int(x) for x in attach_targets[:L].tolist()],
            "attach_target_names": [nodes_indexed.at[int(x), "name"] for x in attach_targets[:L].tolist()],
            "attach_scores": [float(s) for s in attach_scores[:L].tolist()],
            "per_hop": [{"k": int(k), "mass": float(m)} for (k, m) in per_hop],
            "total_mass": float(total_mass),
            "use_cost": int(use_cost),
            "prod_whatif": float(prod_whatif),
            "percentile": pct,
            "percentile_col": matched_col,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[whatif] wrote -> {args.out_json}")

if __name__ == "__main__":
    main()
