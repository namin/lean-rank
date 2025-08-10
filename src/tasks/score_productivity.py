import argparse, json, re
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.text_encoder import MLPEncoder
from src.utils.type_features import featurize_type

def _kth_or_last(scores, k: int) -> float:
    """
    Return the k-th (1-based) element from a sequence/array `scores`.
    If k is beyond length, return the last element.
    If empty or None, return +inf so comparisons (s >= tau) will be false.
    Handles Python lists, tuples, numpy arrays, or pandas arrays.
    """
    if scores is None:
        return float('inf')
    # Coerce to 1D numpy array of floats
    try:
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    except Exception:
        try:
            arr = np.array(list(scores), dtype=np.float32).reshape(-1)
        except Exception:
            return float('inf')
    if arr.size == 0:
        return float('inf')
    idx = min(max(k-1, 0), arr.size - 1)
    return float(arr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type_string", required=True)
    ap.add_argument("--features", default="data/processed/type_features.npz")
    ap.add_argument("--contexts", default="data/processed/contexts.jsonl")
    ap.add_argument("--rankings", default="data/processed/rankings.parquet",
                    help="per-target top-K score lists (from score_text_rankings)")
    ap.add_argument("--nodes", default="data/processed/nodes.parquet")
    ap.add_argument("--decltypes", default="data/processed/decltypes.parquet")
    ap.add_argument("--ckpt", default="outputs/text_ranker.pt")
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--k_list", type=str, default="10,20,50")
    ap.add_argument("--target_kinds", type=str, default="theorem,lemma",
                    help="comma list to restrict targets by kind (e.g. theorem,lemma). Empty = no filter.")
    ap.add_argument("--target_prefixes", type=str, default="",
                    help="comma-separated name prefixes (e.g. 'TopologicalSpace.,Filter.')")
    ap.add_argument("--target_regex", type=str, default="",
                    help="Python regex to match target names (e.g. '^(List|Finset)\\.')")
    ap.add_argument("--regex_ignore_case", action="store_true",
                    help="Use re.I for --target_regex")
    # NEW: show which targets are hits at K
    ap.add_argument("--show_hits", type=int, default=0,
                    help="if >0, print which filtered targets are hits at this K (top-K)")
    ap.add_argument("--hits_limit", type=int, default=50,
                    help="max number of hits to print")
    args = ap.parse_args()

    K_LIST = [int(x) for x in args.k_list.split(",")]

    # Load features + encoder
    X = np.load(args.features)["X"].astype(np.float32)   # (N, D)
    N, D = X.shape
    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc = MLPEncoder(in_dim=ckpt["config"]["in_dim"],
                     emb_dim=ckpt["config"]["emb_dim"],
                     hidden=ckpt["config"]["hidden"])
    enc.load_state_dict(ckpt["state_dict"]); enc.eval()

    # Context targets (order matters)
    all_tgt_ids = [int(json.loads(l)["target_id"]) for l in open(args.contexts, "r")]

    # Join kinds/names
    nodes = pd.read_parquet(args.nodes)            # id,name
    id2name = dict(zip(nodes["id"], nodes["name"]))
    decl = pd.read_parquet(args.decltypes)         # name,kind,...
    kind_by_name = decl.set_index("name")["kind"].to_dict()
    kind_by_id = {i: kind_by_name.get(id2name.get(i, ""), None) for i in all_tgt_ids}

    # Build filters
    tgt_ids = list(all_tgt_ids)
    # kind filter
    if args.target_kinds:
        allowed = {k.strip() for k in args.target_kinds.split(",") if k.strip()}
        tgt_ids = [tid for tid in tgt_ids if kind_by_id.get(tid) in allowed]
    # prefix filter
    if args.target_prefixes.strip():
        prefixes = tuple(p.strip() for p in args.target_prefixes.split(",") if p.strip())
        tgt_ids = [tid for tid in tgt_ids if id2name.get(tid, "").startswith(prefixes)]
    # regex filter
    if args.target_regex.strip():
        flags = re.I if args.regex_ignore_case else 0
        pat = re.compile(args.target_regex, flags)
        tgt_ids = [tid for tid in tgt_ids if pat.search(id2name.get(tid, ""))]

    # Align rankings rows with filtered tgt_ids (intersection in order)
    rk = pd.read_parquet(args.rankings).set_index("target_id")
    missing = [tid for tid in tgt_ids if tid not in rk.index]
    if missing:
        tgt_ids = [tid for tid in tgt_ids if tid in rk.index]
    rk = rk.loc[tgt_ids]

    # Per-target top-K cutoffs for all requested K
    taus = {K: rk["scores"].map(lambda s: _kth_or_last(s, K)).to_numpy(np.float32) for K in K_LIST}

    # Encode targets
    with torch.no_grad():
        G = enc(torch.from_numpy(X[tgt_ids])).cpu().numpy()  # (T, d)

    # Encode NEW statement
    base, tri, head = featurize_type(args.type_string, buckets=args.buckets)
    x_new = np.concatenate([base, tri, head]).astype(np.float32)
    with torch.no_grad():
        z = enc(torch.from_numpy(x_new).unsqueeze(0)).cpu().numpy()[0]  # (d,)

    # Scores vs all filtered targets
    s = G @ z  # (T,)

    # Report adoption@K and lift vs random (random ≈ K/N, with full candidate pool N)
    T = len(tgt_ids)
    print(f"[info] filtered targets: {T} / total contexts {len(all_tgt_ids)}")
    if T == 0:
        print("No targets after filtering.")
        return

    for K in K_LIST:
        adopt_frac = float((s >= taus[K]).mean())
        random_frac = K / N
        lift = adopt_frac / random_frac if random_frac > 0 else float("nan")
        print(f"adoption@{K}: {adopt_frac*100:.3f}%  (~{adopt_frac*T:.0f}/{T});  random={random_frac*100:.3f}%  lift={lift:.2f}")

    # Optionally show which specific targets are hits at chosen K
    if args.show_hits and args.show_hits > 0:
        K = int(args.show_hits)
        tauK = taus.get(K)
        if tauK is None:
            # compute on the fly if not in K_LIST
            tauK = rk["scores"].map(lambda s: _kth_or_last(s, K)).to_numpy(np.float32)
        hits = np.where(s >= tauK)[0]
        if hits.size == 0:
            print(f"[hits@{K}] none")
            return
        # sort hits by margin descending
        margins = (s - tauK)[hits]
        order = np.argsort(-margins)
        hits = hits[order]
        print(f"[hits@{K}] showing up to {args.hits_limit} of {len(hits)}:")
        for idx in hits[:args.hits_limit]:
            tid = int(tgt_ids[idx])
            name = id2name.get(tid, f"<{tid}>")
            print(f" - {name}   score={s[idx]:.6f}  cutoff={tauK[idx]:.6f}  margin={s[idx]-tauK[idx]:.6f}")

if __name__ == "__main__":
    main()
