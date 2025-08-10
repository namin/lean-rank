import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from src.models.text_encoder import MLPEncoder
from src.utils.type_features import featurize_type

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
                    help="comma list to restrict targets (e.g. theorem,lemma). Empty = no filter.")
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

    # Targets = those in contexts.jsonl (order matters)
    tgt_ids = [int(json.loads(l)["target_id"]) for l in open(args.contexts, "r")]

    # Optional: restrict targets by kind (theorem/lemma)
    if args.target_kinds:
        nodes = pd.read_parquet(args.nodes)            # id,name
        decl = pd.read_parquet(args.decltypes)         # name,kind,...
        kind_by_id = nodes.merge(decl[["name","kind"]], on="name", how="left") \
                           .set_index("id")["kind"].to_dict()
        allowed = {k.strip() for k in args.target_kinds.split(",") if k.strip()}
        tgt_ids = [tid for tid in tgt_ids if kind_by_id.get(tid) in allowed]

    # Per-target top-K cutoffs from rankings.parquet (align to tgt_ids)
    rk = pd.read_parquet(args.rankings).set_index("target_id").loc[tgt_ids]
    taus = {K: rk["scores"].map(lambda s: s[K-1]).to_numpy(np.float32) for K in K_LIST}

    # Encode targets
    with torch.no_grad():
        G = enc(torch.from_numpy(X[tgt_ids])).cpu().numpy()  # (T, d)

    # Encode the NEW statement
    base, tri, head = featurize_type(args.type_string, buckets=args.buckets)
    x_new = np.concatenate([base, tri, head]).astype(np.float32)
    with torch.no_grad():
        z = enc(torch.from_numpy(x_new).unsqueeze(0)).cpu().numpy()[0]  # (d,)

    # Scores of new statement vs all targets
    s = G @ z  # (T,)

    # Report adoption@K and lift vs random (random ≈ K/N)
    T = len(tgt_ids)
    for K in K_LIST:
        adopt_frac = float((s >= taus[K]).mean())
        random_frac = K / N
        lift = adopt_frac / random_frac if random_frac > 0 else float("nan")
        print(f"adoption@{K}: {adopt_frac*100:.3f}%  "
              f"(~{adopt_frac*T:.0f}/{T});  random={random_frac*100:.3f}%  lift={lift:.2f}")

if __name__ == "__main__":
    main()
