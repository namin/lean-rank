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
                    help="used to read per-target Top-K cutoffs")
    ap.add_argument("--ckpt", default="outputs/text_ranker.pt")
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--k_list", type=str, default="5,10,20,50")
    args = ap.parse_args()

    K_LIST = [int(x) for x in args.k_list.split(",")]

    # load encoder + features
    X = np.load(args.features)["X"].astype(np.float32)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc = MLPEncoder(in_dim=ckpt["config"]["in_dim"],
                     emb_dim=ckpt["config"]["emb_dim"],
                     hidden=ckpt["config"]["hidden"])
    enc.load_state_dict(ckpt["state_dict"]); enc.eval()

    # targets = those that appear in contexts
    tgt_ids = [json.loads(l)["target_id"] for l in open(args.contexts, "r")]
    X_tgts = torch.from_numpy(X[tgt_ids])

    # encode targets
    with torch.no_grad():
        G = enc(X_tgts).cpu().numpy()  # (T, d)

    # per-target Top-K cutoffs from rankings.parquet (scores are already sorted desc)
    rk = pd.read_parquet(args.rankings)
    # align with tgt_ids order
    rk = rk.set_index("target_id").loc[tgt_ids]
    taus = {K: rk["scores"].map(lambda s: s[K-1]).to_numpy().astype(np.float32) for K in K_LIST}

    # encode the NEW statement
    base, tri, head = featurize_type(args.type_string, buckets=args.buckets)
    x_new = np.concatenate([base, tri, head]).astype(np.float32)
    with torch.no_grad():
        z = enc(torch.from_numpy(x_new).unsqueeze(0)).cpu().numpy()[0]  # (d,)

    # scores of new statement against all targets
    s = G @ z  # (T,)

    # report adoption@K
    T = len(tgt_ids)
    for K in K_LIST:
        frac = float((s >= taus[K]).mean())
        print(f"predicted adoption@{K}: {frac*100:.2f}%  (~{frac*T:.0f} / {T} targets)")

if __name__ == "__main__":
    main()
