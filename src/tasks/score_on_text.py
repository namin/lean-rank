from pathlib import Path
import argparse, json
import numpy as np
import torch

from ..models.text_encoder import MLPEncoder
from ..utils.type_features import featurize_type, feature_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="type_features.npz (for dimensionality + candidates)")
    ap.add_argument("--nodes", required=True, help="nodes.parquet (for mapping names)")
    ap.add_argument("--ckpt", required=True, help="text_ranker.pt")
    ap.add_argument("--type_string", required=True, help="Lean type of the new target")
    ap.add_argument("--out", required=True, help="output parquet of single-row ranking")
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=128000)
    args = ap.parse_args()

    X = np.load(args.features)["X"].astype(np.float32)  # (N, D)
    N, D = X.shape
    X_t = torch.from_numpy(X)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    enc = MLPEncoder(in_dim=cfg["in_dim"], emb_dim=cfg["emb_dim"], hidden=cfg["hidden"])
    enc.load_state_dict(ckpt["state_dict"]); enc.eval()

    # featurize new type - must match training feature dimensions
    from ..utils.type_features import featurize_type
    base, tri, head = featurize_type(args.type_string, buckets=args.buckets)
    x_new = np.concatenate([base, tri, head]).astype(np.float32)
    
    # Check if we need to pad features to match model dimensions
    if x_new.shape[0] < cfg["in_dim"]:
        # Model was trained on combined features (structural + text)
        # We only have text features, so prepend zeros for missing structural features
        padding = cfg["in_dim"] - x_new.shape[0]
        x_new = np.concatenate([np.zeros(padding, dtype=np.float32), x_new])
    elif x_new.shape[0] > cfg["in_dim"]:
        print(f"Warning: features ({x_new.shape[0]}d) larger than model expects ({cfg['in_dim']}d), truncating")
        x_new = x_new[:cfg["in_dim"]]
    
    x_new_t = torch.from_numpy(x_new).unsqueeze(0)  # (1, D)

    # premise embeddings in chunks
    with torch.no_grad():
        E_list = []
        for start in range(0, N, args.chunk):
            feats = X_t[start:start+args.chunk]
            E_list.append(enc(feats).cpu())
        E = torch.cat(E_list, dim=0)  # (N, d)
        g = enc(x_new_t).squeeze(0)   # (d,)
        scores = (E @ g).numpy()
        topk = np.argpartition(-scores, args.topk-1)[:args.topk]
        topk = topk[np.argsort(-scores[topk])]

    import pandas as pd
    pd.DataFrame([{"target_id": -1, "candidate_ids": topk.tolist(), "scores": scores[topk].tolist()}]).to_parquet(args.out, index=False)
    print(f"[score_on_text] wrote -> {args.out}")

if __name__ == "__main__":
    main()
