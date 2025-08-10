from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.text_encoder import MLPEncoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512, help="contexts per batch")
    ap.add_argument("--chunk", type=int, default=128000, help="candidate chunk size")
    args = ap.parse_args()

    X = np.load(args.features)["X"].astype(np.float32)  # (N, D)
    N, D = X.shape
    X_t = torch.from_numpy(X)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    enc = MLPEncoder(in_dim=cfg["in_dim"], emb_dim=cfg["emb_dim"], hidden=cfg["hidden"])
    enc.load_state_dict(ckpt["state_dict"])
    enc.eval()

    # precompute premise embeddings E in chunks to keep memory sane
    with torch.no_grad():
        E_list = []
        for start in range(0, N, args.chunk):
            feats = X_t[start:start+args.chunk]
            E_list.append(enc(feats).cpu())
        E = torch.cat(E_list, dim=0)  # (N, d)

    # stream contexts
    targets = []
    with open(args.contexts, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            targets.append(int(o["target_id"]))

    rows = []
    for i in tqdm(range(0, len(targets), args.batch), desc="scoring"):
        batch_ids = targets[i:i+args.batch]
        with torch.no_grad():
            G = enc(X_t[batch_ids])             # (B, d)
            scores = (E @ G.T).T                # (B, N)
            topk = torch.topk(scores, k=min(args.topk, N), dim=1)
            for j, tid in enumerate(batch_ids):
                rows.append({
                    "target_id": int(tid),
                    "candidate_ids": topk.indices[j].cpu().numpy().tolist(),
                    "scores": topk.values[j].cpu().numpy().tolist(),
                })

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"[score_text_rankings] wrote -> {args.out}  (targets={len(rows)})")

if __name__ == "__main__":
    main()
