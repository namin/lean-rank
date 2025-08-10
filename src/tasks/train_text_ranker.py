from pathlib import Path
import argparse, json, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..models.text_encoder import MLPEncoder

class Contexts(Dataset):
    def __init__(self, contexts_path: Path, num_nodes: int, neg_per_pos: int = 8, batch_triplets: int = 512, seed: int = 42):
        self.targets = []
        self.positives = []
        with contexts_path.open("r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                pos = o.get("positives", [])
                if not pos:
                    continue
                self.targets.append(int(o["target_id"]))
                self.positives.append([int(x) for x in pos])
        self.num_nodes = int(num_nodes)
        self.neg_per_pos = int(neg_per_pos)
        self.batch_triplets = int(batch_triplets)
        self.rng = random.Random(seed)

    def __len__(self):
        # “virtual epoch”: ~10 sampled batches per epoch
        return max(1, len(self.targets) // 10)

    def __getitem__(self, _):
        tgts, pos_ids, neg_ids_list = [], [], []
        for _ in range(self.batch_triplets):
            j = self.rng.randrange(len(self.targets))
            tgt = self.targets[j]
            pos_list = self.positives[j]
            p = self.rng.choice(pos_list)
            bad = set(pos_list); bad.add(tgt)
            negs = []
            while len(negs) < self.neg_per_pos:
                cand = self.rng.randrange(self.num_nodes)
                if cand not in bad:
                    negs.append(cand)
            tgts.append(tgt); pos_ids.append(p); neg_ids_list.append(negs)
        return tgts, pos_ids, neg_ids_list

def to_int_list(x):
    """Recursively convert tensors/np scalars/lists into a plain list[int]."""
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return [int(x.item())]
        return [int(v) for v in x.detach().cpu().view(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        out = []
        for elt in x:
            if isinstance(elt, (list, tuple, torch.Tensor, np.ndarray)):
                out.extend(to_int_list(elt))
            else:
                out.append(int(elt))
        return out
    # scalar
    return [int(x)]

def train(features_npz: Path, contexts_jsonl: Path, out_ckpt: Path,
          emb_dim: int, hidden: int, lr: float, epochs: int, batch: int, neg_per_pos: int, device: str):
    X = np.load(features_npz)["X"].astype(np.float32)  # (N, D)
    N, D = X.shape
    X_t = torch.from_numpy(X)

    enc = MLPEncoder(in_dim=D, emb_dim=emb_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr)

    ds = Contexts(contexts_jsonl, num_nodes=N, neg_per_pos=neg_per_pos, batch_triplets=batch)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    enc.train()
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{epochs}")
        for tgts, pos_ids, neg_ids_list in pbar:
            # unwrap batch_dim=1 and normalize to lists of ints
            tgts = to_int_list(tgts[0])
            pos_ids = to_int_list(pos_ids[0])
            # neg_ids_list[0] can be list[list[int]] or tensor; normalize then regroup
            flat_negs = to_int_list(neg_ids_list[0])
            # K = negs per pos, reconstruct rows in order
            K = len(flat_negs) // max(1, len(tgts))
            neg_rows = [flat_negs[i*K:(i+1)*K] for i in range(len(tgts))]

            if len(tgts) == 0 or K == 0:
                continue

            # unique ids to compute embeddings once
            uniq_ids = list({int(i) for i in (tgts + pos_ids + flat_negs)})
            id2row = {i: r for r, i in enumerate(uniq_ids)}

            feats = X_t[uniq_ids].to(device)
            Z = enc(feats)  # (U, d)

            # map to embeddings
            g   = Z[[id2row[int(i)] for i in tgts]]               # (B, d)
            pos = Z[[id2row[int(i)] for i in pos_ids]]            # (B, d)
            neg = Z[[id2row[int(i)] for i in flat_negs]].view(len(tgts), K, -1)  # (B, K, d)

            s_pos  = (g * pos).sum(-1)                            # (B,)
            s_negs = (g.unsqueeze(1) * neg).sum(-1)               # (B, K)

            loss = -torch.log(torch.sigmoid(s_pos.unsqueeze(1) - s_negs).clamp_min(1e-9)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": enc.state_dict(),
        "config": {"in_dim": D, "emb_dim": emb_dim, "hidden": hidden}
    }, out_ckpt)
    print(f"[train_text_ranker] saved -> {out_ckpt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512, help="triplets per sampled batch")
    ap.add_argument("--neg_per_pos", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(Path(args.features), Path(args.contexts), Path(args.out_ckpt),
          emb_dim=args.emb_dim, hidden=args.hidden, lr=args.lr, epochs=args.epochs,
          batch=args.batch, neg_per_pos=args.neg_per_pos, device=args.device)

if __name__ == "__main__":
    main()
