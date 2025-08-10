from pathlib import Path
import argparse, numpy as np, pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from ..models.gnn_model import GraphSAGEProd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_index", required=True)
    ap.add_argument("--graph_metrics", required=True)
    ap.add_argument("--type_features", default="", help="optional .npz with X matrix")
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--label_col", default="", help="column in graph_metrics to predict (default: first prod_* col)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_nodes", type=int, default=65536)
    ap.add_argument("--fanout", type=str, default="15,15")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    gm = pd.read_parquet(args.graph_metrics)
    prod_cols = [c for c in gm.columns if c.startswith("prod_")]
    if not prod_cols and not args.label_col:
        raise ValueError("No prod_* column found; run build_graph first.")
    label_col = args.label_col or prod_cols[0]
    y = gm[label_col].to_numpy(np.float32)

    graph_cols = []
    for c in ["in_deg","out_deg"] + [c for c in gm.columns if c.startswith("katz_k")] + [c for c in gm.columns if c.startswith("pagerank_d")]:
        if c in gm.columns:
            graph_cols.append(c)
    X_graph = gm[graph_cols].to_numpy(np.float32)
    if args.type_features:
        X_text = np.load(args.type_features)["X"].astype(np.float32)
        X = np.concatenate([X_text, X_graph], axis=1)
    else:
        X = X_graph
    N, Din = X.shape

    ed = np.load(args.edge_index)
    src = torch.from_numpy(ed["src"].astype(np.int64))
    dst = torch.from_numpy(ed["dst"].astype(np.int64))
    edge_index = torch.stack([src, dst], dim=0)

    data = Data(x=torch.from_numpy(X), edge_index=edge_index, y=torch.from_numpy(y))

    idx = np.arange(N); np.random.shuffle(idx)
    split = int(0.9 * N)
    train_idx = torch.from_numpy(idx[:split])
    val_idx   = torch.from_numpy(idx[split:])

    fanout = [int(x) for x in args.fanout.split(",") if x.strip()]
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=fanout,
                                  batch_size=args.batch_nodes, shuffle=True)
    val_loader = NeighborLoader(data, input_nodes=val_idx, num_neighbors=fanout,
                                batch_size=args.batch_nodes, shuffle=False)

    device = torch.device(args.device)
    model = GraphSAGEProd(in_dim=Din, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = {"mae": float("inf"), "state": None}
    for ep in range(1, args.epochs+1):
        model.train()
        tloss = tmae = tn = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {ep}/{args.epochs} [train]"):
            batch = batch.to(device)
            pred, _ = model(batch.x, batch.edge_index)
            bs = getattr(batch, "batch_size", None)
            if bs is None:
                raise RuntimeError("NeighborLoader batch missing batch_size; please upgrade PyG.")
            pb = pred[:bs]
            yb = batch.y[:bs]
            loss = loss_fn(pb, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                mae = (pb - yb).abs().mean().item()
            tloss += loss.item() * bs; tmae += mae * bs; tn += bs
        print(f"[train] loss={tloss/tn:.6f}  mae={tmae/tn:.6f}")

        model.eval(); vmae = vn = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {ep}/{args.epochs} [val]"):
                batch = batch.to(device)
                pred, _ = model(batch.x, batch.edge_index)
                bs = getattr(batch, "batch_size", None)
                pb = pred[:bs]; yb = batch.y[:bs]
                mae = (pb - yb).abs().mean().item()
                vmae += mae * bs; vn += bs
        vmae /= max(1, vn)
        print(f"[val] mae={vmae:.6f}")
        if vmae < best["mae"]:
            best["mae"] = vmae
            best["state"] = model.state_dict()

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best["state"] if best["state"] is not None else model.state_dict(),
        "config": {"in_dim": Din, "hid": args.hid, "layers": args.layers, "dropout": args.dropout},
        "meta": {"label_col": label_col, "graph_cols": graph_cols, "used_text": bool(args.type_features)}
    }, args.out_ckpt)
    print(f"[train_gnn_productivity] saved -> {args.out_ckpt}  best_val_mae={best['mae']:.6f}")

if __name__ == "__main__":
    main()
