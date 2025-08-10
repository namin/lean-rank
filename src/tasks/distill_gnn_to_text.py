from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from ..models.mlp_prod import MLPProd

def main():
    ap = argparse.ArgumentParser(description="Train a text-only MLP to predict graph productivity (distillation).")
    ap.add_argument("--type_features", required=True, help="data/processed/type_features.npz")
    ap.add_argument("--labels_parquet", required=True, help="graph_metrics.parquet or gnn_productivity.parquet")
    ap.add_argument("--label_col", default="", help="e.g., prod_katz_a0.2_b0.2_g0.5 or y_pred")
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    X = np.load(args.type_features)["X"].astype(np.float32)
    df = pd.read_parquet(args.labels_parquet)
    if not args.label_col:
        # prefer gnn y_pred if present, otherwise first prod_* col
        if "y_pred" in df.columns:
            args.label_col = "y_pred"
        else:
            cands = [c for c in df.columns if c.startswith("prod_")]
            if not cands:
                raise ValueError("No label column found; pass --label_col explicitly")
            args.label_col = cands[0]
    y = df[args.label_col].to_numpy(np.float32)
    assert X.shape[0] == y.shape[0], "Row mismatch between features and labels"

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    dev = torch.device(args.device)

    model = MLPProd(in_dim=X.shape[1], hidden=args.hidden).to(dev)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs+1):
        tl, tn = 0.0, 0
        for xb, yb in tqdm(dl, desc=f"epoch {ep}/{args.epochs}"):
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += float(loss.detach().cpu()) * xb.size(0); tn += xb.size(0)
        print(f"[distill] epoch {ep}  mse={tl/tn:.6f}")

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"in_dim": X.shape[1], "hidden": args.hidden}}, args.out_ckpt)
    print(f"[distill_gnn_to_text] saved -> {args.out_ckpt}  label={args.label_col}")

if __name__ == "__main__":
    main()
