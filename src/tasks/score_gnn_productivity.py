from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from ..models.gnn_model import GraphSAGEProd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_index", required=True)
    ap.add_argument("--graph_metrics", required=True)
    ap.add_argument("--type_features", default="", help="optional .npz with X matrix")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fanout", type=str, default="15,15")
    ap.add_argument("--batch_nodes", type=int, default=65536)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    gm = pd.read_parquet(args.graph_metrics)
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

    data = Data(x=torch.from_numpy(X), edge_index=edge_index)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = GraphSAGEProd(in_dim=cfg["in_dim"], hid=cfg["hid"], layers=cfg["layers"], dropout=cfg["dropout"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    fanout = [int(x) for x in args.fanout.split(",") if x.strip()]
    loader = NeighborLoader(data, input_nodes=torch.arange(N), num_neighbors=fanout,
                            batch_size=args.batch_nodes, shuffle=False)

    y_pred = torch.empty(N, dtype=torch.float32)
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            batch = batch.to(args.device)
            yb, _ = model(batch.x, batch.edge_index)
            bs = getattr(batch, "batch_size", None)
            seeds_global = batch.n_id[:bs].cpu()
            y_pred[seeds_global] = yb[:bs].cpu()

    nodes = pd.read_parquet(args.nodes)
    out = pd.DataFrame({
        "id": nodes["id"].astype(np.int64),
        "name": nodes["name"].astype(str),
        "y_pred": y_pred.numpy().astype(np.float32),
    })
    for c in graph_cols:
        out[c] = gm[c].astype(np.float32 if c.startswith(("katz_k","pagerank_d")) else np.int64)

    out.to_parquet(args.out, index=False)
    print(f"[score_gnn_productivity] wrote -> {args.out}")

if __name__ == "__main__":
    main()
