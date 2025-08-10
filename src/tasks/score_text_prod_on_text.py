from pathlib import Path
import argparse, numpy as np, torch
from ..models.mlp_prod import MLPProd
from src.utils.type_features import featurize_type  # uses your existing util

def main():
    ap = argparse.ArgumentParser(description="Score a new type string with the distilled text-only productivity model.")
    ap.add_argument("--ckpt", required=True, help="outputs/text_prod.pt (from distill_gnn_to_text)")
    ap.add_argument("--type_string", required=True)
    ap.add_argument("--buckets", type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model = MLPProd(in_dim=cfg["in_dim"], hidden=cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    base, tri, head = featurize_type(args.type_string, buckets=args.buckets)
    x = np.concatenate([base, tri, head]).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(x).unsqueeze(0)).item()
    print(f"predicted_productivity (text-only) = {y:.6f}")

if __name__ == "__main__":
    main()
