from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utils.type_features import featurize_type, feature_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decltypes", required=True, help="decltypes.parquet")
    ap.add_argument("--nodes", required=True, help="nodes.parquet")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--buckets", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_parquet(args.decltypes)  # name, kind, type_text
    nodes = pd.read_parquet(args.nodes)
    id_map = dict(zip(nodes["name"], nodes["id"]))

    D = 6 + args.buckets + args.buckets
    X = np.zeros((len(nodes), D), dtype=np.float32)

    # featurize only those that are in nodes
    for _, row in tqdm(df.iterrows(), total=len(df), desc="featurizing"):
        name = row["name"]
        ttext = row.get("type_text", "")
        i = id_map.get(name)
        if i is None: continue
        base, tri, head = featurize_type(ttext, buckets=args.buckets)
        X[i, :6] = base
        X[i, 6:6+args.buckets] = tri
        X[i, 6+args.buckets:] = head

    np.savez_compressed(args.out, X=X)
    (Path(args.out).with_suffix(".features.json")).write_text(json.dumps(feature_names(args.buckets), indent=2))
    print(f"[build_type_features] wrote {args.out} with shape {X.shape}")

if __name__ == "__main__":
    main()
