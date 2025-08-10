from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

def pagerank_col_stochastic(A_csr, d=0.85, iters=60, tol=1e-6):
    n = A_csr.shape[0]
    B = A_csr.transpose().tocsr().astype(np.float64)
    col_sums = np.array(B.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - d) / n
    for _ in range(iters):
        x = pr / col_sums
        y = B.dot(x)
        pr_new = d * y + teleport
        if np.linalg.norm(pr_new - pr, 1) < tol:
            pr = pr_new
            break
        pr = pr_new
    return pr.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out_edges", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--katz_k", type=int, default=8)
    ap.add_argument("--katz_beta", type=float, default=0.2)
    ap.add_argument("--reach_h", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()

    nodes = pd.read_parquet(args.nodes)
    n = int(nodes["id"].max()) + 1

    src_list, dst_list = [], []
    in_deg = np.zeros(n, dtype=np.int64)
    out_deg = np.zeros(n, dtype=np.int64)

    with open(args.contexts, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading contexts"):
            o = json.loads(line)
            t = int(o["target_id"])
            if t >= n: continue
            for p in o.get("positives", []):
                p = int(p)
                if p >= n: continue
                src_list.append(p)   # premise ->
                dst_list.append(t)   #          target
                out_deg[p] += 1
                in_deg[t]  += 1

    E = len(src_list)
    print(f"[graph] N={n}  E={E}")
    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    A = csr_matrix((np.ones(E, dtype=np.float32), (src, dst)), shape=(n, n))

    # Truncated Katz
    beta, K = args.katz_beta, args.katz_k
    v = np.ones(n, dtype=np.float32)
    katz = np.zeros(n, dtype=np.float32)
    for k in range(1, K+1):
        v = A.dot(v)
        katz += (beta**k) * v
        if k == 1:
            print(f"[katz] k=1: mean={v.mean():.2f} max={v.max():.0f}")

    # Finite-hop reach
    H = args.reach_h
    v2 = np.ones(n, dtype=np.float32)
    reach = np.zeros(n, dtype=np.float32)
    for h in range(1, H+1):
        v2 = A.dot(v2)
        reach += v2

    pr = pagerank_col_stochastic(A, d=0.85)

    numer_katz = out_deg.astype(np.float32) + args.alpha * katz
    denom = (1.0 + in_deg.astype(np.float32)) ** args.gamma
    prod_katz = np.log1p(numer_katz / denom)

    numer_reach = out_deg.astype(np.float32) + args.alpha * reach
    prod_reach = np.log1p(numer_reach / denom)

    metrics = pd.DataFrame({
        "id": np.arange(n, dtype=np.int64),
        "in_deg": in_deg,
        "out_deg": out_deg,
        f"katz_k{K}_b{beta}": katz,
        f"reach_h{H}": reach,
        f"pagerank_d{0.85}": pr,
        f"prod_katz_a{args.alpha}_b{beta}_g{args.gamma}": prod_katz,
        f"prod_reach_a{args.alpha}_h{H}_g{args.gamma}": prod_reach,
    })
    metrics.to_parquet(args.out_metrics, index=False)
    np.savez_compressed(args.out_edges, src=src, dst=dst)
    print(f"[graph] wrote edges -> {args.out_edges} and metrics -> {args.out_metrics}")

if __name__ == "__main__":
    main()
