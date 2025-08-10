from pathlib import Path
import argparse, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Mix premise-selection scores with a productivity prior.")
    ap.add_argument("--rankings_in", required=True, help="rankings.parquet from score_text_rankings.py")
    ap.add_argument("--prior_parquet", required=True, help="gnn_productivity.parquet or graph_metrics.parquet")
    ap.add_argument("--prior_id_col", default="id")
    ap.add_argument("--prior_score_col", default="", help="column name with prior score (e.g., y_pred or prod_katz_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.2, help="mix weight for prior in linear mix")
    ap.add_argument("--mix", choices=["linear","mult"], default="linear", help="linear: s'=s+alpha*prior; mult: s'=s*(1+alpha*prior_z)")
    ap.add_argument("--zscore_prior", action="store_true", help="z-score the prior before mixing")
    args = ap.parse_args()

    rk = pd.read_parquet(args.rankings_in)
    pr = pd.read_parquet(args.prior_parquet)
    if not args.prior_score_col:
        # pick first plausible prior column
        for c in ["y_pred"] + [c for c in pr.columns if c.startswith("prod_")]:
            if c in pr.columns:
                args.prior_score_col = c; break
        if not args.prior_score_col:
            raise ValueError("Could not infer prior score column; pass --prior_score_col")
    pr = pr[[args.prior_id_col, args.prior_score_col]].rename(columns={args.prior_id_col:"id","%s"%args.prior_score_col:"prior"})

    # z-score if requested
    if args.zscore_prior:
        pr_mean = pr["prior"].mean()
        pr_std = pr["prior"].std() + 1e-8
        pr["prior"] = (pr["prior"] - pr_mean) / pr_std

    prior_map = dict(zip(pr["id"].astype(int), pr["prior"].astype(float)))

    def mix_row(row):
        cids = row["candidate_ids"]
        scores = np.array(row["scores"], dtype=np.float32)
        pri = np.array([prior_map.get(int(i), 0.0) for i in cids], dtype=np.float32)
        if args.mix == "linear":
            new = scores + args.alpha * pri
        else:
            # multiply by (1 + alpha * normalized prior)
            if not args.zscore_prior:
                # normalize to [0,1] using min/max over mapped values
                # to avoid per-row leakage, we use global min/max
                # compute once: but for simplicity, zscore_prior is recommended
                pass
            new = scores * (1.0 + args.alpha * pri)
        idx = np.argsort(-new)
        return pd.Series({"candidate_ids": [int(cids[i]) for i in idx], "scores": [float(new[i]) for i in idx]})

    out = rk.copy()
    mixed = out.apply(mix_row, axis=1)
    out["candidate_ids"] = mixed["candidate_ids"]
    out["scores"] = mixed["scores"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[reweight_rankings] wrote -> {args.out}  using prior={args.prior_parquet}:{args.prior_score_col}  alpha={args.alpha} mix={args.mix}")

if __name__ == "__main__":
    main()
