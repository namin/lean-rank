import argparse, json, re
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd

def load_nodes(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    df = pd.read_parquet(path)
    id2name = dict(zip(df["id"].astype(int), df["name"].astype(str)))
    name2id = dict(zip(df["name"].astype(str), df["id"].astype(int)))
    return id2name, name2id

def load_contexts(path: Path) -> Dict[int, Set[int]]:
    pos: Dict[int, Set[int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            tid = int(o["target_id"])
            pos[tid] = set(int(x) for x in o.get("positives", []))
    return pos

def recall_at_k(pred: List[int], pos: Set[int], k: int) -> float:
    if not pos:
        return 0.0
    return len(set(pred[:k]) & pos) / float(len(pos))

def mrr_at_k(pred: List[int], pos: Set[int], k: int) -> float:
    seen = set()
    for rank, pid in enumerate(pred[:k], start=1):
        if pid in pos and pid not in seen:
            return 1.0 / rank
        seen.add(pid)
    return 0.0

def explain_row(row, id2name: Dict[int, str], positives: Dict[int, Set[int]], k: int):
    tid = int(row["target_id"])
    pred_ids = list(row["candidate_ids"])[:k]
    scores = list(row.get("scores", [None] * len(pred_ids)))[:k]
    pos = positives.get(tid, set())

    names = [id2name.get(i, f"<{i}>") for i in pred_ids]
    hits_names = [id2name.get(i, f"<{i}>") for i in pred_ids if i in pos]

    return {
        "target_id": tid,
        "target_name": id2name.get(tid, f"<{tid}>"),
        "k": k,
        "recall_at_k": recall_at_k(pred_ids, pos, k),
        "mrr_at_k": mrr_at_k(pred_ids, pos, k),
        "num_pos_total": len(pos),
        "num_hits_topk": len(hits_names),
        "topk_ids": pred_ids,
        "topk_names": names,
        "topk_scores": scores,
        "hits_in_topk": hits_names,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=200, help="max number of targets to include")
    ap.add_argument("--start", type=int, default=0, help="start offset into rankings")
    ap.add_argument("--format", choices=["parquet","csv","md","txt"], default="parquet")
    ap.add_argument("--sort_by", choices=["index","recall","mrr","hits"], default="index",
                    help="sort rows by recall, mrr, hits, or keep original index")
    ap.add_argument("--filter_target", type=str, default=None, help="substring or regex to match target names")
    ap.add_argument("--regex", action="store_true", help="treat --filter_target as regex")
    args = ap.parse_args()

    id2name, name2id = load_nodes(Path(args.nodes))
    positives = load_contexts(Path(args.contexts))

    rk = pd.read_parquet(args.rankings)

    # Optional filtering by name
    if args.filter_target:
        pat = args.filter_target if args.regex else re.escape(args.filter_target)
        mask = rk["target_id"].map(lambda i: re.search(pat, id2name.get(int(i), ""), flags=re.I) is not None)
        rk = rk[mask]

    # Slice
    if args.start:
        rk = rk.iloc[args.start:]
    if args.limit:
        rk = rk.iloc[:args.limit]

    # Build explained rows
    rows = [explain_row(rk.iloc[i], id2name, positives, args.topk) for i in range(len(rk))]
    df = pd.DataFrame(rows)

    # Sorting
    if args.sort_by == "recall":
        df = df.sort_values("recall_at_k", ascending=False)
    elif args.sort_by == "mrr":
        df = df.sort_values("mrr_at_k", ascending=False)
    elif args.sort_by == "hits":
        df = df.sort_values("num_hits_topk", ascending=False)
    # else "index": keep original order

    # Write output
    out_path = Path(args.out)
    fmt = args.format or out_path.suffix.lstrip(".")
    if fmt == "parquet" or out_path.suffix == ".parquet":
        out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
    elif fmt == "csv" or out_path.suffix == ".csv":
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    elif fmt in {"md","txt"} or out_path.suffix in {".md",".txt"}:
        out_path = out_path.with_suffix(".md" if fmt == "md" else ".txt")
        with out_path.open("w", encoding="utf-8") as w:
            for _, r in df.iterrows():
                w.write(f"## {r['target_name']} (id={r['target_id']})\n")
                w.write(f"- recall@{r['k']}: {r['recall_at_k']:.3f}   mrr@{r['k']}: {r['mrr_at_k']:.3f}   "
                        f"hits/topk: {int(r['num_hits_topk'])}/{int(r['k'])}   positives_total: {int(r['num_pos_total'])}\n")
                topn = min(10, len(r['topk_names']))
                w.write(f"- top{topn}: " + ", ".join(r['topk_names'][:topn]) + "\n")
                if r['hits_in_topk']:
                    w.write(f"- hits_in_top{r['k']}: " + ", ".join(r['hits_in_topk']) + "\n")
                w.write("\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")

    print(f"Wrote report to {out_path} with {len(df)} targets. "
          f"Mean recall@{args.topk}={df['recall_at_k'].mean():.4f}  "
          f"Mean MRR@{args.topk}={df['mrr_at_k'].mean():.4f}")

if __name__ == "__main__":
    main()
