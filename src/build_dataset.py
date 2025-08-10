from pathlib import Path
import argparse, json, re
import pyarrow as pa, pyarrow.parquet as pq
import pandas as pd

BLOCK_RE = re.compile(r"^---\s*$")
STAR_RE  = re.compile(r"^\*\s+")
SIMP_RE  = re.compile(r"^s\s+")

def build(premises: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = out_dir / "contexts.jsonl"

    node2id = {}
    nodes = []

    def ensure_id(name: str):
        i = node2id.get(name)
        if i is None:
            i = len(nodes); node2id[name] = i; nodes.append(name)
        return i

    n_ctx = 0
    with premises.open("r", encoding="utf-8") as f, ctx_path.open("w", encoding="utf-8") as wctx:
        cur = []
        for raw in f:
            line = raw.rstrip("\n")
            if BLOCK_RE.match(line):
                if cur:
                    tgt = cur[0].strip()
                    tid = ensure_id(tgt)
                    pos_ids = []
                    for rawp in cur[1:]:
                        l = rawp.lstrip()
                        flag = (STAR_RE.match(l) or SIMP_RE.match(l))
                        name = l[2:].strip() if flag else l.strip()
                        if not name: continue
                        pid = ensure_id(name)
                        pos_ids.append(pid)
                    if pos_ids:
                        wctx.write(json.dumps({"target_id": tid, "positives": pos_ids}) + "\n")
                        n_ctx += 1
                cur = []
            elif line and not line.startswith("premises"):
                cur.append(line)
        if cur:
            tgt = cur[0].strip()
            tid = ensure_id(tgt)
            pos_ids = []
            for rawp in cur[1:]:
                l = rawp.lstrip()
                flag = (STAR_RE.match(l) or SIMP_RE.match(l))
                name = l[2:].strip() if flag else l.strip()
                if not name: continue
                pid = ensure_id(name)
                pos_ids.append(pid)
            if pos_ids:
                wctx.write(json.dumps({"target_id": tid, "positives": pos_ids}) + "\n")
                n_ctx += 1

    # nodes parquet
    table = pa.Table.from_arrays(
        [pa.array(list(range(len(nodes)))), pa.array(nodes)],
        names=["id","name"]
    )
    pq.write_table(table, out_dir/"nodes.parquet")

    (out_dir/"meta.json").write_text(json.dumps({
        "num_nodes": len(nodes),
        "num_contexts": n_ctx
    }, indent=2))
    print(f"[build_dataset] wrote nodes.parquet, contexts.jsonl, meta.json in {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    build(Path(args.premises), Path(args.out_dir))

if __name__ == "__main__":
    main()
