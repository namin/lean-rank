from pathlib import Path
import argparse, re, json
import pyarrow as pa, pyarrow.parquet as pq
import pandas as pd

BLOCK_RE = re.compile(r"^---\s*$")

def parse_decltypes(in_path: Path):
    with in_path.open("r", encoding="utf-8") as f:
        cur = []
        for raw in f:
            line = raw.rstrip("\n")
            if BLOCK_RE.match(line):
                if cur:
                    kind = cur[0].strip()
                    name = cur[1].strip() if len(cur) > 1 else ""
                    type_text = "\n".join(cur[2:])
                    yield {"kind": kind, "name": name, "type_text": type_text}
                cur = []
            elif line:
                cur.append(line)
        if cur:
            kind = cur[0].strip()
            name = cur[1].strip() if len(cur) > 1 else ""
            type_text = "\n".join(cur[2:])
            yield {"kind": kind, "name": name, "type_text": type_text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decltypes", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(parse_decltypes(Path(args.decltypes)))
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), out/"decltypes.parquet")

    # optional alignment stats
    nodes = pd.read_parquet(args.nodes)
    in_nodes = set(nodes["name"])
    covered = df["name"].isin(in_nodes).sum()
    print(f"[build_decltypes] rows={len(df)}  aligned_to_nodes={covered}")

if __name__ == "__main__":
    main()
