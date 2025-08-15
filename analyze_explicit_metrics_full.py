#!/usr/bin/env python3
"""
Analyze theorem importance using explicit, interpretable metrics with FULL transitive closure.
Based on mathematician's suggestions:
1. Direct usage count (how many theorems call this one)
2. Full transitive dependency count (how many theorems rely on this one, directly or indirectly)
3. Combined metric: (direct * transitive) / (existential_quantifiers + 1)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy.sparse import csr_matrix
from tqdm import tqdm


def build_dependency_graph(contexts_file):
    """Build adjacency matrix from contexts.jsonl."""
    print("Building dependency graph...")
    
    # First pass: find max id
    max_id = 0
    with open(contexts_file, "r") as f:
        for line in tqdm(f, desc="Finding max ID"):
            o = json.loads(line)
            max_id = max(max_id, o.get("target_id", 0))
            for p in o.get("positives", []):
                max_id = max(max_id, p)
    
    n = max_id + 1
    print(f"Graph size: {n} nodes")
    
    # Second pass: build edges
    edges = []
    with open(contexts_file, "r") as f:
        for line in tqdm(f, desc="Building edges"):
            o = json.loads(line)
            target = o.get("target_id")
            if target is None or target >= n:
                continue
            for premise in o.get("positives", []):
                if premise < n:
                    edges.append((premise, target))  # premise -> target
    
    print(f"Total edges: {len(edges)}")
    
    # Create sparse adjacency matrix
    if edges:
        src, dst = zip(*edges)
        data = np.ones(len(edges), dtype=np.float32)
        A = csr_matrix((data, (src, dst)), shape=(n, n))
    else:
        A = csr_matrix((n, n), dtype=np.float32)
    
    return A, n


def compute_transitive_closure(A):
    """Compute full transitive closure using Floyd-Warshall approach (sparse version)."""
    print("Computing full transitive closure...")
    n = A.shape[0]
    
    # Convert to dense for smaller graphs or use iterative approach for large
    if n < 5000:  # For smaller graphs, use dense computation
        print("Using dense matrix computation...")
        A_dense = A.toarray()
        TC = (A_dense > 0).astype(np.float32)
        
        # Floyd-Warshall
        for k in tqdm(range(n), desc="Computing closure"):
            TC = np.logical_or(TC, (TC[:, k:k+1] @ TC[k:k+1, :]).astype(bool)).astype(np.float32)
        
        # Count reachable nodes (excluding self)
        transitive_count = TC.sum(axis=1) - np.diag(TC)
    else:
        # For larger graphs, use iterative sparse approach
        print("Using sparse iterative computation...")
        TC = A.copy()
        TC_prev = csr_matrix((n, n), dtype=np.float32)
        
        iteration = 0
        while (TC != TC_prev).nnz > 0 and iteration < 100:
            iteration += 1
            print(f"Iteration {iteration}: {TC.nnz} edges")
            TC_prev = TC.copy()
            TC = (TC + TC @ A).astype(bool).astype(np.float32)
        
        # Count reachable nodes
        transitive_count = np.array(TC.sum(axis=1)).ravel()
    
    return transitive_count


def load_data(data_dir="data/number_theory_filtered"):
    """Load graph metrics, nodes, and declaration structures."""
    
    # Build full dependency graph
    contexts_file = f"{data_dir}/processed/contexts.jsonl"
    A, n = build_dependency_graph(contexts_file)
    
    # Compute in-degree (direct usage)
    in_deg = np.array(A.sum(axis=0)).ravel()
    
    # Compute full transitive closure
    transitive_deps = compute_transitive_closure(A)
    
    # Load existing data for comparison
    graph_metrics = pd.read_parquet(f"{data_dir}/processed/graph_metrics.parquet")
    nodes = pd.read_parquet(f"{data_dir}/processed/nodes.parquet")
    
    # Load declaration structures
    declarations = []
    with open(f"{data_dir}/declaration_structures.jsonl", "r") as f:
        for line in f:
            declarations.append(json.loads(line))
    
    decl_map = {d["name"]: d for d in declarations}
    
    return graph_metrics, nodes, decl_map, in_deg, transitive_deps


def compute_explicit_metrics(graph_metrics, nodes, decl_map, in_deg, transitive_deps):
    """Compute the three explicit metrics with FULL transitive closure."""
    
    # Start with nodes dataframe
    df = nodes.copy()
    
    # Add our computed metrics
    df["in_deg_computed"] = in_deg[:len(df)]
    df["transitive_full"] = transitive_deps[:len(df)]
    
    # Merge with existing graph metrics for comparison
    df = df.merge(graph_metrics[["id", "in_deg", "reach_h3"]], on="id", how="left")
    
    # Add declaration info
    def get_decl_info(name):
        if name in decl_map:
            d = decl_map[name]
            return pd.Series({
                "kind": d.get("kind", "unknown"),
                "num_forall": d.get("num_forall", 0),
                "num_exists": d.get("num_exists", 0),
                "num_arrows": d.get("num_arrows", 0),
                "type_string": d.get("type", "")[:100]
            })
        return pd.Series({
            "kind": "unknown",
            "num_forall": 0,
            "num_exists": 0,
            "num_arrows": 0,
            "type_string": ""
        })
    
    decl_info = df["name"].apply(get_decl_info)
    df = pd.concat([df, decl_info], axis=1)
    
    # Compute explicit metrics
    df["metric1_direct_usage"] = df["in_deg_computed"]
    df["metric2_transitive_full"] = df["transitive_full"]
    df["metric2_transitive_3hop"] = df["reach_h3"]  # For comparison
    
    # Combined metric with full transitive
    df["metric3_combined_full"] = (
        df["metric1_direct_usage"] * df["metric2_transitive_full"] / 
        (df["num_exists"] + 1)
    )
    
    # Combined metric with 3-hop (for comparison)
    df["metric3_combined_3hop"] = (
        df["metric1_direct_usage"] * df["metric2_transitive_3hop"] / 
        (df["num_exists"] + 1)
    )
    
    return df


def analyze_and_compare(df, top_n=15):
    """Analyze results and compare 3-hop vs full transitive closure."""
    
    # Filter to theorems and lemmas
    theorem_df = df[df["kind"].isin(["theorem", "lemma"])].copy()
    
    print("=" * 80)
    print("EXPLICIT METRICS WITH FULL TRANSITIVE CLOSURE")
    print("=" * 80)
    
    # Compare transitive metrics
    print("\n## TRANSITIVE CLOSURE COMPARISON")
    print("-" * 60)
    print(f"Average 3-hop reach: {theorem_df['metric2_transitive_3hop'].mean():.2f}")
    print(f"Average full transitive: {theorem_df['metric2_transitive_full'].mean():.2f}")
    print(f"Max 3-hop reach: {theorem_df['metric2_transitive_3hop'].max():.0f}")
    print(f"Max full transitive: {theorem_df['metric2_transitive_full'].max():.0f}")
    
    # Correlation between 3-hop and full
    corr = theorem_df[["metric2_transitive_3hop", "metric2_transitive_full"]].corr().iloc[0, 1]
    print(f"Correlation between 3-hop and full: {corr:.3f}")
    
    # Top by full transitive closure
    print(f"\n## TOP {top_n} BY FULL TRANSITIVE CLOSURE")
    print("-" * 60)
    top_trans = theorem_df.nlargest(top_n, "metric2_transitive_full")
    for _, row in top_trans.iterrows():
        print(f"{row['metric2_transitive_full']:4.0f} deps (3-hop: {row['metric2_transitive_3hop']:3.0f}) | {row['name']}")
    
    # Top by combined metric (full transitive)
    print(f"\n## TOP {top_n} BY COMBINED METRIC (FULL TRANSITIVE)")
    print("Formula: (direct × full_transitive) / (∃ + 1)")
    print("-" * 60)
    top_combined = theorem_df.nlargest(top_n, "metric3_combined_full")
    for _, row in top_combined.iterrows():
        print(f"{row['metric3_combined_full']:8.1f} | {row['name']}")
        print(f"           | Direct={row['metric1_direct_usage']:.0f}, Trans={row['metric2_transitive_full']:.0f}, ∃={row['num_exists']:.0f}")
    
    # Find theorems where full transitive >> 3-hop
    print("\n## THEOREMS WITH LARGE DIFFERENCE (FULL >> 3-HOP)")
    print("-" * 60)
    theorem_df["trans_diff"] = theorem_df["metric2_transitive_full"] - theorem_df["metric2_transitive_3hop"]
    large_diff = theorem_df.nlargest(10, "trans_diff")
    for _, row in large_diff.head(5).iterrows():
        print(f"{row['name']}:")
        print(f"  Full: {row['metric2_transitive_full']:.0f}, 3-hop: {row['metric2_transitive_3hop']:.0f}, Diff: {row['trans_diff']:.0f}")
    
    # Rank comparison
    print("\n## RANKING CHANGES (3-HOP vs FULL)")
    print("-" * 60)
    theorem_df["rank_3hop"] = theorem_df["metric3_combined_3hop"].rank(ascending=False, method="min")
    theorem_df["rank_full"] = theorem_df["metric3_combined_full"].rank(ascending=False, method="min")
    theorem_df["rank_change"] = theorem_df["rank_3hop"] - theorem_df["rank_full"]
    
    # Biggest rank improvements with full transitive
    print("Biggest rank improvements with full transitive:")
    improvements = theorem_df.nlargest(5, "rank_change")
    for _, row in improvements.iterrows():
        print(f"  {row['name']}: {row['rank_3hop']:.0f} → {row['rank_full']:.0f} (↑{row['rank_change']:.0f})")
    
    print("\nBiggest rank drops with full transitive:")
    drops = theorem_df.nsmallest(5, "rank_change")
    for _, row in drops.iterrows():
        print(f"  {row['name']}: {row['rank_3hop']:.0f} → {row['rank_full']:.0f} (↓{abs(row['rank_change']):.0f})")
    
    return theorem_df


def export_full_rankings(df, output_file="outputs/explicit_metrics_full_rankings.csv"):
    """Export rankings with full transitive closure."""
    
    export_df = df[df["kind"].isin(["theorem", "lemma"])][
        ["name", "kind", "metric1_direct_usage", "metric2_transitive_3hop", 
         "metric2_transitive_full", "metric3_combined_3hop", "metric3_combined_full",
         "num_exists", "num_forall"]
    ].sort_values("metric3_combined_full", ascending=False)
    
    export_df["rank_full"] = export_df["metric3_combined_full"].rank(ascending=False, method="min")
    export_df["rank_3hop"] = export_df["metric3_combined_3hop"].rank(ascending=False, method="min")
    
    Path(output_file).parent.mkdir(exist_ok=True)
    export_df.to_csv(output_file, index=False)
    print(f"\nFull rankings exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze theorems with full transitive closure")
    parser.add_argument("--data_dir", default="data/number_theory_filtered",
                       help="Directory containing processed data")
    parser.add_argument("--top_n", type=int, default=15,
                       help="Number of top theorems to display")
    parser.add_argument("--export", default="outputs/explicit_metrics_full_rankings.csv",
                       help="Output file for rankings")
    args = parser.parse_args()
    
    # Load data and compute full transitive closure
    print("Loading data and computing full transitive closure...")
    graph_metrics, nodes, decl_map, in_deg, transitive_deps = load_data(args.data_dir)
    
    # Compute metrics
    print("\nComputing explicit metrics...")
    df = compute_explicit_metrics(graph_metrics, nodes, decl_map, in_deg, transitive_deps)
    
    # Analyze and compare
    analyzed_df = analyze_and_compare(df, args.top_n)
    
    # Export
    if args.export:
        export_full_rankings(analyzed_df, args.export)


if __name__ == "__main__":
    main()