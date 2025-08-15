#!/usr/bin/env python3
"""
Analyze theorem importance using explicit, interpretable metrics.
Based on mathematician's suggestions:
1. Direct usage count (how many theorems call this one)
2. Transitive dependency count (how many theorems rely on this one)
3. Combined metric: (direct * transitive) / (existential_quantifiers + 1)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_data(data_dir="data/number_theory_filtered"):
    """Load graph metrics and declaration structures."""
    
    # Load graph metrics
    graph_metrics = pd.read_parquet(f"{data_dir}/processed/graph_metrics.parquet")
    
    # Load nodes to get theorem names
    nodes = pd.read_parquet(f"{data_dir}/processed/nodes.parquet")
    
    # Load declaration structures for quantifier counts
    declarations = []
    with open(f"{data_dir}/declaration_structures.jsonl", "r") as f:
        for line in f:
            declarations.append(json.loads(line))
    
    # Create mapping from name to declaration info
    decl_map = {d["name"]: d for d in declarations}
    
    return graph_metrics, nodes, decl_map


def compute_explicit_metrics(graph_metrics, nodes, decl_map):
    """Compute the three explicit metrics suggested by the mathematician."""
    
    # Merge graph metrics with node info
    df = graph_metrics.merge(nodes[["id", "name"]], on="id", how="left")
    
    # Add declaration info including kind
    def get_decl_info(name):
        if name in decl_map:
            d = decl_map[name]
            return pd.Series({
                "kind": d.get("kind", "unknown"),
                "num_forall": d.get("num_forall", 0),
                "num_exists": d.get("num_exists", 0),
                "num_arrows": d.get("num_arrows", 0),
                "type_string": d.get("type", "")[:100]  # First 100 chars
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
    # Metric 1: Direct usage (already have as in_deg)
    df["metric1_direct_usage"] = df["in_deg"]
    
    # Metric 2: Transitive dependencies (using reach_h3 as approximation)
    # Note: reach_h3 is 3-hop reachability, not full transitive closure
    df["metric2_transitive_deps"] = df["reach_h3"]
    
    # Metric 3: Combined weighted metric
    # (direct_uses * transitive_deps) / (existential_quantifiers + 1)
    df["metric3_combined"] = (
        df["metric1_direct_usage"] * df["metric2_transitive_deps"] / 
        (df["num_exists"] + 1)
    )
    
    return df


def analyze_top_theorems(df, top_n=20):
    """Analyze and display top theorems by each metric."""
    
    # Filter to only theorems and lemmas
    theorem_df = df[df["kind"].isin(["theorem", "lemma"])].copy()
    
    print("=" * 80)
    print("EXPLICIT METRIC ANALYSIS FOR NUMBER THEORY THEOREMS")
    print("=" * 80)
    
    # Metric 1: Direct Usage
    print("\n## METRIC 1: DIRECT USAGE COUNT")
    print("(Number of theorems that directly call this theorem)")
    print("-" * 60)
    
    top_direct = theorem_df.nlargest(top_n, "metric1_direct_usage")
    for idx, row in top_direct.iterrows():
        print(f"{row['metric1_direct_usage']:3.0f} uses | {row['name']}")
        if row['type_string']:
            print(f"         | Type: {row['type_string'][:70]}...")
    
    # Metric 2: Transitive Dependencies
    print("\n## METRIC 2: TRANSITIVE DEPENDENCIES (3-HOP)")
    print("(Number of theorems reachable within 3 dependency hops)")
    print("-" * 60)
    
    top_transitive = theorem_df.nlargest(top_n, "metric2_transitive_deps")
    for idx, row in top_transitive.iterrows():
        print(f"{row['metric2_transitive_deps']:3.0f} deps | {row['name']}")
        
    # Metric 3: Combined Weighted
    print("\n## METRIC 3: COMBINED WEIGHTED SCORE")
    print("Formula: (direct_uses × transitive_deps) / (existential_quantifiers + 1)")
    print("-" * 60)
    
    top_combined = theorem_df.nlargest(top_n, "metric3_combined")
    for idx, row in top_combined.iterrows():
        print(f"{row['metric3_combined']:7.1f} | {row['name']}")
        print(f"         | Direct={row['metric1_direct_usage']:.0f}, Trans={row['metric2_transitive_deps']:.0f}, ∃={row['num_exists']:.0f}")
    
    # Summary statistics
    print("\n## SUMMARY STATISTICS")
    print("-" * 60)
    
    print(f"Total theorems/lemmas analyzed: {len(theorem_df)}")
    print(f"\nDirect usage (Metric 1):")
    print(f"  Mean: {theorem_df['metric1_direct_usage'].mean():.2f}")
    print(f"  Median: {theorem_df['metric1_direct_usage'].median():.0f}")
    print(f"  Max: {theorem_df['metric1_direct_usage'].max():.0f}")
    print(f"  % with 0 uses: {(theorem_df['metric1_direct_usage'] == 0).mean()*100:.1f}%")
    
    print(f"\nTransitive deps (Metric 2):")
    print(f"  Mean: {theorem_df['metric2_transitive_deps'].mean():.2f}")
    print(f"  Median: {theorem_df['metric2_transitive_deps'].median():.0f}")
    print(f"  Max: {theorem_df['metric2_transitive_deps'].max():.0f}")
    
    print(f"\nExistential quantifiers:")
    print(f"  % with ∃: {(theorem_df['num_exists'] > 0).mean()*100:.1f}%")
    print(f"  Mean (when > 0): {theorem_df[theorem_df['num_exists'] > 0]['num_exists'].mean():.2f}")
    
    # Correlation analysis
    print("\n## METRIC CORRELATIONS")
    print("-" * 60)
    
    metrics = ["metric1_direct_usage", "metric2_transitive_deps", "metric3_combined"]
    corr_matrix = theorem_df[metrics].corr()
    
    print("Correlation matrix:")
    print(corr_matrix.round(3))
    
    # Find theorems that rank very differently across metrics
    print("\n## INTERESTING DIVERGENCES")
    print("-" * 60)
    
    # High direct usage but low transitive
    high_direct_low_trans = theorem_df[
        (theorem_df["metric1_direct_usage"] > theorem_df["metric1_direct_usage"].quantile(0.9)) &
        (theorem_df["metric2_transitive_deps"] < theorem_df["metric2_transitive_deps"].quantile(0.5))
    ]
    
    if len(high_direct_low_trans) > 0:
        print("\nHigh direct usage but low transitive reach:")
        for idx, row in high_direct_low_trans.head(5).iterrows():
            print(f"  {row['name']}: direct={row['metric1_direct_usage']:.0f}, trans={row['metric2_transitive_deps']:.0f}")
    
    # Low direct usage but high transitive
    low_direct_high_trans = theorem_df[
        (theorem_df["metric1_direct_usage"] < theorem_df["metric1_direct_usage"].quantile(0.5)) &
        (theorem_df["metric2_transitive_deps"] > theorem_df["metric2_transitive_deps"].quantile(0.9))
    ]
    
    if len(low_direct_high_trans) > 0:
        print("\nLow direct usage but high transitive reach:")
        for idx, row in low_direct_high_trans.head(5).iterrows():
            print(f"  {row['name']}: direct={row['metric1_direct_usage']:.0f}, trans={row['metric2_transitive_deps']:.0f}")
    
    return theorem_df


def export_rankings(df, output_file="outputs/explicit_metrics_rankings.csv"):
    """Export rankings to CSV for further analysis."""
    
    # Select relevant columns and sort by combined metric
    export_df = df[df["kind"].isin(["theorem", "lemma"])][
        ["name", "kind", "metric1_direct_usage", "metric2_transitive_deps", 
         "metric3_combined", "num_exists", "num_forall"]
    ].sort_values("metric3_combined", ascending=False)
    
    # Add rank columns
    export_df["rank_direct"] = export_df["metric1_direct_usage"].rank(ascending=False, method="min")
    export_df["rank_transitive"] = export_df["metric2_transitive_deps"].rank(ascending=False, method="min")
    export_df["rank_combined"] = export_df["metric3_combined"].rank(ascending=False, method="min")
    
    # Save to CSV
    Path(output_file).parent.mkdir(exist_ok=True)
    export_df.to_csv(output_file, index=False)
    print(f"\nRankings exported to: {output_file}")
    
    return export_df


def main():
    parser = argparse.ArgumentParser(description="Analyze theorems using explicit metrics")
    parser.add_argument("--data_dir", default="data/number_theory_filtered",
                       help="Directory containing processed data")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top theorems to display")
    parser.add_argument("--export", default="outputs/explicit_metrics_rankings.csv",
                       help="Output file for rankings")
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    graph_metrics, nodes, decl_map = load_data(args.data_dir)
    
    # Compute explicit metrics
    print("Computing explicit metrics...")
    df = compute_explicit_metrics(graph_metrics, nodes, decl_map)
    
    # Analyze top theorems
    analyzed_df = analyze_top_theorems(df, args.top_n)
    
    # Export rankings
    if args.export:
        export_rankings(analyzed_df, args.export)


if __name__ == "__main__":
    main()