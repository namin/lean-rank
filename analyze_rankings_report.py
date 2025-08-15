#!/usr/bin/env python3
"""
Generate a comprehensive sorted report from the explicit metrics rankings CSV.
Displays theorems sorted by each metric with detailed analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_rankings(csv_file):
    """Load rankings CSV file."""
    df = pd.read_csv(csv_file)
    # Filter to only theorems and lemmas
    df = df[df["kind"].isin(["theorem", "lemma"])].copy()
    return df


def print_section_header(title, width=80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title, width=60):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * width)


def analyze_metric1_direct_usage(df, top_n=20):
    """Analyze and display Metric 1: Direct Usage Count."""
    print_section_header("METRIC 1: DIRECT USAGE COUNT")
    print("Number of theorems that directly call this theorem")
    
    # Top theorems
    print_subsection(f"Top {top_n} Theorems by Direct Usage")
    top_df = df.nlargest(top_n, "metric1_direct_usage")
    
    for idx, row in top_df.iterrows():
        print(f"{row['metric1_direct_usage']:3.0f} uses | {row['name']}")
    
    # Statistics
    print_subsection("Statistics")
    metric = df["metric1_direct_usage"]
    print(f"Mean:     {metric.mean():.2f}")
    print(f"Median:   {metric.median():.0f}")
    print(f"Std Dev:  {metric.std():.2f}")
    print(f"Max:      {metric.max():.0f}")
    print(f"Min:      {metric.min():.0f}")
    print(f"Zero usage: {(metric == 0).sum()} theorems ({(metric == 0).mean()*100:.1f}%)")
    
    # Distribution
    print_subsection("Usage Distribution")
    bins = [0, 1, 2, 3, 5, 10, 20, np.inf]
    labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]
    df["usage_bin"] = pd.cut(metric, bins=bins, labels=labels, right=False)
    dist = df["usage_bin"].value_counts().sort_index()
    
    for bin_label, count in dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"{bin_label:6} uses: {count:4} ({pct:5.1f}%) {bar}")


def analyze_metric2_transitive(df, top_n=20):
    """Analyze and display Metric 2: Transitive Dependencies."""
    print_section_header("METRIC 2: TRANSITIVE DEPENDENCIES (FULL CLOSURE)")
    print("Number of theorems that depend on this one (directly or indirectly)")
    
    # Top theorems
    print_subsection(f"Top {top_n} Theorems by Transitive Dependencies")
    top_df = df.nlargest(top_n, "metric2_transitive_full")
    
    for idx, row in top_df.iterrows():
        diff = row['metric2_transitive_full'] - row['metric2_transitive_3hop']
        print(f"{row['metric2_transitive_full']:4.0f} deps | {row['name']}")
        if diff > 50:
            print(f"          | (3-hop: {row['metric2_transitive_3hop']:3.0f}, diff: +{diff:.0f})")
    
    # Statistics
    print_subsection("Statistics")
    metric_full = df["metric2_transitive_full"]
    metric_3hop = df["metric2_transitive_3hop"]
    
    print(f"Full Closure:")
    print(f"  Mean:     {metric_full.mean():.2f}")
    print(f"  Median:   {metric_full.median():.0f}")
    print(f"  Max:      {metric_full.max():.0f}")
    print(f"\n3-Hop Approximation:")
    print(f"  Mean:     {metric_3hop.mean():.2f}")
    print(f"  Median:   {metric_3hop.median():.0f}")
    print(f"  Max:      {metric_3hop.max():.0f}")
    
    print(f"\nCorrelation (3-hop vs full): {metric_full.corr(metric_3hop):.3f}")
    
    # Biggest differences
    print_subsection("Largest Underestimations by 3-Hop")
    df["trans_diff"] = df["metric2_transitive_full"] - df["metric2_transitive_3hop"]
    large_diff = df.nlargest(10, "trans_diff")
    
    for idx, row in large_diff.iterrows():
        ratio = row['metric2_transitive_full'] / max(row['metric2_transitive_3hop'], 1)
        print(f"{row['name']:50} | 3-hop: {row['metric2_transitive_3hop']:3.0f} → full: {row['metric2_transitive_full']:3.0f} ({ratio:.1f}x)")


def analyze_metric3_combined(df, top_n=20):
    """Analyze and display Metric 3: Combined Weighted Score."""
    print_section_header("METRIC 3: COMBINED WEIGHTED SCORE")
    print("Formula: (direct_usage × transitive_deps) / (existential_quantifiers + 1)")
    
    # Top theorems
    print_subsection(f"Top {top_n} Theorems by Combined Score (Full Transitive)")
    top_df = df.nlargest(top_n, "metric3_combined_full")
    
    for idx, row in top_df.iterrows():
        print(f"{row['metric3_combined_full']:8.1f} | {row['name']}")
        components = f"D={row['metric1_direct_usage']:.0f}, T={row['metric2_transitive_full']:.0f}"
        if row['num_exists'] > 0:
            components += f", ∃={row['num_exists']:.0f}"
        print(f"           | {components}")
    
    # Compare with 3-hop version
    print_subsection("Ranking Changes: 3-Hop → Full Transitive")
    
    # Calculate rank changes
    df["rank_change"] = df["rank_3hop"] - df["rank_full"]
    
    print("\nBiggest Rank Improvements (better with full transitive):")
    improvements = df.nlargest(10, "rank_change")
    for idx, row in improvements.head(5).iterrows():
        arrow = "↑" if row["rank_change"] > 0 else "↓"
        print(f"  {row['name']:45} | #{row['rank_3hop']:.0f} → #{row['rank_full']:.0f} ({arrow}{abs(row['rank_change']):.0f})")
    
    print("\nBiggest Rank Drops (worse with full transitive):")
    drops = df.nsmallest(5, "rank_change")
    for idx, row in drops.iterrows():
        arrow = "↓" 
        print(f"  {row['name']:45} | #{row['rank_3hop']:.0f} → #{row['rank_full']:.0f} ({arrow}{abs(row['rank_change']):.0f})")
    
    # Stability analysis
    print_subsection("Ranking Stability Analysis")
    small_changes = (df["rank_change"].abs() <= 5).mean() * 100
    medium_changes = ((df["rank_change"].abs() > 5) & (df["rank_change"].abs() <= 20)).mean() * 100
    large_changes = (df["rank_change"].abs() > 20).mean() * 100
    
    print(f"Stable (≤5 rank change):    {small_changes:.1f}%")
    print(f"Moderate (6-20 rank change): {medium_changes:.1f}%")
    print(f"Volatile (>20 rank change):  {large_changes:.1f}%")


def analyze_quantifiers(df):
    """Analyze the role of existential quantifiers."""
    print_section_header("EXISTENTIAL QUANTIFIER ANALYSIS")
    print("How ∃ quantifiers affect theorem importance")
    
    # Statistics
    print_subsection("Quantifier Statistics")
    has_exists = df[df["num_exists"] > 0]
    no_exists = df[df["num_exists"] == 0]
    
    print(f"Theorems with ∃: {len(has_exists)} ({len(has_exists)/len(df)*100:.1f}%)")
    print(f"Theorems without ∃: {len(no_exists)} ({len(no_exists)/len(df)*100:.1f}%)")
    
    if len(has_exists) > 0:
        print(f"\nFor theorems with ∃:")
        print(f"  Mean ∃ count: {has_exists['num_exists'].mean():.2f}")
        print(f"  Max ∃ count: {has_exists['num_exists'].max():.0f}")
    
    # Impact on metrics
    print_subsection("Impact on Metrics")
    
    if len(has_exists) > 0 and len(no_exists) > 0:
        print("\nAverage Direct Usage:")
        print(f"  With ∃: {has_exists['metric1_direct_usage'].mean():.2f}")
        print(f"  Without ∃: {no_exists['metric1_direct_usage'].mean():.2f}")
        
        print("\nAverage Transitive Dependencies:")
        print(f"  With ∃: {has_exists['metric2_transitive_full'].mean():.2f}")
        print(f"  Without ∃: {no_exists['metric2_transitive_full'].mean():.2f}")
        
        print("\nAverage Combined Score:")
        print(f"  With ∃: {has_exists['metric3_combined_full'].mean():.2f}")
        print(f"  Without ∃: {no_exists['metric3_combined_full'].mean():.2f}")
    
    # Top theorems with existential quantifiers
    if len(has_exists) > 0:
        print_subsection("Top Theorems with ∃ Quantifiers")
        top_exists = has_exists.nlargest(10, "metric3_combined_full")
        for idx, row in top_exists.iterrows():
            print(f"∃={row['num_exists']:.0f} | Score={row['metric3_combined_full']:7.1f} | {row['name']}")


def analyze_correlations(df):
    """Analyze correlations between metrics."""
    print_section_header("METRIC CORRELATIONS")
    
    metrics = [
        "metric1_direct_usage",
        "metric2_transitive_3hop", 
        "metric2_transitive_full",
        "metric3_combined_3hop",
        "metric3_combined_full",
        "num_exists",
        "num_forall"
    ]
    
    # Select available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    corr_matrix = df[available_metrics].corr()
    
    print_subsection("Correlation Matrix")
    
    # Simplified labels for display
    labels = {
        "metric1_direct_usage": "Direct",
        "metric2_transitive_3hop": "Trans-3hop",
        "metric2_transitive_full": "Trans-Full",
        "metric3_combined_3hop": "Comb-3hop",
        "metric3_combined_full": "Comb-Full",
        "num_exists": "∃",
        "num_forall": "∀"
    }
    
    # Print header
    print("            ", end="")
    for col in available_metrics:
        print(f"{labels.get(col, col)[:10]:>11}", end="")
    print()
    
    # Print rows
    for row in available_metrics:
        print(f"{labels.get(row, row)[:10]:>11}", end="")
        for col in available_metrics:
            val = corr_matrix.loc[row, col]
            if row == col:
                print(f"{'1.000':>11}", end="")
            else:
                print(f"{val:>11.3f}", end="")
        print()
    
    # Key insights
    print_subsection("Key Correlation Insights")
    
    # Find strongest correlations (excluding diagonal)
    corr_values = []
    for i, row in enumerate(available_metrics):
        for j, col in enumerate(available_metrics):
            if i < j:  # Upper triangle only
                corr_values.append((abs(corr_matrix.loc[row, col]), row, col, corr_matrix.loc[row, col]))
    
    corr_values.sort(reverse=True)
    
    print("\nStrongest Correlations:")
    for abs_corr, row, col, corr in corr_values[:5]:
        sign = "+" if corr > 0 else "-"
        print(f"  {labels.get(row, row)} ↔ {labels.get(col, col)}: {sign}{abs_corr:.3f}")
    
    print("\nWeakest Correlations:")
    for abs_corr, row, col, corr in corr_values[-3:]:
        sign = "+" if corr > 0 else "-"
        print(f"  {labels.get(row, row)} ↔ {labels.get(col, col)}: {sign}{abs_corr:.3f}")


def generate_summary(df):
    """Generate executive summary."""
    print_section_header("EXECUTIVE SUMMARY")
    
    print(f"Dataset: {len(df)} theorems and lemmas analyzed")
    print(f"Metrics: 3 explicit, interpretable importance measures")
    
    # Top theorem by each metric
    print_subsection("Champions by Each Metric")
    
    top_direct = df.nlargest(1, "metric1_direct_usage").iloc[0]
    print(f"Most Directly Used: {top_direct['name']}")
    print(f"  → {top_direct['metric1_direct_usage']:.0f} direct uses")
    
    top_trans = df.nlargest(1, "metric2_transitive_full").iloc[0]
    print(f"\nMost Dependencies: {top_trans['name']}")
    print(f"  → {top_trans['metric2_transitive_full']:.0f} theorems depend on it")
    
    top_combined = df.nlargest(1, "metric3_combined_full").iloc[0]
    print(f"\nHighest Combined Score: {top_combined['name']}")
    print(f"  → Score: {top_combined['metric3_combined_full']:.1f}")
    
    # Key findings
    print_subsection("Key Findings")
    
    print(f"• Full transitive closure reveals {df['metric2_transitive_full'].mean() / df['metric2_transitive_3hop'].mean():.1f}x more dependencies than 3-hop")
    print(f"• {(df['metric1_direct_usage'] == 0).mean()*100:.1f}% of theorems have zero direct usage")
    print(f"• Only {(df['num_exists'] > 0).mean()*100:.1f}% of theorems contain existential quantifiers")
    print(f"• Direct usage and transitive reach correlation: {df['metric1_direct_usage'].corr(df['metric2_transitive_full']):.3f}")
    
    # Recommendation
    print_subsection("Recommendation")
    print("The combined metric (Metric 3) with full transitive closure provides")
    print("the most comprehensive importance ranking, balancing immediate utility")
    print("(direct usage) with foundational importance (transitive dependencies)")
    print("while accounting for statement complexity (existential quantifiers).")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive report from rankings CSV")
    parser.add_argument("--input", default="outputs/explicit_metrics_full_rankings.csv",
                       help="Input CSV file with rankings")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top items to show in each category")
    parser.add_argument("--sections", nargs="+", 
                       choices=["summary", "metric1", "metric2", "metric3", "quantifiers", "correlations", "all"],
                       default=["all"],
                       help="Which sections to include in report")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                       help="Output format (text or markdown)")
    parser.add_argument("--output", help="Output file (if not specified, prints to stdout)")
    args = parser.parse_args()
    
    # Load data
    df = load_rankings(args.input)
    
    sections = args.sections
    if "all" in sections:
        sections = ["summary", "metric1", "metric2", "metric3", "quantifiers", "correlations"]
    
    # Generate report sections
    if "summary" in sections:
        generate_summary(df)
    if "metric1" in sections:
        analyze_metric1_direct_usage(df, args.top_n)
    if "metric2" in sections:
        analyze_metric2_transitive(df, args.top_n)
    if "metric3" in sections:
        analyze_metric3_combined(df, args.top_n)
    if "quantifiers" in sections:
        analyze_quantifiers(df)
    if "correlations" in sections:
        analyze_correlations(df)
    
    print("\n" + "=" * 80)
    print("  END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()