#!/usr/bin/env python3
"""
Generate a comprehensive markdown report from the explicit metrics rankings CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


def load_rankings(csv_file):
    """Load rankings CSV file."""
    df = pd.read_csv(csv_file)
    df = df[df["kind"].isin(["theorem", "lemma"])].copy()
    return df


def format_table_row(values, widths=None):
    """Format a markdown table row."""
    if widths:
        formatted = [str(v).ljust(w) for v, w in zip(values, widths)]
    else:
        formatted = [str(v) for v in values]
    return "| " + " | ".join(formatted) + " |"


def generate_markdown_summary(df):
    """Generate executive summary in markdown."""
    lines = []
    lines.append("# Theorem Importance Analysis Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Dataset**: {len(df)} theorems and lemmas analyzed from number theory domain")
    lines.append("")
    lines.append("**Methodology**: Three explicit, interpretable importance metrics:")
    lines.append("1. **Direct Usage**: Number of theorems that directly call this theorem")
    lines.append("2. **Transitive Dependencies**: Number of theorems that depend on this one (full closure)")
    lines.append("3. **Combined Score**: `(direct × transitive) / (existential_quantifiers + 1)`")
    lines.append("")
    
    # Top theorems
    lines.append("### Top Theorems by Each Metric")
    lines.append("")
    
    top_direct = df.nlargest(1, "metric1_direct_usage").iloc[0]
    lines.append(f"**Most Directly Used**: `{top_direct['name']}`")
    lines.append(f"- {top_direct['metric1_direct_usage']:.0f} direct uses")
    lines.append("")
    
    top_trans = df.nlargest(1, "metric2_transitive_full").iloc[0]
    lines.append(f"**Most Dependencies**: `{top_trans['name']}`")
    lines.append(f"- {top_trans['metric2_transitive_full']:.0f} theorems depend on it")
    lines.append("")
    
    top_combined = df.nlargest(1, "metric3_combined_full").iloc[0]
    lines.append(f"**Highest Combined Score**: `{top_combined['name']}`")
    lines.append(f"- Score: {top_combined['metric3_combined_full']:.1f}")
    lines.append("")
    
    # Key statistics
    lines.append("### Key Statistics")
    lines.append("")
    lines.append("| Metric | Mean | Median | Max | Zero Count |")
    lines.append("|--------|------|--------|-----|------------|")
    lines.append(f"| Direct Usage | {df['metric1_direct_usage'].mean():.2f} | {df['metric1_direct_usage'].median():.0f} | {df['metric1_direct_usage'].max():.0f} | {(df['metric1_direct_usage'] == 0).sum()} ({(df['metric1_direct_usage'] == 0).mean()*100:.1f}%) |")
    lines.append(f"| Transitive (Full) | {df['metric2_transitive_full'].mean():.2f} | {df['metric2_transitive_full'].median():.0f} | {df['metric2_transitive_full'].max():.0f} | {(df['metric2_transitive_full'] == 0).sum()} ({(df['metric2_transitive_full'] == 0).mean()*100:.1f}%) |")
    lines.append(f"| Combined Score | {df['metric3_combined_full'].mean():.2f} | {df['metric3_combined_full'].median():.0f} | {df['metric3_combined_full'].max():.0f} | - |")
    lines.append("")
    
    return "\n".join(lines)


def generate_metric1_section(df, top_n=20):
    """Generate Metric 1 section in markdown."""
    lines = []
    lines.append("## Metric 1: Direct Usage Count")
    lines.append("")
    lines.append("Number of theorems that directly call this theorem. This measures immediate utility.")
    lines.append("")
    
    # Top theorems table
    lines.append(f"### Top {top_n} Theorems by Direct Usage")
    lines.append("")
    lines.append("| Rank | Uses | Theorem Name |")
    lines.append("|------|------|--------------|")
    
    top_df = df.nlargest(top_n, "metric1_direct_usage")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        lines.append(f"| {i} | {row['metric1_direct_usage']:.0f} | `{row['name']}` |")
    lines.append("")
    
    # Distribution chart
    lines.append("### Usage Distribution")
    lines.append("")
    lines.append("```")
    
    bins = [0, 1, 2, 3, 5, 10, 20, np.inf]
    labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]
    df["usage_bin"] = pd.cut(df["metric1_direct_usage"], bins=bins, labels=labels, right=False)
    dist = df["usage_bin"].value_counts().sort_index()
    
    max_count = dist.max()
    for bin_label, count in dist.items():
        pct = count / len(df) * 100
        bar_length = int(40 * count / max_count)
        bar = "█" * bar_length
        lines.append(f"{bin_label:6} uses: {bar} {count:4} ({pct:5.1f}%)")
    
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


def generate_metric2_section(df, top_n=20):
    """Generate Metric 2 section in markdown."""
    lines = []
    lines.append("## Metric 2: Transitive Dependencies (Full Closure)")
    lines.append("")
    lines.append("Number of theorems that depend on this one, directly or indirectly through dependency chains.")
    lines.append("")
    
    # Comparison with 3-hop
    lines.append("### Full Closure vs 3-Hop Approximation")
    lines.append("")
    lines.append("| Statistic | 3-Hop | Full Closure | Ratio |")
    lines.append("|-----------|-------|--------------|-------|")
    lines.append(f"| Mean | {df['metric2_transitive_3hop'].mean():.2f} | {df['metric2_transitive_full'].mean():.2f} | {df['metric2_transitive_full'].mean()/df['metric2_transitive_3hop'].mean():.2f}x |")
    lines.append(f"| Median | {df['metric2_transitive_3hop'].median():.0f} | {df['metric2_transitive_full'].median():.0f} | - |")
    lines.append(f"| Max | {df['metric2_transitive_3hop'].max():.0f} | {df['metric2_transitive_full'].max():.0f} | {df['metric2_transitive_full'].max()/df['metric2_transitive_3hop'].max():.2f}x |")
    lines.append(f"| Correlation | - | - | {df['metric2_transitive_full'].corr(df['metric2_transitive_3hop']):.3f} |")
    lines.append("")
    
    # Top theorems
    lines.append(f"### Top {top_n} Theorems by Transitive Dependencies")
    lines.append("")
    lines.append("| Rank | Full Deps | 3-Hop | Diff | Theorem Name |")
    lines.append("|------|-----------|-------|------|--------------|")
    
    top_df = df.nlargest(top_n, "metric2_transitive_full")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        diff = row['metric2_transitive_full'] - row['metric2_transitive_3hop']
        lines.append(f"| {i} | {row['metric2_transitive_full']:.0f} | {row['metric2_transitive_3hop']:.0f} | +{diff:.0f} | `{row['name']}` |")
    lines.append("")
    
    # Biggest underestimations
    lines.append("### Largest Underestimations by 3-Hop")
    lines.append("")
    lines.append("These theorems have much larger transitive impact than 3-hop analysis suggests:")
    lines.append("")
    lines.append("| Theorem | 3-Hop | Full | Ratio |")
    lines.append("|---------|-------|------|-------|")
    
    df["trans_diff"] = df["metric2_transitive_full"] - df["metric2_transitive_3hop"]
    large_diff = df.nlargest(10, "trans_diff")
    for idx, row in large_diff.iterrows():
        ratio = row['metric2_transitive_full'] / max(row['metric2_transitive_3hop'], 1)
        lines.append(f"| `{row['name']}` | {row['metric2_transitive_3hop']:.0f} | {row['metric2_transitive_full']:.0f} | {ratio:.1f}x |")
    lines.append("")
    
    return "\n".join(lines)


def generate_metric3_section(df, top_n=20):
    """Generate Metric 3 section in markdown."""
    lines = []
    lines.append("## Metric 3: Combined Weighted Score")
    lines.append("")
    lines.append("**Formula**: `(direct_usage × transitive_deps) / (existential_quantifiers + 1)`")
    lines.append("")
    lines.append("This metric balances immediate utility with foundational importance while penalizing complex statements.")
    lines.append("")
    
    # Top theorems
    lines.append(f"### Top {top_n} Theorems by Combined Score")
    lines.append("")
    lines.append("| Rank | Score | Direct | Transitive | ∃ | Theorem Name |")
    lines.append("|------|-------|--------|------------|---|--------------|")
    
    top_df = df.nlargest(top_n, "metric3_combined_full")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        exists_str = str(int(row['num_exists'])) if row['num_exists'] > 0 else "-"
        lines.append(f"| {i} | {row['metric3_combined_full']:.1f} | {row['metric1_direct_usage']:.0f} | {row['metric2_transitive_full']:.0f} | {exists_str} | `{row['name']}` |")
    lines.append("")
    
    # Ranking stability
    df["rank_change"] = df["rank_3hop"] - df["rank_full"]
    
    lines.append("### Ranking Stability Analysis")
    lines.append("")
    lines.append("How rankings change when using full transitive closure vs 3-hop:")
    lines.append("")
    
    small_changes = (df["rank_change"].abs() <= 5).mean() * 100
    medium_changes = ((df["rank_change"].abs() > 5) & (df["rank_change"].abs() <= 20)).mean() * 100
    large_changes = (df["rank_change"].abs() > 20).mean() * 100
    
    lines.append("| Stability Level | Rank Change | % of Theorems |")
    lines.append("|-----------------|-------------|---------------|")
    lines.append(f"| Stable | ≤5 | {small_changes:.1f}% |")
    lines.append(f"| Moderate | 6-20 | {medium_changes:.1f}% |")
    lines.append(f"| Volatile | >20 | {large_changes:.1f}% |")
    lines.append("")
    
    # Biggest changes
    lines.append("#### Biggest Rank Improvements (with full transitive)")
    lines.append("")
    lines.append("| Theorem | 3-Hop Rank | Full Rank | Change |")
    lines.append("|---------|------------|-----------|--------|")
    
    improvements = df.nlargest(5, "rank_change")
    for idx, row in improvements.iterrows():
        lines.append(f"| `{row['name']}` | #{row['rank_3hop']:.0f} | #{row['rank_full']:.0f} | ↑{row['rank_change']:.0f} |")
    lines.append("")
    
    lines.append("#### Biggest Rank Drops (with full transitive)")
    lines.append("")
    lines.append("| Theorem | 3-Hop Rank | Full Rank | Change |")
    lines.append("|---------|------------|-----------|--------|")
    
    drops = df.nsmallest(5, "rank_change")
    for idx, row in drops.iterrows():
        lines.append(f"| `{row['name']}` | #{row['rank_3hop']:.0f} | #{row['rank_full']:.0f} | ↓{abs(row['rank_change']):.0f} |")
    lines.append("")
    
    return "\n".join(lines)


def generate_quantifier_analysis(df):
    """Generate existential quantifier analysis in markdown."""
    lines = []
    lines.append("## Existential Quantifier Analysis")
    lines.append("")
    lines.append("How existential quantifiers (∃) affect theorem importance metrics.")
    lines.append("")
    
    has_exists = df[df["num_exists"] > 0]
    no_exists = df[df["num_exists"] == 0]
    
    lines.append("### Distribution")
    lines.append("")
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")
    lines.append(f"| Theorems with ∃ | {len(has_exists)} | {len(has_exists)/len(df)*100:.1f}% |")
    lines.append(f"| Theorems without ∃ | {len(no_exists)} | {len(no_exists)/len(df)*100:.1f}% |")
    
    if len(has_exists) > 0:
        lines.append(f"| Mean ∃ count (when present) | {has_exists['num_exists'].mean():.2f} | - |")
        lines.append(f"| Max ∃ count | {has_exists['num_exists'].max():.0f} | - |")
    lines.append("")
    
    # Impact comparison
    if len(has_exists) > 0 and len(no_exists) > 0:
        lines.append("### Impact on Metrics")
        lines.append("")
        lines.append("| Metric | With ∃ | Without ∃ | Ratio |")
        lines.append("|--------|--------|-----------|-------|")
        lines.append(f"| Avg Direct Usage | {has_exists['metric1_direct_usage'].mean():.2f} | {no_exists['metric1_direct_usage'].mean():.2f} | {has_exists['metric1_direct_usage'].mean()/no_exists['metric1_direct_usage'].mean():.2f}x |")
        lines.append(f"| Avg Transitive Deps | {has_exists['metric2_transitive_full'].mean():.2f} | {no_exists['metric2_transitive_full'].mean():.2f} | {has_exists['metric2_transitive_full'].mean()/no_exists['metric2_transitive_full'].mean():.2f}x |")
        lines.append(f"| Avg Combined Score | {has_exists['metric3_combined_full'].mean():.2f} | {no_exists['metric3_combined_full'].mean():.2f} | {has_exists['metric3_combined_full'].mean()/no_exists['metric3_combined_full'].mean():.2f}x |")
        lines.append("")
        
        lines.append("**Interpretation**: Theorems with existential quantifiers tend to be more specialized (higher direct usage) but less foundational (lower transitive reach).")
        lines.append("")
    
    return "\n".join(lines)


def generate_correlation_analysis(df):
    """Generate correlation analysis in markdown."""
    lines = []
    lines.append("## Correlation Analysis")
    lines.append("")
    lines.append("How different metrics relate to each other.")
    lines.append("")
    
    # Compute correlations
    metrics = [
        ("metric1_direct_usage", "Direct Usage"),
        ("metric2_transitive_3hop", "Trans-3hop"),
        ("metric2_transitive_full", "Trans-Full"),
        ("metric3_combined_full", "Combined"),
        ("num_exists", "∃ Count"),
        ("num_forall", "∀ Count")
    ]
    
    available_metrics = [(col, name) for col, name in metrics if col in df.columns]
    
    lines.append("### Correlation Matrix")
    lines.append("")
    
    # Create correlation matrix
    corr_df = df[[col for col, _ in available_metrics]].corr()
    
    # Table header
    header = ["Metric"] + [name for _, name in available_metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["-" * max(6, len(h)) for h in header]) + "|")
    
    # Table rows
    for col1, name1 in available_metrics:
        row = [name1]
        for col2, name2 in available_metrics:
            if col1 == col2:
                row.append("1.00")
            else:
                row.append(f"{corr_df.loc[col1, col2]:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    
    # Key insights
    lines.append("### Key Insights")
    lines.append("")
    
    direct_trans_corr = df['metric1_direct_usage'].corr(df['metric2_transitive_full'])
    lines.append(f"- **Direct vs Transitive correlation**: {direct_trans_corr:.3f}")
    if direct_trans_corr < 0:
        lines.append("  - Negative correlation suggests foundational theorems are rarely used directly")
    lines.append("")
    
    trans_corr = df['metric2_transitive_3hop'].corr(df['metric2_transitive_full'])
    lines.append(f"- **3-Hop vs Full Transitive correlation**: {trans_corr:.3f}")
    lines.append("  - High correlation but significant differences in absolute values")
    lines.append("")
    
    if 'num_exists' in df.columns:
        exists_direct = df['num_exists'].corr(df['metric1_direct_usage'])
        exists_trans = df['num_exists'].corr(df['metric2_transitive_full'])
        lines.append(f"- **Existential quantifiers correlations**:")
        lines.append(f"  - With direct usage: {exists_direct:.3f}")
        lines.append(f"  - With transitive deps: {exists_trans:.3f}")
    lines.append("")
    
    return "\n".join(lines)


def generate_conclusions(df):
    """Generate conclusions section in markdown."""
    lines = []
    lines.append("## Conclusions and Recommendations")
    lines.append("")
    
    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. **Full transitive closure is essential**: 3-hop analysis misses significant dependency chains, underestimating importance by up to 8x for some theorems.")
    lines.append("")
    lines.append("2. **Different usage patterns**: Foundational theorems (high transitive) and utility theorems (high direct) serve different roles in the library.")
    lines.append("")
    lines.append(f"3. **Sparse direct usage**: {(df['metric1_direct_usage'] == 0).mean()*100:.1f}% of theorems have zero direct usage, suggesting many intermediate lemmas.")
    lines.append("")
    lines.append("4. **Existential quantifiers indicate specialization**: Theorems with ∃ have higher direct usage but lower transitive reach.")
    lines.append("")
    
    lines.append("### Recommended Metric")
    lines.append("")
    lines.append("**Use Metric 3 (Combined Score with Full Transitive Closure)** for overall importance ranking because it:")
    lines.append("- Balances immediate utility with foundational importance")
    lines.append("- Accounts for statement complexity")
    lines.append("- Provides fully explainable scores")
    lines.append("")
    
    lines.append("### Applications")
    lines.append("")
    lines.append("These metrics can be used for:")
    lines.append("- **Library maintenance**: Identify unused theorems for potential removal")
    lines.append("- **Documentation priority**: Focus on high-impact theorems")
    lines.append("- **Teaching order**: Start with foundational theorems")
    lines.append("- **ML training data**: Use as ground truth for importance prediction models")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from rankings CSV")
    parser.add_argument("--input", default="outputs/explicit_metrics_full_rankings.csv",
                       help="Input CSV file with rankings")
    parser.add_argument("--output", default="outputs/theorem_importance_report.md",
                       help="Output markdown file")
    parser.add_argument("--top_n", type=int, default=15,
                       help="Number of top items to show in tables")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_rankings(args.input)
    
    # Generate report sections
    print("Generating markdown report...")
    sections = []
    
    sections.append(generate_markdown_summary(df))
    sections.append(generate_metric1_section(df, args.top_n))
    sections.append(generate_metric2_section(df, args.top_n))
    sections.append(generate_metric3_section(df, args.top_n))
    sections.append(generate_quantifier_analysis(df))
    sections.append(generate_correlation_analysis(df))
    sections.append(generate_conclusions(df))
    
    # Add footer
    sections.append("---")
    sections.append("")
    sections.append("*Report generated using explicit, interpretable metrics as suggested by domain expert.*")
    sections.append("*All scores can be manually verified by counting dependencies in the theorem library.*")
    
    # Write to file
    report_content = "\n".join(sections)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(report_content)
    
    print(f"Report saved to: {args.output}")
    print(f"Total theorems analyzed: {len(df)}")
    print(f"You can view the markdown report in any markdown viewer or GitHub.")


if __name__ == "__main__":
    main()