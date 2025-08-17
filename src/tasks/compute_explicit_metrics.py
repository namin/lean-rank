#!/usr/bin/env python3
"""
Compute explicit, interpretable theorem importance metrics.

Three metrics as suggested by domain expert:
1. Direct usage count: How many theorems directly call this one
2. Transitive dependency count: How many theorems depend on this (full closure)
3. Combined weighted: (direct × transitive) / (existential_quantifiers + 1)

These metrics are fully explainable and can be manually verified.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
from datetime import datetime


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
    
    # Compute in-degree (direct usage)
    in_deg = np.array(A.sum(axis=0)).ravel()
    
    return A, in_deg, n


def compute_transitive_closure(A):
    """Compute full transitive closure."""
    print("Computing full transitive closure...")
    n = A.shape[0]
    
    if n < 5000:  # For smaller graphs, use dense computation
        print("Using dense matrix computation...")
        A_dense = A.toarray()
        TC = (A_dense > 0).astype(np.float32)
        
        # Floyd-Warshall algorithm
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
        
        transitive_count = np.array(TC.sum(axis=1)).ravel()
    
    return transitive_count


def load_declaration_info(declaration_file):
    """Load theorem declarations and extract metadata."""
    print("Loading declaration structures...")
    declarations = []
    with open(declaration_file, "r") as f:
        for line in tqdm(f, desc="Loading declarations"):
            declarations.append(json.loads(line))
    
    # Create mapping
    decl_map = {d["name"]: d for d in declarations}
    return decl_map


def compute_explicit_metrics(data_dir="data/number_theory_filtered", use_cache=True):
    """Compute all explicit metrics for theorems."""
    
    # Check for cached transitive closure
    cache_file = f"{data_dir}/processed/transitive_closure.npz"
    contexts_file = f"{data_dir}/processed/contexts.jsonl"
    
    if use_cache and Path(cache_file).exists():
        print("Loading cached transitive closure...")
        cache = np.load(cache_file)
        in_deg = cache['in_deg']
        transitive_deps = cache['transitive_deps']
        n = len(in_deg)
        print(f"Loaded cache: {n} nodes")
    else:
        # Build dependency graph and compute transitive closure
        A, in_deg, n = build_dependency_graph(contexts_file)
        transitive_deps = compute_transitive_closure(A)
        
        # Save to cache
        if use_cache:
            print("Saving transitive closure to cache...")
            np.savez_compressed(cache_file, 
                               in_deg=in_deg, 
                               transitive_deps=transitive_deps)
            print(f"Cache saved to: {cache_file}")
    
    # Load node names
    nodes_df = pd.read_parquet(f"{data_dir}/processed/nodes.parquet")
    
    # Load declaration info
    decl_map = load_declaration_info(f"{data_dir}/declaration_structures.jsonl")
    
    # Build results dataframe
    print("Computing explicit metrics...")
    results = []
    
    for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Processing theorems"):
        node_id = row["id"]
        name = row["name"]
        
        # Get declaration info
        if name in decl_map:
            decl = decl_map[name]
            kind = decl.get("kind", "unknown")
            num_exists = decl.get("num_exists", 0)
            num_forall = decl.get("num_forall", 0)
            type_str = decl.get("type", "")[:200]
        else:
            kind = "unknown"
            num_exists = 0
            num_forall = 0
            type_str = ""
        
        # Compute metrics
        metric1_direct = in_deg[node_id] if node_id < len(in_deg) else 0
        metric2_transitive = transitive_deps[node_id] if node_id < len(transitive_deps) else 0
        metric3_combined = (metric1_direct * metric2_transitive) / (num_exists + 1)
        
        results.append({
            "id": node_id,
            "name": name,
            "kind": kind,
            "metric1_direct_usage": metric1_direct,
            "metric2_transitive_deps": metric2_transitive,
            "metric3_combined": metric3_combined,
            "num_exists": num_exists,
            "num_forall": num_forall,
            "type_string": type_str
        })
    
    df = pd.DataFrame(results)
    
    # Add rankings
    theorem_df = df[df["kind"].isin(["theorem", "lemma"])].copy()
    theorem_df["rank_direct"] = theorem_df["metric1_direct_usage"].rank(ascending=False, method="min")
    theorem_df["rank_transitive"] = theorem_df["metric2_transitive_deps"].rank(ascending=False, method="min")
    theorem_df["rank_combined"] = theorem_df["metric3_combined"].rank(ascending=False, method="min")
    
    return theorem_df


def generate_text_report(df, top_n=20):
    """Generate human-readable text report."""
    
    print("\n" + "=" * 80)
    print("THEOREM IMPORTANCE ANALYSIS - EXPLICIT METRICS")
    print("=" * 80)
    
    print(f"\nDataset: {len(df)} theorems and lemmas")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Metric 1: Direct Usage
    print("\n" + "-" * 60)
    print("METRIC 1: DIRECT USAGE COUNT")
    print("(How many theorems directly call this one)")
    print("-" * 60)
    
    top_direct = df.nlargest(min(top_n, len(df)), "metric1_direct_usage")
    for idx, row in top_direct.iterrows():
        print(f"{row['metric1_direct_usage']:3.0f} uses | {row['name']}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean: {df['metric1_direct_usage'].mean():.2f}")
    print(f"  Max: {df['metric1_direct_usage'].max():.0f}")
    print(f"  Zero usage: {(df['metric1_direct_usage'] == 0).sum()} ({(df['metric1_direct_usage'] == 0).mean()*100:.1f}%)")
    
    # Metric 2: Transitive Dependencies
    print("\n" + "-" * 60)
    print("METRIC 2: TRANSITIVE DEPENDENCIES (FULL CLOSURE)")
    print("(How many theorems depend on this one)")
    print("-" * 60)
    
    top_trans = df.nlargest(min(top_n, len(df)), "metric2_transitive_deps")
    for idx, row in top_trans.iterrows():
        print(f"{row['metric2_transitive_deps']:4.0f} deps | {row['name']}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {df['metric2_transitive_deps'].mean():.2f}")
    print(f"  Max: {df['metric2_transitive_deps'].max():.0f}")
    
    # Metric 3: Combined
    print("\n" + "-" * 60)
    print("METRIC 3: COMBINED WEIGHTED SCORE")
    print("Formula: (direct × transitive) / (∃ + 1)")
    print("-" * 60)
    
    top_combined = df.nlargest(min(top_n, len(df)), "metric3_combined")
    for idx, row in top_combined.iterrows():
        print(f"{row['metric3_combined']:8.1f} | {row['name']}")
        print(f"           | D={row['metric1_direct_usage']:.0f}, T={row['metric2_transitive_deps']:.0f}, ∃={row['num_exists']:.0f}")
    
    # Correlations
    print("\n" + "-" * 60)
    print("METRIC CORRELATIONS")
    print("-" * 60)
    
    corr_direct_trans = df['metric1_direct_usage'].corr(df['metric2_transitive_deps'])
    print(f"Direct vs Transitive: {corr_direct_trans:.3f}")
    
    if corr_direct_trans < 0:
        print("  → Negative correlation: foundational theorems rarely used directly")
    
    # Summary
    print("\n" + "-" * 60)
    print("KEY INSIGHTS")
    print("-" * 60)
    
    print(f"• {(df['metric1_direct_usage'] == 0).mean()*100:.1f}% of theorems have zero direct usage")
    print(f"• {(df['num_exists'] > 0).mean()*100:.1f}% of theorems contain existential quantifiers")
    print(f"• Top theorem by combined metric: {df.nlargest(1, 'metric3_combined').iloc[0]['name']}")
    
    print("\n" + "=" * 80)


def load_theorem_statements(data_dir, max_length=None):
    """Load theorem statements from declaration_types.txt."""
    statements = {}
    types_file = f"{data_dir}/declaration_types.txt"
    
    if Path(types_file).exists():
        with open(types_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by separator
            declarations = content.split("---\n")
            
            for decl in declarations:
                if not decl.strip():
                    continue
                lines = decl.strip().split("\n")
                if len(lines) >= 3:
                    # kind = lines[0]  # theorem, definition, etc.
                    name = lines[1]
                    type_str = " ".join(lines[2:])  # Join multi-line types
                    
                    # Clean up the type string for display
                    type_str = type_str.replace("\n", " ").strip()
                    # Truncate if requested
                    if max_length and len(type_str) > max_length:
                        type_str = type_str[:max_length-3] + "..."
                    statements[name] = type_str
    return statements


def generate_markdown_report(df, output_file, top_n=20, unified_table=False, data_dir=None, 
                           statement_length=100):
    """Generate markdown report for sharing."""
    
    lines = []
    lines.append("# Theorem Importance Analysis - Explicit Metrics")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    
    if unified_table:
        # Load theorem statements if available
        statements = {}
        if data_dir:
            # Load without truncation initially
            statements = load_theorem_statements(data_dir, max_length=None)
        
        # Generate unified table sorted by combined metric
        lines.append("## Unified Ranking Table")
        lines.append("")
        lines.append("Sorted by Combined Score (most important first)")
        lines.append("")
        lines.append("| Rank | Combined Score | Direct Usage | Transitive Deps | ∃ | Theorem | Statement |")
        lines.append("|------|----------------|--------------|-----------------|---|---------|-----------|")
        
        top_df = df.nlargest(min(top_n, len(df)), "metric3_combined")
        for i, (idx, row) in enumerate(top_df.iterrows(), 1):
            exists_str = str(int(row['num_exists'])) if row['num_exists'] > 0 else "-"
            theorem_name = row['name']
            
            # Get statement if available
            statement = statements.get(theorem_name, "")
            if statement:
                # Escape pipes in statements for markdown tables
                statement = statement.replace("|", "\\|")
                # Truncate for table display if needed
                if statement_length and len(statement) > statement_length:
                    statement = statement[:statement_length-3] + "..."
            
            lines.append(f"| {i} | {row['metric3_combined']:.1f} | {row['metric1_direct_usage']:.0f} | "
                        f"{row['metric2_transitive_deps']:.0f} | {exists_str} | `{theorem_name}` | {statement} |")
        
        lines.append("")
        lines.append("### Metric Explanations")
        lines.append("")
        lines.append("- **Combined Score**: `(direct × transitive) / (∃ + 1)` - Overall importance metric")
        lines.append("- **Direct Usage**: Number of theorems that directly call this theorem")
        lines.append("- **Transitive Deps**: Number of theorems that depend on this (full closure)")
        lines.append("- **∃**: Count of existential quantifiers in the statement")
        lines.append("")
        
        # Add summary statistics
        lines.append("### Summary Statistics")
        lines.append("")
        lines.append(f"- Total theorems analyzed: {len(df)}")
        lines.append(f"- Theorems with zero direct usage: {(df['metric1_direct_usage'] == 0).sum()} "
                    f"({(df['metric1_direct_usage'] == 0).mean()*100:.1f}%)")
        lines.append(f"- Average direct usage: {df['metric1_direct_usage'].mean():.2f}")
        lines.append(f"- Average transitive dependencies: {df['metric2_transitive_deps'].mean():.2f}")
        lines.append("")
        
        lines.append("---")
        lines.append("*All metrics are explainable and can be manually verified.*")
        
    else:
        # Original multi-section format
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total theorems analyzed**: {len(df)}")
        lines.append(f"- **Theorems with zero usage**: {(df['metric1_direct_usage'] == 0).sum()} ({(df['metric1_direct_usage'] == 0).mean()*100:.1f}%)")
        lines.append(f"- **Theorems with existential quantifiers**: {(df['num_exists'] > 0).sum()} ({(df['num_exists'] > 0).mean()*100:.1f}%)")
        lines.append("")
    
        # Top theorems by each metric
        lines.append("## Top Theorems by Each Metric")
        lines.append("")
        
        lines.append("### Metric 1: Direct Usage Count")
        lines.append("")
        lines.append("| Rank | Uses | Theorem |")
        lines.append("|------|------|---------|")
        
        top_direct = df.nlargest(min(top_n, len(df)), "metric1_direct_usage")
        for i, (idx, row) in enumerate(top_direct.iterrows(), 1):
            lines.append(f"| {i} | {row['metric1_direct_usage']:.0f} | `{row['name']}` |")
        
        lines.append("")
        lines.append("### Metric 2: Transitive Dependencies (Full Closure)")
        lines.append("")
        lines.append("| Rank | Dependencies | Theorem |")
        lines.append("|------|--------------|---------|")
        
        top_trans = df.nlargest(min(top_n, len(df)), "metric2_transitive_deps")
        for i, (idx, row) in enumerate(top_trans.iterrows(), 1):
            lines.append(f"| {i} | {row['metric2_transitive_deps']:.0f} | `{row['name']}` |")
        
        lines.append("")
        lines.append("### Metric 3: Combined Weighted Score")
        lines.append("")
        lines.append("**Formula**: `(direct_usage × transitive_deps) / (existential_quantifiers + 1)`")
        lines.append("")
        lines.append("| Rank | Score | Direct | Transitive | ∃ | Theorem |")
        lines.append("|------|-------|--------|------------|---|---------|")
        
        top_combined = df.nlargest(min(top_n, len(df)), "metric3_combined")
        for i, (idx, row) in enumerate(top_combined.iterrows(), 1):
            exists_str = str(int(row['num_exists'])) if row['num_exists'] > 0 else "-"
            lines.append(f"| {i} | {row['metric3_combined']:.1f} | {row['metric1_direct_usage']:.0f} | {row['metric2_transitive_deps']:.0f} | {exists_str} | `{row['name']}` |")
        
        lines.append("")
        lines.append("## Methodology")
        lines.append("")
        lines.append("All metrics are explicit and can be manually verified:")
        lines.append("1. **Direct usage**: Count theorems where this appears in premises")
        lines.append("2. **Transitive dependencies**: Full transitive closure of dependency graph")
        lines.append("3. **Combined score**: Balances both metrics, penalizes complex statements")
        lines.append("")
        lines.append("---")
        lines.append("*Generated using explainable metrics as suggested by domain expert.*")
    
    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Markdown report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute explicit theorem importance metrics")
    parser.add_argument("--data_dir", default="data/number_theory_filtered",
                       help="Directory containing processed data")
    parser.add_argument("--output_csv", default="outputs/explicit_metrics.csv",
                       help="Output CSV file for metrics")
    parser.add_argument("--output_md", default="outputs/explicit_metrics_report.md",
                       help="Output markdown report")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top theorems to display")
    parser.add_argument("--format", choices=["text", "markdown", "both"], default="both",
                       help="Output format")
    parser.add_argument("--force", action="store_true",
                       help="Force recomputation (ignore cache)")
    parser.add_argument("--unified-table", action="store_true",
                       help="Generate unified table format (markdown only)")
    parser.add_argument("--statement-length", type=int, default=100,
                       help="Maximum length for theorem statements in table (0 for no limit)")
    args = parser.parse_args()
    
    # Handle cache invalidation
    if args.force:
        cache_file = Path(f"{args.data_dir}/processed/transitive_closure.npz")
        if cache_file.exists():
            cache_file.unlink()
            print("Cache invalidated, forcing recomputation...")
    
    # Compute metrics
    print("Computing explicit metrics...")
    df = compute_explicit_metrics(args.data_dir, use_cache=True)
    
    # Save CSV
    Path(args.output_csv).parent.mkdir(exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to: {args.output_csv}")
    
    # Generate reports
    if args.format in ["text", "both"]:
        generate_text_report(df, args.top_n)
    
    if args.format in ["markdown", "both"]:
        statement_length = args.statement_length if args.statement_length > 0 else None
        generate_markdown_report(df, args.output_md, args.top_n, 
                               unified_table=args.unified_table,
                               data_dir=args.data_dir,
                               statement_length=statement_length)
    
    print("\nAnalysis complete!")
    print(f"Total theorems analyzed: {len(df)}")
    print(f"CSV output: {args.output_csv}")
    if args.format in ["markdown", "both"]:
        print(f"Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()