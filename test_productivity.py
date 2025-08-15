#!/usr/bin/env python3
"""
Test both adoption rates and semantic productivity for theorem statements.
Provides a comprehensive productivity analysis.
"""

import subprocess
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Define theorem statements to test
THEOREMS = {
    "Fermat's Little Theorem": "∀ p : ℕ, Prime p → ∀ a : ℕ, ¬(p ∣ a) → a ^ (p - 1) ≡ 1 [MOD p]",
    "Divisibility Transitivity": "∀ a b c : ℕ, a ∣ b → b ∣ c → a ∣ c",
    "Infinitude of Primes": "∀ n : ℕ, ∃ p : ℕ, Prime p ∧ p > n",
    "Prime Oddness": "∀ p : ℕ, Prime p → p = 2 ∨ Odd p",
    "Coprimality Divisibility": "∀ a b : ℕ, Coprime a b → ∀ c : ℕ, a ∣ c → b ∣ c → a * b ∣ c",
    "Euclid's Lemma": "∀ p : ℕ, Prime p → ∀ a b : ℕ, p ∣ a * b → p ∣ a ∨ p ∣ b",
    "GCD Commutativity": "∀ a b : ℕ, gcd a b = gcd b a",
    "Prime Factorial": "∀ p : ℕ, Prime p → ∀ n : ℕ, n < p → ¬(p ∣ n!)",
    "Wilson's Theorem": "∀ p : ℕ, Prime p ↔ (p - 1)! ≡ -1 [MOD p]",
    "Bezout's Identity": "∀ a b : ℕ, ∃ x y : ℤ, a * x + b * y = gcd a b"
}

def parse_adoption_output(output: str) -> Dict[str, Tuple[float, float, int]]:
    """Parse adoption rates from score_productivity output."""
    results = {}
    pattern = r'adoption@(\d+):\s+([\d.]+)%\s+\(~(\d+)/\d+\);\s+random=[\d.]+%\s+lift=([\d.]+)'
    
    for match in re.finditer(pattern, output):
        k = match.group(1)
        adoption = float(match.group(2))
        count = int(match.group(3))
        lift = float(match.group(4))
        results[f'adoption@{k}'] = (adoption, lift, count)
    
    return results

def parse_semantic_output(output: str) -> Dict[str, float]:
    """Parse semantic productivity predictions."""
    results = {}
    
    # Extract predicted productivity values
    patterns = {
        'weighted': r'Weighted by similarity:\s+([\d.]+)',
        'top10_mean': r'Mean of top 10 similar:\s+([\d.]+)',
        'top50_mean': r'Mean of top 50 similar:\s+([\d.]+)',
        'max_similar': r"Most similar's productivity:\s+([\d.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[key] = float(match.group(1))
    
    # Extract most similar theorems
    similar_pattern = r'(\S+)\s+\|\s+sim=([\d.]+)\s+\|\s+prod=(\d+)'
    similar_theorems = []
    for match in re.finditer(similar_pattern, output):
        theorem = match.group(1)
        similarity = float(match.group(2))
        prod = int(match.group(3))
        similar_theorems.append((theorem, similarity, prod))
    
    if similar_theorems:
        results['most_similar'] = similar_theorems[0]
        results['similar_count'] = len(similar_theorems)
    
    return results

def run_adoption_test(
    type_string: str,
    domain: str = "number_theory",
    k_list: str = "10,20,50"
) -> Dict[str, Tuple[float, float, int]]:
    """Run adoption rate scoring."""
    
    data_dir = f"data/{domain}_filtered"
    proc_dir = f"{data_dir}/processed"
    out_dir = f"outputs/{domain}_filtered"
    
    cmd = [
        sys.executable, "-m", "src.tasks.score_productivity",
        "--type_string", type_string,
        "--features", f"{proc_dir}/structure_features.npz",
        "--contexts", f"{proc_dir}/contexts.jsonl",
        "--rankings", f"{proc_dir}/rankings.parquet",
        "--nodes", f"{proc_dir}/nodes.parquet",
        "--decltypes", f"{proc_dir}/decltypes.parquet",
        "--ckpt", f"{out_dir}/text_ranker.pt",
        "--buckets", "128",
        "--k_list", k_list,
        "--target_kinds", "theorem,lemma"
    ]
    
    env = {
        **subprocess.os.environ,
        "DATA_DIR": data_dir,
        "PROC_DIR": proc_dir,
        "OUT_DIR": out_dir,
        "TARGET_PREFIXES": ""
    }
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        if result.returncode != 0:
            print(f"Error in adoption test: {result.stderr}")
            return {}
        return parse_adoption_output(result.stdout)
    except Exception as e:
        print(f"Error: {e}")
        return {}

def run_semantic_test(
    type_string: str,
    domain: str = "number_theory"
) -> Dict[str, float]:
    """Run semantic productivity prediction."""
    
    data_dir = f"data/{domain}_filtered"
    proc_dir = f"{data_dir}/processed"
    out_dir = f"outputs/{domain}_filtered"
    
    # Check if required files exist
    graph_metrics_path = f"{proc_dir}/graph_metrics.parquet"
    if not Path(graph_metrics_path).exists():
        print(f"Warning: {graph_metrics_path} not found, skipping semantic prediction")
        return {}
    
    cmd = [
        sys.executable, "-m", "src.tasks.predict_productivity_semantic",
        "--structures", f"{proc_dir}/structures.parquet",
        "--graph_metrics", graph_metrics_path,
        "--features", f"{proc_dir}/structure_features.npz",
        "--ckpt", f"{out_dir}/text_ranker.pt",
        "--type_string", type_string
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Error in semantic test: {result.stderr}")
            return {}
        return parse_semantic_output(result.stdout)
    except Exception as e:
        print(f"Error: {e}")
        return {}

def generate_comprehensive_table(results: Dict[str, Dict]) -> str:
    """Generate a comprehensive markdown table."""
    lines = []
    
    # Header
    lines.append("| Theorem | Adoption@10 | Adoption@50 | Semantic Pred | Similarity | Category |")
    lines.append("|---------|-------------|-------------|---------------|------------|----------|")
    
    # Sort by semantic prediction (weighted)
    def get_sort_key(item):
        _, data = item
        semantic = data.get('semantic', {})
        return semantic.get('weighted', 0)
    
    sorted_theorems = sorted(results.items(), key=get_sort_key, reverse=True)
    
    for name, data in sorted_theorems:
        adoption = data.get('adoption', {})
        semantic = data.get('semantic', {})
        
        # Format adoption
        a10 = adoption.get('adoption@10', (0, 0, 0))
        a50 = adoption.get('adoption@50', (0, 0, 0))
        
        adoption10_str = f"{a10[0]:.2f}% (L={a10[1]:.2f})" if a10[0] > 0 else "0%"
        adoption50_str = f"{a50[0]:.2f}% (~{a50[2]})" if a50[0] > 0 else "0%"
        
        # Format semantic prediction
        semantic_pred = semantic.get('weighted', 0)
        semantic_str = f"**{semantic_pred:.1f}**" if semantic_pred > 5 else f"{semantic_pred:.1f}"
        
        # Similarity to most productive
        similarity_str = "-"
        if 'most_similar' in semantic:
            sim_info = semantic['most_similar']
            similarity_str = f"{sim_info[1]:.3f}"
        
        # Categorize
        category = categorize_theorem(a10[1], semantic_pred)
        
        lines.append(f"| {name} | {adoption10_str} | {adoption50_str} | {semantic_str} | {similarity_str} | {category} |")
    
    return "\n".join(lines)

def categorize_theorem(lift: float, semantic_pred: float) -> str:
    """Categorize a theorem based on its metrics."""
    if lift > 1.0 and semantic_pred > 5:
        return "🌟 **High Impact**"
    elif lift > 0.5 or semantic_pred > 3:
        return "✓ Useful"
    elif semantic_pred > 1:
        return "◐ Moderate"
    else:
        return "○ Low"

def main():
    parser = argparse.ArgumentParser(description="Comprehensive productivity testing")
    parser.add_argument("--domain", default="number_theory",
                       help="Domain to test (default: number_theory)")
    parser.add_argument("--theorems", nargs="+",
                       help="Specific theorems to test")
    parser.add_argument("--custom", metavar="NAME:TYPE", action="append",
                       help="Add custom theorem")
    parser.add_argument("--skip-semantic", action="store_true",
                       help="Skip semantic productivity prediction")
    args = parser.parse_args()
    
    # Select theorems
    theorems_to_test = THEOREMS.copy()
    if args.theorems:
        theorems_to_test = {k: v for k, v in THEOREMS.items() if k in args.theorems}
    if args.custom:
        for custom in args.custom:
            if ':' in custom:
                name, type_str = custom.split(':', 1)
                theorems_to_test[name] = type_str
    
    print(f"Testing {len(theorems_to_test)} theorems in domain: {args.domain}")
    print("=" * 70)
    
    results = {}
    for i, (name, type_string) in enumerate(theorems_to_test.items(), 1):
        print(f"\n[{i}/{len(theorems_to_test)}] {name}")
        print(f"    Type: {type_string[:60]}...")
        
        # Run adoption test
        print("    Testing adoption rates...", end=" ")
        adoption_result = run_adoption_test(type_string, args.domain)
        if adoption_result:
            a10 = adoption_result.get('adoption@10', (0, 0, 0))
            print(f"✓ ({a10[0]:.2f}% @ k=10)")
        else:
            print("✗")
        
        # Run semantic test
        semantic_result = {}
        if not args.skip_semantic:
            print("    Testing semantic productivity...", end=" ")
            semantic_result = run_semantic_test(type_string, args.domain)
            if semantic_result:
                pred = semantic_result.get('weighted', 0)
                print(f"✓ (pred={pred:.1f})")
            else:
                print("✗")
        
        results[name] = {
            'adoption': adoption_result,
            'semantic': semantic_result
        }
    
    # Generate comprehensive report
    print("\n" + "=" * 70)
    print("\n## COMPREHENSIVE PRODUCTIVITY ANALYSIS\n")
    print(generate_comprehensive_table(results))
    
    # Summary statistics
    print("\n## Key Insights\n")
    
    # Find best performers
    best_adoption = max(results.items(), 
                       key=lambda x: x[1]['adoption'].get('adoption@10', (0,0,0))[0])
    
    if not args.skip_semantic:
        best_semantic = max(results.items(),
                           key=lambda x: x[1]['semantic'].get('weighted', 0))
        
        print(f"**Best by adoption**: {best_adoption[0]} "
              f"({best_adoption[1]['adoption']['adoption@10'][0]:.2f}% @ k=10)")
        print(f"**Best by semantic prediction**: {best_semantic[0]} "
              f"(predicted {best_semantic[1]['semantic']['weighted']:.1f} downstream uses)")
        
        # Count high impact
        high_impact = sum(1 for _, data in results.items()
                         if data['adoption'].get('adoption@10', (0,0,0))[1] > 1.0
                         and data['semantic'].get('weighted', 0) > 5)
        print(f"**High impact theorems**: {high_impact}/{len(results)}")
    else:
        print(f"**Best by adoption**: {best_adoption[0]} "
              f"({best_adoption[1]['adoption']['adoption@10'][0]:.2f}% @ k=10)")
    
    # Explain metrics
    print("\n### Metric Explanations:")
    print("- **Adoption@k**: % of existing theorems that would use this (higher is better)")
    print("- **L=lift**: Performance vs random (>1.0 means better than random)")
    print("- **Semantic Pred**: Predicted number of downstream theorems based on similarity")
    print("- **Similarity**: How similar to existing theorems (1.0 = identical)")

if __name__ == "__main__":
    main()