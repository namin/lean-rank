#!/usr/bin/env python3
"""
Filter Mathlib data to a specific domain, keeping only dependencies within that domain.
This creates a clean dataset for domain-specific analysis without infrastructure noise.
"""

import argparse
import json
from pathlib import Path
from typing import Set, Dict, List, Callable

# Domain-specific pattern definitions
DOMAIN_PATTERNS = {
    'number_theory': [
        "Nat.Prime", "Nat.prime", "Nat.Coprime", "Nat.gcd", "Nat.lcm",
        "Nat.divisor", "Nat.factor", "Nat.div", "Nat.mod",
        "Int.Prime", "Int.gcd", "Int.div", "Int.mod", 
        "Prime", "Coprime", "IsPrime",
        "PythagoreanTriple", "Pell",
        "NumberTheory", "ArithmeticFunction", 
        "ZMod", "Zsqrtd",
        "LegendreSymbol", "JacobiSymbol", "QuadraticReciprocity",
        "Fermat", "Wilson", "Euler"
    ],
    'topology': [
        "TopologicalSpace", "Opens", "Continuous", "Homeomorph",
        "CompactSpace", "HausdorffSpace", "T0Space", "T1Space",
        "Connected", "PathConnected", "SimplyConnected",
        "Metric", "Uniform", "Cauchy", "Complete",
        "Dense", "Closure", "Interior", "Boundary",
        "Neighborhood", "Filter.Tendsto", "ContinuousAt"
    ],
    'algebra': [
        "Group", "Ring", "Field", "Module", "Algebra",
        "Monoid", "CommRing", "CommGroup", "GroupHom", "RingHom",
        "Subgroup", "Subring", "Ideal", "PrimeIdeal", "MaximalIdeal",
        "Polynomial", "MvPolynomial", "PowerSeries",
        "Matrix", "LinearMap", "LinearEquiv", "Basis"
    ],
    'analysis': [
        "Derivative", "Integral", "Differentiable", "ContDiff",
        "HasDerivAt", "HasFDerivAt", "Convex", "ConvexHull",
        "Measure", "MeasurableSpace", "Integrable", "AEMeasurable",
        "Lp", "L1", "L2", "Hilbert", "Banach",
        "Norm", "InnerProduct", "OrthonormalBasis"
    ],
    'category_theory': [
        "Category", "Functor", "NatTrans", "Equivalence",
        "Limits", "Colimits", "Adjunction", "Monad",
        "Abelian", "Topos", "Sheaf", "Presheaf",
        "CategoryTheory", "CommRing", "Module"
    ],
    'order': [
        "PartialOrder", "LinearOrder", "Lattice", "CompleteLattice",
        "OrderedGroup", "OrderedRing", "OrderedField",
        "Monotone", "Antitone", "OrderIso", "OrderHom",
        "SupSet", "InfSet", "DirectedSet", "Chain",
        "WellFounded", "WellOrder"
    ]
}

def get_domain_checker(domain: str, custom_patterns: List[str] = None) -> Callable[[str], bool]:
    """Get a function that checks if a declaration belongs to a domain."""
    if custom_patterns:
        patterns = custom_patterns
    elif domain in DOMAIN_PATTERNS:
        patterns = DOMAIN_PATTERNS[domain]
    else:
        raise ValueError(f"Unknown domain '{domain}'. Use --patterns or choose from: {list(DOMAIN_PATTERNS.keys())}")
    
    def is_in_domain(name: str) -> bool:
        return any(pattern in name for pattern in patterns)
    
    return is_in_domain

def parse_premises(file_path: Path) -> Dict[str, Set[str]]:
    """Parse premises.txt to get declaration dependencies."""
    premises = {}
    current_decl = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line == '---':
                current_decl = None
            elif current_decl is None:
                current_decl = line
                premises[current_decl] = set()
            elif line:
                dep = line.lstrip('* s')
                premises[current_decl].add(dep)
    
    return premises

def get_declaration_types(file_path: Path) -> Dict[str, str]:
    """Parse declaration_types.txt to get kinds (theorem, lemma, etc)."""
    decl_types = {}
    with open(file_path, 'r') as f:
        current_block = []
        for line in f:
            if line.strip() == '---':
                if len(current_block) >= 2:
                    kind = current_block[0].strip()
                    name = current_block[1].strip()
                    decl_types[name] = kind
                current_block = []
            else:
                current_block.append(line)
        if len(current_block) >= 2:
            kind = current_block[0].strip()
            name = current_block[1].strip()
            decl_types[name] = kind
    return decl_types

def filter_domain(premises: Dict[str, Set[str]], 
                  decl_types: Dict[str, str],
                  is_in_domain: Callable[[str], bool],
                  kinds: Set[str] = {'theorem', 'lemma'}) -> tuple:
    """
    Filter to only domain-specific theorems/lemmas.
    Only keeps dependencies between declarations in the domain.
    This eliminates infrastructure noise (Eq, congrArg, etc.) for clean domain analysis.
    """
    # First pass: identify all domain theorems/lemmas
    domain_decls = set()
    for name, kind in decl_types.items():
        if kind in kinds and is_in_domain(name):
            domain_decls.add(name)
    
    print(f"Found {len(domain_decls)} domain {'/'.join(kinds)}")
    
    # Second pass: filter premises to only include dependencies within domain
    filtered_premises = {}
    for decl in domain_decls:
        if decl in premises:
            domain_deps = {dep for dep in premises[decl] if dep in domain_decls}
            if domain_deps:
                filtered_premises[decl] = domain_deps
    
    print(f"Kept {len(filtered_premises)} declarations with domain dependencies")
    
    # Collect all declarations that appear
    all_decls = set(filtered_premises.keys())
    for deps in filtered_premises.values():
        all_decls.update(deps)
    
    return all_decls, filtered_premises

def write_filtered_premises(premises: Dict[str, Set[str]], 
                          keep_decls: Set[str],
                          output_path: Path):
    """Write filtered premises file."""
    with open(output_path, 'w') as f:
        for decl, deps in premises.items():
            if decl in keep_decls:
                filtered_deps = {d for d in deps if d in keep_decls}
                if filtered_deps:
                    f.write('---\n')
                    f.write(f'{decl}\n')
                    for dep in sorted(filtered_deps):
                        f.write(f'  {dep}\n')

def write_filtered_types(input_path: Path, output_path: Path, keep_decls: Set[str]):
    """Write filtered declaration_types file."""
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        current_block = []
        for line in fin:
            if line.strip() == '---':
                if len(current_block) >= 2:
                    name = current_block[1].strip()
                    if name in keep_decls:
                        fout.write('---\n')
                        for l in current_block:
                            fout.write(l)
                current_block = []
            else:
                current_block.append(line)
        if len(current_block) >= 2:
            name = current_block[1].strip()
            if name in keep_decls:
                fout.write('---\n')
                for l in current_block:
                    fout.write(l)

def write_filtered_structures(input_path: Path, output_path: Path, keep_decls: Set[str]):
    """Write filtered declaration_structures file."""
    if not input_path.exists():
        return
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
                if obj.get('name') in keep_decls:
                    fout.write(line)
            except json.JSONDecodeError:
                continue

def main():
    parser = argparse.ArgumentParser(
        description="Filter Mathlib data to a specific domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter to number theory (only intra-domain dependencies)
  python -m src.tasks.filter_domain --domain number_theory
  
  # Filter to topology
  python -m src.tasks.filter_domain --domain topology
  
  # Custom domain with patterns
  python -m src.tasks.filter_domain --patterns "Measure,Integral,Lp" --output data/measure_theory
  
Available domains: """ + ", ".join(DOMAIN_PATTERNS.keys()))
    
    parser.add_argument('--domain', choices=list(DOMAIN_PATTERNS.keys()),
                       help='Predefined domain to filter to')
    parser.add_argument('--patterns', type=str,
                       help='Comma-separated custom patterns to match')
    parser.add_argument('--kinds', default='theorem,lemma',
                       help='Declaration kinds to keep (default: theorem,lemma)')
    parser.add_argument('--input', type=Path, default=Path('data'),
                       help='Input directory with Mathlib data')
    parser.add_argument('--output', type=Path,
                       help='Output directory (default: data/<domain>_strict or data/<domain>)')
    
    args = parser.parse_args()
    
    # Determine domain and patterns
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(',')]
        domain_name = 'custom'
        is_in_domain = get_domain_checker(None, patterns)
    elif args.domain:
        domain_name = args.domain
        is_in_domain = get_domain_checker(args.domain)
    else:
        parser.error("Either --domain or --patterns must be specified")
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.input / f"{domain_name}_filtered"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse input files
    print(f"Parsing {args.input}/premises.txt...")
    all_premises = parse_premises(args.input / "premises.txt")
    
    print(f"Parsing {args.input}/declaration_types.txt...")
    decl_types = get_declaration_types(args.input / "declaration_types.txt")
    
    # Parse kinds
    kinds = set(k.strip() for k in args.kinds.split(','))
    
    # Apply filtering
    print(f"\nFiltering to domain '{domain_name}' (only intra-domain dependencies)...")
    
    keep_decls, filtered_premises = filter_domain(
        all_premises, decl_types, is_in_domain, kinds)
    
    print(f"\nFinal dataset: {len(keep_decls)} declarations")
    
    # Write filtered files
    print(f"\nWriting filtered premises.txt...")
    write_filtered_premises(filtered_premises, keep_decls, output_dir / "premises.txt")
    
    print(f"Writing filtered declaration_types.txt...")
    write_filtered_types(
        args.input / "declaration_types.txt",
        output_dir / "declaration_types.txt",
        keep_decls
    )
    
    if (args.input / "declaration_structures.jsonl").exists():
        print(f"Writing filtered declaration_structures.jsonl...")
        write_filtered_structures(
            args.input / "declaration_structures.jsonl",
            output_dir / "declaration_structures.jsonl",
            keep_decls
        )
    
    # Print statistics
    print(f"\nFiltered dataset written to {output_dir}/")
    print(f"Statistics:")
    print(f"  - Declarations: {len(keep_decls)}")
    print(f"  - With dependencies: {len([d for d in filtered_premises if d in keep_decls])}")
    
    # Sample some declarations
    sample = list(keep_decls)[:10]
    if sample:
        print(f"\nSample declarations:")
        for name in sample:
            print(f"  - {name}")

if __name__ == "__main__":
    main()