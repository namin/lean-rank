#!/usr/bin/env python3
"""
Analyze hypothesis patterns by extracting meaningful predicates/functions
from hypothesis strings, ignoring fvar placeholders.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class HypothesisPatternAnalyzer:
    def __init__(self, jsonl_path: Path):
        self.theorems = {}
        self.load_structures(jsonl_path)
        self.discover_predicates()
    
    def load_structures(self, path: Path):
        """Load declaration_structures.jsonl."""
        print(f"Loading structures from {path}")
        with open(path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('kind') in ['theorem', 'lemma']:
                        self.theorems[obj['name']] = obj
                except:
                    continue
        print(f"Loaded {len(self.theorems)} theorems/lemmas")
    
    def extract_predicates_from_hypothesis(self, hyp_str: str) -> List[str]:
        """Extract meaningful predicates/functions from a hypothesis string."""
        predicates = []
        
        # Remove _fvar.XXX patterns
        cleaned = re.sub(r'_fvar\.\d+', 'X', hyp_str)
        
        # Extract capitalized names (likely predicates/types)
        # Examples: Nat.Prime, IsCoprime, Squarefree, etc.
        capital_names = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)*\b', cleaned)
        predicates.extend(capital_names)
        
        # Extract common mathematical operators
        if '∣' in cleaned or 'Dvd' in cleaned:
            predicates.append('DIVIDES')
        if '=' in cleaned and '≠' not in cleaned and '≡' not in cleaned:
            predicates.append('EQUALS')
        if any(op in cleaned for op in ['≤', '≥', '<', '>', '≠']):
            predicates.append('INEQUALITY')
        if '≡' in cleaned or 'mod' in cleaned.lower():
            predicates.append('MODULAR')
        if 'gcd' in cleaned.lower():
            predicates.append('GCD')
        if 'lcm' in cleaned.lower():
            predicates.append('LCM')
        
        # Extract function names (lowercase start, often after dots)
        func_names = re.findall(r'(?:^|\.)([a-z][a-zA-Z0-9]*)', cleaned)
        for fname in func_names:
            if fname not in ['fun', 'let', 'have', 'show', 'by']:  # Skip Lean keywords
                predicates.append(f"fn:{fname}")
        
        return predicates
    
    def discover_predicates(self):
        """Discover all predicates used in hypotheses."""
        self.all_predicates = Counter()
        self.predicate_cooccurrence = defaultdict(Counter)
        
        for theorem in self.theorems.values():
            hyps = theorem.get('hypotheses', [])
            theorem_predicates = []
            
            for hyp in hyps:
                preds = self.extract_predicates_from_hypothesis(hyp)
                theorem_predicates.extend(preds)
                for pred in preds:
                    self.all_predicates[pred] += 1
            
            # Track co-occurrence
            unique_preds = list(set(theorem_predicates))
            for i, pred1 in enumerate(unique_preds):
                for pred2 in unique_preds[i+1:]:
                    self.predicate_cooccurrence[pred1][pred2] += 1
                    self.predicate_cooccurrence[pred2][pred1] += 1
        
        print(f"\nDiscovered {len(self.all_predicates)} unique predicates")
        print("Top 20 predicates:")
        for pred, count in self.all_predicates.most_common(20):
            print(f"  {pred}: {count}")
    
    def extract_features(self, theorem: Dict) -> Dict:
        """Extract feature vector for a theorem."""
        features = {
            'num_hypotheses': len(theorem.get('hypotheses', [])),
            'num_forall': theorem.get('num_forall', 0),
            'num_exists': theorem.get('num_exists', 0),
            'num_arrows': theorem.get('num_arrows', 0),
            'max_depth': theorem.get('max_nesting_depth', 0),
            'conclusion_head': theorem.get('conclusion_head', 'unknown'),
            'predicates': []
        }
        
        # Extract predicates from hypotheses
        for hyp in theorem.get('hypotheses', []):
            features['predicates'].extend(self.extract_predicates_from_hypothesis(hyp))
        
        # Count predicate occurrences
        pred_counts = Counter(features['predicates'])
        
        # Add features for top predicates
        for pred, _ in self.all_predicates.most_common(30):
            features[f'has_{pred}'] = 1 if pred in pred_counts else 0
            features[f'count_{pred}'] = pred_counts.get(pred, 0)
        
        return features
    
    def cluster_by_hypothesis_patterns(self, n_clusters: int = 10):
        """Cluster theorems by their hypothesis patterns."""
        print("\n" + "="*60)
        print("CLUSTERING BY HYPOTHESIS PATTERNS")
        print("="*60)
        
        # Build feature matrix using top predicates
        top_predicates = [p for p, _ in self.all_predicates.most_common(20)]
        
        theorem_names = []
        feature_vectors = []
        
        for name, theorem in self.theorems.items():
            # Extract predicates from this theorem
            theorem_preds = []
            for hyp in theorem.get('hypotheses', []):
                theorem_preds.extend(self.extract_predicates_from_hypothesis(hyp))
            pred_counts = Counter(theorem_preds)
            
            # Build feature vector
            vector = [
                theorem.get('num_forall', 0),
                theorem.get('num_exists', 0),
                theorem.get('num_arrows', 0),
                len(theorem.get('hypotheses', [])),
            ]
            
            # Add predicate features
            for pred in top_predicates:
                vector.append(pred_counts.get(pred, 0))
            
            theorem_names.append(name)
            feature_vectors.append(vector)
        
        if len(feature_vectors) < n_clusters:
            n_clusters = max(2, len(feature_vectors) // 10)
        
        X = np.array(feature_vectors)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append((theorem_names[i], self.theorems[theorem_names[i]]))
        
        # Report clusters
        for cluster_id in sorted(cluster_groups.keys()):
            members = cluster_groups[cluster_id]
            print(f"\n--- Cluster {cluster_id} ({len(members)} theorems) ---")
            
            # Collect all predicates in this cluster
            cluster_predicates = Counter()
            cluster_conclusions = Counter()
            
            for name, theorem in members:
                for hyp in theorem.get('hypotheses', []):
                    preds = self.extract_predicates_from_hypothesis(hyp)
                    for pred in preds:
                        cluster_predicates[pred] += 1
                cluster_conclusions[theorem.get('conclusion_head', 'unknown')] += 1
            
            # Report cluster characteristics
            print(f"Top predicates: {dict(cluster_predicates.most_common(5))}")
            print(f"Top conclusions: {dict(cluster_conclusions.most_common(3))}")
            
            # Show examples
            print("Examples:")
            for name, theorem in members[:3]:
                print(f"  • {name}")
                hyps = theorem.get('hypotheses', [])
                if hyps:
                    print(f"    Hypotheses:")
                    for h in hyps[:2]:
                        # Clean up for display
                        cleaned = re.sub(r'_fvar\.\d+', '∗', h)[:60]
                        print(f"      - {cleaned}...")
                print(f"    Conclusion: {theorem.get('conclusion_head')}")
    
    def find_theorem_families(self):
        """Find families of theorems with similar predicate patterns."""
        print("\n" + "="*60)
        print("THEOREM FAMILIES BY PREDICATE PATTERNS")
        print("="*60)
        
        # Group by predicate combinations
        pattern_groups = defaultdict(list)
        
        for name, theorem in self.theorems.items():
            # Extract unique predicates from hypotheses
            theorem_preds = set()
            for hyp in theorem.get('hypotheses', []):
                preds = self.extract_predicates_from_hypothesis(hyp)
                # Filter to meaningful predicates only
                for pred in preds:
                    if pred in self.all_predicates and self.all_predicates[pred] >= 3:
                        theorem_preds.add(pred)
            
            # Create signature
            pred_sig = tuple(sorted(theorem_preds))
            concl = theorem.get('conclusion_head', 'unknown')
            
            signature = (pred_sig, concl)
            pattern_groups[signature].append((name, theorem))
        
        # Sort by group size
        sorted_groups = sorted(pattern_groups.items(), key=lambda x: -len(x[1]))
        
        # Report major families
        for (pred_sig, concl), members in sorted_groups[:15]:
            if len(members) < 3:
                continue
            
            pred_str = ', '.join(pred_sig[:5]) if pred_sig else 'no-predicates'
            print(f"\n=== Family: [{pred_str}] → {concl} ({len(members)} theorems) ===")
            
            # Show examples
            for name, theorem in members[:2]:
                print(f"  • {name}")
                for hyp in theorem.get('hypotheses', [])[:2]:
                    cleaned = re.sub(r'_fvar\.\d+', '∗', hyp)[:70]
                    print(f"    H: {cleaned}...")
    
    def analyze_predicate_relationships(self):
        """Analyze which predicates commonly appear together."""
        print("\n" + "="*60)
        print("PREDICATE RELATIONSHIPS")
        print("="*60)
        
        print("\nMost common predicate pairs:")
        all_pairs = []
        for pred1, others in self.predicate_cooccurrence.items():
            for pred2, count in others.items():
                if pred1 < pred2:  # Avoid duplicates
                    all_pairs.append(((pred1, pred2), count))
        
        all_pairs.sort(key=lambda x: -x[1])
        for (pred1, pred2), count in all_pairs[:15]:
            print(f"  {pred1} + {pred2}: {count} theorems")

def main():
    parser = argparse.ArgumentParser(description="Analyze hypothesis patterns")
    parser.add_argument("--jsonl", required=True, help="Path to declaration_structures.jsonl")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    args = parser.parse_args()
    
    analyzer = HypothesisPatternAnalyzer(Path(args.jsonl))
    
    analyzer.cluster_by_hypothesis_patterns(args.clusters)
    analyzer.find_theorem_families()
    analyzer.analyze_predicate_relationships()

if __name__ == "__main__":
    main()