#!/usr/bin/env python3
"""
Create embeddings using hypothesis patterns for better premise selection.
"""

import json
import numpy as np
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F

class HypothesisEmbedder:
    def __init__(self, jsonl_path: Path):
        self.theorems = {}
        self.load_structures(jsonl_path)
        self.build_predicate_vocabulary()
        
    def load_structures(self, path: Path):
        """Load declaration_structures.jsonl"""
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
    
    def extract_predicates(self, hyp_str: str) -> List[str]:
        """Extract predicates from hypothesis string."""
        predicates = []
        cleaned = re.sub(r'_fvar\.\d+', 'X', hyp_str)
        
        # Skip basic operators
        skip = {'Eq', 'LE', 'LT', 'GE', 'GT', 'Ne', 'HAdd', 'HMul', 'List', 'Set'}
        
        # Extract meaningful predicates
        capital_names = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)*\b', cleaned)
        for name in capital_names:
            base = name.split('.')[-1] if '.' in name else name
            if base not in skip:
                predicates.append(name)
        
        # Add operator categories
        if '∣' in cleaned or 'Dvd' in cleaned:
            predicates.append('_DIVIDES_')
        if 'gcd' in cleaned.lower():
            predicates.append('_GCD_')
        if '≡' in cleaned or 'mod' in cleaned.lower():
            predicates.append('_MODULAR_')
            
        return predicates
    
    def build_predicate_vocabulary(self):
        """Build vocabulary of all predicates."""
        self.predicate_counts = Counter()
        self.predicate_cooccurrence = defaultdict(Counter)
        
        for theorem in self.theorems.values():
            preds = []
            for hyp in theorem.get('hypotheses', []):
                preds.extend(self.extract_predicates(hyp))
            
            unique_preds = list(set(preds))
            for pred in unique_preds:
                self.predicate_counts[pred] += 1
            
            # Track co-occurrence
            for i, p1 in enumerate(unique_preds):
                for p2 in unique_preds[i+1:]:
                    self.predicate_cooccurrence[p1][p2] += 1
                    self.predicate_cooccurrence[p2][p1] += 1
        
        # Create vocabulary (top predicates + special tokens)
        self.vocab = ['<PAD>', '<UNK>']
        self.vocab.extend([p for p, _ in self.predicate_counts.most_common(100)])
        self.pred2idx = {p: i for i, p in enumerate(self.vocab)}
        
        print(f"Built vocabulary of {len(self.vocab)} predicates")
        print(f"Top predicates: {self.vocab[2:12]}")
    
    def create_embedding_features(self, theorem: Dict) -> Dict:
        """Create rich feature representation for a theorem."""
        features = {}
        
        # 1. Structural features (existing in lean-rank)
        features['num_forall'] = theorem.get('num_forall', 0)
        features['num_exists'] = theorem.get('num_exists', 0)
        features['num_arrows'] = theorem.get('num_arrows', 0)
        features['max_depth'] = theorem.get('max_nesting_depth', 0)
        
        # 2. Hypothesis predicate features
        hyp_preds = []
        for hyp in theorem.get('hypotheses', []):
            hyp_preds.extend(self.extract_predicates(hyp))
        
        # Binary features for top predicates
        pred_vector = np.zeros(len(self.vocab))
        for pred in hyp_preds:
            if pred in self.pred2idx:
                pred_vector[self.pred2idx[pred]] = 1
            else:
                pred_vector[1] = 1  # UNK token
        features['predicate_vector'] = pred_vector
        
        # 3. Predicate statistics
        features['num_unique_predicates'] = len(set(hyp_preds))
        features['num_total_predicates'] = len(hyp_preds)
        
        # 4. Conclusion features
        conclusion_head = theorem.get('conclusion_head', 'unknown')
        features['conclusion_type'] = conclusion_head
        
        # 5. Hypothesis-conclusion alignment
        # Does conclusion match hypothesis predicates?
        if conclusion_head in hyp_preds:
            features['aligned'] = 1
        else:
            features['aligned'] = 0
        
        return features
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise similarity between all theorems."""
        theorem_names = list(self.theorems.keys())
        n = len(theorem_names)
        
        # Extract features for all theorems
        feature_vectors = []
        for name in theorem_names:
            features = self.create_embedding_features(self.theorems[name])
            feature_vectors.append(features['predicate_vector'])
        
        X = np.array(feature_vectors)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(X)
        
        return similarity, theorem_names
    
    def find_similar_by_hypotheses(self, target_name: str, k: int = 10) -> List[Tuple[str, float]]:
        """Find theorems with similar hypothesis patterns."""
        if target_name not in self.theorems:
            return []
        
        target_features = self.create_embedding_features(self.theorems[target_name])
        target_vec = target_features['predicate_vector']
        
        similarities = []
        for name, theorem in self.theorems.items():
            if name == target_name:
                continue
            
            features = self.create_embedding_features(theorem)
            vec = features['predicate_vector']
            
            # Cosine similarity
            sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10)
            similarities.append((name, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def learn_predicate_embeddings(self, dim: int = 32):
        """Learn embeddings for predicates based on co-occurrence."""
        # Build co-occurrence matrix
        vocab_size = len(self.vocab)
        cooc_matrix = np.zeros((vocab_size, vocab_size))
        
        for p1, others in self.predicate_cooccurrence.items():
            if p1 in self.pred2idx:
                i = self.pred2idx[p1]
                for p2, count in others.items():
                    if p2 in self.pred2idx:
                        j = self.pred2idx[p2]
                        cooc_matrix[i, j] = count
        
        # Use SVD to get embeddings (like word2vec)
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=dim, random_state=42)
        predicate_embeddings = svd.fit_transform(cooc_matrix)
        
        self.predicate_embeddings = predicate_embeddings
        return predicate_embeddings
    
    def create_graph_features(self) -> Dict:
        """Create graph-based features from predicate relationships."""
        import networkx as nx
        
        # Build predicate co-occurrence graph
        G = nx.Graph()
        
        for p1, others in self.predicate_cooccurrence.items():
            for p2, weight in others.items():
                if weight > 2:  # Filter weak connections
                    G.add_edge(p1, p2, weight=weight)
        
        # Compute graph statistics
        features = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
        }
        
        # Find communities
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(G))
        features['num_communities'] = len(communities)
        
        # Store for later use
        self.predicate_graph = G
        self.communities = communities
        
        print(f"Predicate graph: {features['num_nodes']} nodes, {features['num_edges']} edges")
        print(f"Found {features['num_communities']} communities")
        
        return features


class HypothesisAwareRanker(nn.Module):
    """Neural ranker that uses hypothesis patterns."""
    
    def __init__(self, vocab_size: int, struct_dim: int = 10, embed_dim: int = 32):
        super().__init__()
        
        # Predicate embedding
        self.pred_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Structural features encoder
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Combine and rank
        self.ranker = nn.Sequential(
            nn.Linear(embed_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, pred_indices, struct_features):
        # Embed predicates
        pred_embeds = self.pred_embed(pred_indices)
        pred_pooled = torch.mean(pred_embeds, dim=1)  # Average pooling
        
        # Encode structure
        struct_encoded = self.struct_encoder(struct_features)
        
        # Combine and score
        combined = torch.cat([pred_pooled, struct_encoded], dim=1)
        score = self.ranker(combined)
        
        return score


def analyze_hypothesis_embeddings(jsonl_path: Path):
    """Run analysis of hypothesis-based embeddings."""
    embedder = HypothesisEmbedder(jsonl_path)
    
    # Learn predicate embeddings
    print("\n=== LEARNING PREDICATE EMBEDDINGS ===")
    pred_embeds = embedder.learn_predicate_embeddings(dim=32)
    print(f"Learned {pred_embeds.shape[0]} x {pred_embeds.shape[1]} embeddings")
    
    # Build predicate graph
    print("\n=== PREDICATE RELATIONSHIP GRAPH ===")
    graph_features = embedder.create_graph_features()
    
    # Test similarity search
    print("\n=== HYPOTHESIS-BASED SIMILARITY ===")
    test_theorem = list(embedder.theorems.keys())[100]
    print(f"Finding similar to: {test_theorem}")
    similar = embedder.find_similar_by_hypotheses(test_theorem, k=5)
    
    test_obj = embedder.theorems[test_theorem]
    print(f"Target hypotheses: {test_obj.get('hypotheses', [])[:2]}")
    print("\nSimilar theorems:")
    for name, sim in similar:
        obj = embedder.theorems[name]
        print(f"  {sim:.3f}: {name}")
        if obj.get('hypotheses'):
            print(f"         {obj['hypotheses'][0][:60]}...")
    
    return embedder


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to declaration_structures.jsonl")
    args = parser.parse_args()
    
    embedder = analyze_hypothesis_embeddings(Path(args.jsonl))
    
    # Save embeddings for later use
    output_path = Path(args.jsonl).parent / "hypothesis_embeddings.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'vocab': embedder.vocab,
            'predicate_embeddings': embedder.predicate_embeddings,
            'predicate_counts': dict(embedder.predicate_counts)
        }, f)
    print(f"\nSaved embeddings to {output_path}")


if __name__ == "__main__":
    main()
