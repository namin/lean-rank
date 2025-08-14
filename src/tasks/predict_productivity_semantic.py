#!/usr/bin/env python3
"""Predict productivity using semantic similarity to existing high-productivity theorems."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text_model(ckpt_path: Path) -> Tuple[nn.Module, dict]:
    """Load the trained text ranking model."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Reconstruct model (matching the actual saved structure)
    class TextEncoder(nn.Module):
        def __init__(self, input_dim: int, emb_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, emb_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    input_dim = checkpoint['config'].get('in_dim', checkpoint['config'].get('input_dim', 313))
    emb_dim = checkpoint['config'].get('emb_dim', 64)
    
    model = TextEncoder(input_dim, emb_dim)
    model.load_state_dict(checkpoint.get('model_state', checkpoint.get('state_dict')))
    model.eval()
    
    return model, checkpoint['config']


def get_embeddings(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Get embeddings for all theorems."""
    with torch.no_grad():
        features_t = torch.FloatTensor(features)
        embeddings = model(features_t).numpy()
    return embeddings


def predict_productivity_by_similarity(
    new_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_productivity: np.ndarray,
    top_k: int = 50
) -> dict:
    """Predict productivity based on similar theorems."""
    
    # Compute cosine similarities
    new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
    ref_norms = reference_embeddings / (np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = ref_norms @ new_norm.T
    similarities = similarities.flatten()
    
    # Find top-k most similar
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    top_productivity = reference_productivity[top_indices]
    
    # Weighted average prediction
    weights = np.exp(top_similarities * 2)  # Temperature scaling
    weights = weights / weights.sum()
    predicted_productivity = np.sum(weights * top_productivity)
    
    # Also compute percentile-based predictions
    predictions = {
        'weighted_mean': predicted_productivity,
        'top10_mean': top_productivity[:10].mean(),
        'top50_mean': top_productivity.mean(),
        'max_similar': top_productivity[0],
        'median_similar': np.median(top_productivity)
    }
    
    return predictions, top_indices, top_similarities


def analyze_productivity_clusters(
    embeddings: np.ndarray,
    productivity: np.ndarray,
    n_clusters: int = 20
) -> pd.DataFrame:
    """Cluster theorems and analyze productivity patterns."""
    from sklearn.cluster import KMeans
    
    # Cluster in embedding space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Analyze each cluster
    cluster_stats = []
    for i in range(n_clusters):
        mask = clusters == i
        cluster_prod = productivity[mask]
        
        cluster_stats.append({
            'cluster': i,
            'size': mask.sum(),
            'mean_productivity': cluster_prod.mean(),
            'median_productivity': np.median(cluster_prod),
            'max_productivity': cluster_prod.max(),
            'high_prod_ratio': (cluster_prod > 100).mean()
        })
    
    return pd.DataFrame(cluster_stats).sort_values('mean_productivity', ascending=False)


def find_productivity_anchors(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    min_productivity: int = 1000
) -> dict:
    """Find high-productivity 'anchor' theorems that define productive regions."""
    
    # Find anchor theorems
    anchors_mask = df['productivity'] >= min_productivity
    anchor_indices = np.where(anchors_mask)[0]
    anchor_embeddings = embeddings[anchor_indices]
    anchor_names = df.iloc[anchor_indices]['name'].values
    anchor_productivity = df.iloc[anchor_indices]['productivity'].values
    
    print(f"\nFound {len(anchor_indices)} anchor theorems with productivity >= {min_productivity}")
    print("\nTop 10 Productivity Anchors:")
    for i in range(min(10, len(anchor_indices))):
        idx = np.argsort(anchor_productivity)[-i-1]
        print(f"  {anchor_names[idx][:50]:50} | productivity={anchor_productivity[idx]:.0f}")
    
    return {
        'indices': anchor_indices,
        'embeddings': anchor_embeddings,
        'names': anchor_names,
        'productivity': anchor_productivity
    }


def predict_new_theorem_productivity(
    type_string: str,
    model: nn.Module,
    features: np.ndarray,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    buckets: int = 128
) -> None:
    """Predict productivity for a new theorem given its type string."""
    
    # Featurize the new type
    from src.tasks.score_graph_whatif import _featurize_new_type
    new_features = _featurize_new_type(type_string, buckets=buckets, target_dim=features.shape[1])
    
    # Get embedding
    with torch.no_grad():
        new_features_t = torch.FloatTensor(new_features)
        new_embedding = model(new_features_t).numpy()
    
    # Predict by similarity
    predictions, top_indices, top_sims = predict_productivity_by_similarity(
        new_embedding,
        embeddings,
        df['productivity'].values,
        top_k=50
    )
    
    print(f"\n{'='*60}")
    print("PRODUCTIVITY PREDICTION FOR NEW THEOREM")
    print(f"{'='*60}")
    print(f"\nType: {type_string[:100]}...")
    
    print(f"\nPredicted Productivity (downstream uses):")
    print(f"  Weighted by similarity:  {predictions['weighted_mean']:>6.1f}")
    print(f"  Mean of top 10 similar:  {predictions['top10_mean']:>6.1f}")
    print(f"  Mean of top 50 similar:  {predictions['top50_mean']:>6.1f}")
    print(f"  Most similar's productivity: {predictions['max_similar']:>6.1f}")
    
    print(f"\nMost similar existing theorems:")
    for i in range(min(10, len(top_indices))):
        idx = top_indices[i]
        print(f"  {df.iloc[idx]['name'][:45]:45} | sim={top_sims[i]:.3f} | prod={df.iloc[idx]['productivity']:.0f}")
    
    # Check distance to high-productivity anchors
    anchors = find_productivity_anchors(df, embeddings, min_productivity=1000)
    if len(anchors['indices']) > 0:
        # Distance to nearest anchor
        anchor_sims = anchors['embeddings'] @ new_embedding.T
        nearest_anchor_idx = np.argmax(anchor_sims)
        nearest_anchor_sim = anchor_sims[nearest_anchor_idx]
        
        print(f"\nNearest high-productivity anchor:")
        print(f"  {anchors['names'][nearest_anchor_idx][:45]:45} | sim={nearest_anchor_sim:.3f} | prod={anchors['productivity'][nearest_anchor_idx]:.0f}")
        
        if nearest_anchor_sim > 0.8:
            print(f"  → High similarity to anchor suggests HIGH productivity potential")
        elif nearest_anchor_sim > 0.6:
            print(f"  → Moderate similarity to anchor suggests MEDIUM productivity potential")
        else:
            print(f"  → Low similarity to anchors suggests LIMITED productivity potential")


def main():
    parser = argparse.ArgumentParser(description="Predict productivity using semantic similarity")
    parser.add_argument("--structures", type=Path, required=True, help="Path to structures.parquet")
    parser.add_argument("--graph_metrics", type=Path, required=True, help="Path to graph_metrics.parquet")
    parser.add_argument("--features", type=Path, required=True, help="Path to structure_features.npz or type_features.npz")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to text_ranker.pt checkpoint")
    parser.add_argument("--type_string", type=str, help="Type string for new theorem to predict")
    parser.add_argument("--analyze_clusters", action="store_true", help="Analyze productivity clusters")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading structures from {args.structures}")
    structures_df = pd.read_parquet(args.structures)
    
    logger.info(f"Loading graph metrics from {args.graph_metrics}")
    graph_df = pd.read_parquet(args.graph_metrics)
    
    # Merge productivity data
    df = structures_df.merge(graph_df[['id', 'out_deg']], on='id', how='inner')
    df['productivity'] = df['out_deg']
    
    # Load features and model
    logger.info(f"Loading features from {args.features}")
    features_data = np.load(args.features)
    if 'X' in features_data:
        features = features_data['X']
    else:
        features = features_data['features']
    
    logger.info(f"Loading model from {args.ckpt}")
    model, config = load_text_model(args.ckpt)
    
    # Get embeddings for all theorems
    logger.info("Computing embeddings for all theorems...")
    embeddings = get_embeddings(model, features)
    
    # Analyze clusters if requested
    if args.analyze_clusters:
        print(f"\n{'='*60}")
        print("PRODUCTIVITY CLUSTERS")
        print(f"{'='*60}")
        cluster_df = analyze_productivity_clusters(embeddings, df['productivity'].values)
        print(cluster_df.to_string())
    
    # Find high-productivity anchors
    anchors = find_productivity_anchors(df, embeddings)
    
    # Predict for new theorem if provided
    if args.type_string:
        predict_new_theorem_productivity(
            args.type_string,
            model,
            features,
            embeddings,
            df,
            buckets=config.get('buckets', 128)
        )
    else:
        # Test on some examples
        print(f"\n{'='*60}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'='*60}")
        
        # Test on existing high-productivity theorems
        high_prod_samples = df.nlargest(5, 'productivity').index
        print("\nValidation: Predicting known high-productivity theorems...")
        
        for idx in high_prod_samples[:3]:
            theorem = df.iloc[idx]
            embedding = embeddings[idx:idx+1]
            
            # Predict using other theorems
            mask = np.ones(len(embeddings), dtype=bool)
            mask[idx] = False  # Exclude self
            
            predictions, _, _ = predict_productivity_by_similarity(
                embedding,
                embeddings[mask],
                df['productivity'].values[mask],
                top_k=50
            )
            
            print(f"\n{theorem['name'][:50]:50}")
            print(f"  Actual: {theorem['productivity']:.0f}, Predicted: {predictions['weighted_mean']:.1f}")


if __name__ == "__main__":
    main()