"""Build structured declaration features from Lean metaprogramming output."""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lean_extractor(output_path: Path, force: bool = False) -> Path:
    """Run the Lean declaration_structures script if needed."""
    
    if output_path.exists() and not force:
        logger.info(f"Using existing {output_path} ({output_path.stat().st_size // 1024}KB)")
        return output_path
    
    logger.info("Running Lean structure extractor (this will take several minutes)...")
    logger.info("Note: The extractor may timeout on some complex declarations - this is normal")
    
    # Run the Lean script from the lean-training-data directory
    cmd = ["lake", "exe", "declaration_structures", "Mathlib"]
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        try:
            result = subprocess.run(
                cmd,
                cwd="lean-training-data",
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            if result.returncode != 0 and result.stderr:
                # Log warning but continue - some failures are expected
                logger.warning(f"Lean extraction had some errors (this is normal): {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            logger.warning("Lean extraction timed out (this is normal for large libraries)")
    
    logger.info(f"Saved structures to {output_path} ({output_path.stat().st_size // 1024}KB)")
    return output_path

def parse_structures(jsonl_path: Path, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Parse JSONL structures and align with nodes."""
    
    structures = []
    errors = 0
    
    with open(jsonl_path) as f:
        for line_num, line in enumerate(tqdm(f, desc="Parsing structures")):
            if line.strip():
                try:
                    data = json.loads(line)
                    structures.append(data)
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Line {line_num}: JSON decode error: {e}")
    
    if errors > 0:
        logger.warning(f"Total JSON decode errors: {errors}")
    
    logger.info(f"Parsed {len(structures)} structures")
    
    # Convert to DataFrame
    df = pd.DataFrame(structures)
    
    # Align with nodes by name
    df = df.merge(
        nodes_df[['id', 'name']], 
        on='name', 
        how='inner'
    )
    
    logger.info(f"Aligned {len(df)} structures with nodes")
    
    return df

def compute_use_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sophisticated use-cost features from structure."""
    
    # Basic premise burden
    df['premise_cost'] = (
        df['num_explicit_premises'] + 
        df['num_typeclass_constraints'] * 0.5
    )
    
    # Complexity penalty for deep nesting
    df['nesting_penalty'] = np.log1p(df['max_nesting_depth'])
    
    # Polymorphism bonus (more general = more reusable)
    df['polymorphism_bonus'] = df['is_polymorphic'].astype(float) * 0.5
    
    # Classical logic penalty (less constructive)
    df['classical_penalty'] = df['uses_classical'].astype(float) * 0.3
    
    # Decidability bonus (easier to compute with)
    df['decidable_bonus'] = df['has_decidable_instances'].astype(float) * 0.2
    
    # Namespace specificity (deeper = more specialized)
    # Clip to avoid negative values before log1p
    df['specificity_cost'] = np.log1p(np.maximum(0, df['namespace_depth'] - 2))
    
    # Combined sophisticated use-cost
    df['use_cost'] = (
        1.0 + 
        df['premise_cost'] + 
        df['nesting_penalty'] + 
        df['specificity_cost'] +
        df['classical_penalty'] -
        df['polymorphism_bonus'] -
        df['decidable_bonus']
    ).clip(lower=0.1)  # Avoid negative/zero costs
    
    # Additional features for ML
    df['has_premises'] = (df['num_explicit_premises'] > 0).astype(float)
    df['premise_ratio'] = df['num_explicit_premises'] / (1 + df['num_forall'])
    df['implicit_ratio'] = df['num_implicit_args'] / (1 + df['num_forall'])
    
    return df

def create_structured_features(
    structures_df: pd.DataFrame,
    buckets: int = 128,
    include_text_features: bool = True,
    type_features_path: Optional[Path] = None
) -> np.ndarray:
    """Create feature matrix from structures, optionally combined with text features."""
    
    n = len(structures_df)
    
    # Feature groups
    features = []
    
    # 1. Basic counts (normalized)
    count_features = np.column_stack([
        np.log1p(structures_df['num_explicit_premises'].fillna(0)),
        np.log1p(structures_df['num_implicit_args'].fillna(0)),
        np.log1p(structures_df['num_typeclass_constraints'].fillna(0)),
        np.log1p(structures_df['num_forall'].fillna(0)),
        np.log1p(structures_df['num_arrows'].fillna(0)),
        np.log1p(structures_df['max_nesting_depth'].fillna(0)),
        np.log1p(structures_df['conclusion_arity'].fillna(0)),
        np.log1p(structures_df['num_exists'].fillna(0)),  # Add exists count
    ])
    features.append(count_features)
    
    # 2. Binary features
    binary_features = np.column_stack([
        structures_df['uses_classical'].fillna(False).astype(float),
        structures_df['is_polymorphic'].fillna(False).astype(float),
        structures_df['has_decidable_instances'].fillna(False).astype(float),
        structures_df['has_premises'].fillna(False).astype(float),
    ])
    features.append(binary_features)
    
    # 3. Ratio features
    ratio_features = np.column_stack([
        structures_df['premise_ratio'].fillna(0),
        structures_df['implicit_ratio'].fillna(0),
    ])
    features.append(ratio_features)
    
    # 4. Use-cost features
    cost_features = np.column_stack([
        np.log1p(structures_df['use_cost'].fillna(1)),
        structures_df['premise_cost'].fillna(0),
        structures_df['nesting_penalty'].fillna(0),
        structures_df['specificity_cost'].fillna(0),
    ])
    features.append(cost_features)
    
    # 5. Namespace depth as feature
    namespace_features = np.column_stack([
        np.log1p(structures_df['namespace_depth'].fillna(0)),
    ])
    features.append(namespace_features)
    
    # 6. Conclusion head symbol (one-hot encoded with hashing)
    conclusion_heads = structures_df['conclusion_head'].fillna('unknown').values
    head_features = np.zeros((n, buckets // 4))  # Reserve 1/4 of buckets for heads
    for i, head in enumerate(conclusion_heads):
        if pd.notna(head):
            idx = hash(str(head)) % (buckets // 4)
            head_features[i, idx] = 1.0
    features.append(head_features)
    
    # Combine structural features
    X_struct = np.hstack(features).astype(np.float32)
    
    # 7. Don't combine here - just return structural features
    # Combination will happen at the full dataset level in main()
    return X_struct

def create_fallback_features(nodes_df: pd.DataFrame, buckets: int = 128, target_dim: int = None) -> tuple:
    """Create minimal fallback features for nodes without structures."""
    
    n = len(nodes_df)
    
    # Match the dimension of structured features if provided
    if target_dim is not None:
        X = np.zeros((n, target_dim), dtype=np.float32)
    else:
        # Default: 20 basic features + buckets//4 for head symbol
        X = np.zeros((n, 20 + buckets // 4), dtype=np.float32)
    
    # Add some basic features from node names
    for i, name in enumerate(nodes_df['name'].values):
        # Simple heuristics from name
        if 'theorem' in name.lower():
            X[i, 0] = 1.0
        if 'lemma' in name.lower():
            X[i, 1] = 1.0
        if 'def' in name.lower():
            X[i, 2] = 1.0
        
        # Hash the name for some variety
        name_hash = hash(name) % (buckets // 4)
        X[i, 16 + name_hash] = 1.0
    
    # Create default structures DataFrame with ALL columns
    structures_df = pd.DataFrame({
        'id': nodes_df['id'],
        'name': nodes_df['name'],
        'kind': 'unknown',
        'type': '',
        'binders': None,
        'num_explicit_premises': 0,
        'num_implicit_args': 0,
        'num_typeclass_constraints': 0,
        'num_forall': 0,
        'num_exists': 0,
        'num_arrows': 0,
        'max_nesting_depth': 0,
        'conclusion_head': '',
        'conclusion_arity': 0,
        'uses_classical': False,
        'namespace_depth': 0,
        'is_polymorphic': False,
        'has_decidable_instances': False,
        'premise_cost': 0.0,
        'nesting_penalty': 0.0,
        'polymorphism_bonus': 0.0,
        'classical_penalty': 0.0,
        'decidable_bonus': 0.0,
        'specificity_cost': 0.0,
        'use_cost': 1.0,
        'has_premises': False,
        'premise_ratio': 0.0,
        'implicit_ratio': 0.0
    })
    
    return X, structures_df

def main():
    """Main pipeline integration."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=Path, 
                       default=Path('data/processed/nodes.parquet'))
    parser.add_argument('--structures-jsonl', type=Path,
                       default=Path('data/declaration_structures.jsonl'))
    parser.add_argument('--out-structures', type=Path,
                       default=Path('data/processed/structures.parquet'))
    parser.add_argument('--out-features', type=Path,
                       default=Path('data/processed/structure_features.npz'))
    parser.add_argument('--buckets', type=int, default=128)
    parser.add_argument('--force', action='store_true',
                       help='Force re-extraction even if file exists')
    parser.add_argument('--combine-with-text', action='store_true',
                       help='Combine structural features with original text features')
    parser.add_argument('--text-features', type=Path,
                       default=Path('data/processed/type_features.npz'),
                       help='Path to original text features to combine with')
    args = parser.parse_args()
    
    # Load nodes
    nodes_df = pd.read_parquet(args.nodes)
    logger.info(f"Loaded {len(nodes_df)} nodes")
    
    # Run Lean extractor if needed
    jsonl_path = run_lean_extractor(args.structures_jsonl, force=args.force)
    
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        # Parse and align
        structures_df = parse_structures(jsonl_path, nodes_df)
        
        if len(structures_df) > 0:
            # Compute use-cost features
            structures_df = compute_use_cost_features(structures_df)
            
            # Create feature matrix
            X = create_structured_features(
                structures_df, 
                args.buckets,
                include_text_features=False  # Don't combine yet
            )
            
            # Handle nodes without structures by filling with defaults
            if len(structures_df) < len(nodes_df):
                logger.warning(f"Only {len(structures_df)}/{len(nodes_df)} nodes have structures")
                missing_ids = set(nodes_df['id']) - set(structures_df['id'])
                missing_nodes = nodes_df[nodes_df['id'].isin(missing_ids)]
                
                # Create default features for missing nodes with matching dimensions
                missing_X, missing_structs = create_fallback_features(missing_nodes, args.buckets, target_dim=X.shape[1])
                
                # Combine
                full_X = np.zeros((len(nodes_df), X.shape[1]), dtype=np.float32)
                struct_id_to_idx = {sid: i for i, sid in enumerate(structures_df['id'])}
                missing_id_to_idx = {mid: i for i, mid in enumerate(missing_nodes['id'])}
                
                for i, node_id in enumerate(nodes_df['id']):
                    if node_id in struct_id_to_idx:
                        full_X[i] = X[struct_id_to_idx[node_id]]
                    elif node_id in missing_id_to_idx:
                        full_X[i] = missing_X[missing_id_to_idx[node_id]]
                
                X = full_X
                structures_df = pd.concat([structures_df, missing_structs], ignore_index=True)
        else:
            logger.warning("No structures parsed successfully, using fallback features")
            X, structures_df = create_fallback_features(nodes_df, args.buckets)
    else:
        logger.warning("No structures file found, using fallback features")
        X, structures_df = create_fallback_features(nodes_df, args.buckets)
    
    # Sort by node id to ensure alignment
    # Ensure structures_df has same order as nodes_df
    id_to_idx = {sid: i for i, sid in enumerate(structures_df['id'])}
    reorder_idx = [id_to_idx[nid] for nid in nodes_df['id'] if nid in id_to_idx]
    
    # If we have all nodes (after filling missing), reorder to match nodes_df
    if len(structures_df) == len(nodes_df):
        reorder_idx = [id_to_idx[nid] for nid in nodes_df['id']]
        X = X[reorder_idx]
        structures_df = structures_df.iloc[reorder_idx].reset_index(drop=True)
    
    # Combine with text features if requested
    if args.combine_with_text and args.text_features.exists():
        try:
            X_text = np.load(args.text_features)['X'].astype(np.float32)
            if X_text.shape[0] == X.shape[0]:
                logger.info(f"Combining structural features {X.shape} with text features {X_text.shape}")
                X = np.hstack([X, X_text])
            else:
                logger.warning(f"Text features shape {X_text.shape[0]} doesn't match {X.shape[0]}, using only structural")
        except Exception as e:
            logger.warning(f"Could not load text features: {e}")
    
    # Save structures
    structures_df.to_parquet(args.out_structures)
    logger.info(f"Saved structures to {args.out_structures}")
    
    # Save features
    np.savez_compressed(args.out_features, X=X)
    logger.info(f"Saved features {X.shape} to {args.out_features}")
    
    # Print statistics
    print("\n=== Structure Statistics ===")
    print(f"Total declarations: {len(structures_df)}")
    if 'num_explicit_premises' in structures_df.columns:
        print(f"Avg explicit premises: {structures_df['num_explicit_premises'].mean():.2f}")
        print(f"Avg use-cost: {structures_df['use_cost'].mean():.2f}")
        print(f"Has premises: {structures_df['has_premises'].mean():.1%}")
        if 'uses_classical' in structures_df.columns:
            print(f"Uses classical: {structures_df['uses_classical'].mean():.1%}")
        if 'is_polymorphic' in structures_df.columns:
            print(f"Is polymorphic: {structures_df['is_polymorphic'].mean():.1%}")

if __name__ == '__main__':
    main()