#!/usr/bin/env python3
"""Train a model to predict use-cost from structural features using actual usage patterns."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

class UseCostModel(nn.Module):
    """MLP to predict use-cost from structural features."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive output
        )
    
    def forward(self, x):
        # Add small constant to ensure minimum cost
        return self.mlp(x) + 0.1


def compute_usage_signal(
    structures: pd.DataFrame,
    graph_metrics: pd.DataFrame = None,
    contexts: str = None
) -> np.ndarray:
    """Compute usage signal from various sources.
    
    Returns inverse usage (high value = low usage = high cost).
    """
    usage = np.ones(len(structures), dtype=np.float32)
    
    # Option 1: Use out-degree from graph (how many things depend on this)
    if graph_metrics is not None and "out_deg" in graph_metrics.columns:
        # Merge by id
        merged = structures.merge(
            graph_metrics[["id", "out_deg"]], 
            on="id", 
            how="left"
        )
        out_deg = merged["out_deg"].fillna(0).values
        # Convert to cost: high usage = low cost
        # Use log to handle heavy-tailed distribution
        usage = 1.0 / (1.0 + np.log1p(out_deg))
        print(f"Using out-degree signal: mean={usage.mean():.3f}, std={usage.std():.3f}")
    
    # Option 2: Count occurrences in contexts (how often used as premise)
    elif contexts:
        import json
        premise_counts = {}
        with open(contexts) as f:
            for line in f:
                ctx = json.loads(line)
                for pid in ctx.get("positives", []):
                    premise_counts[pid] = premise_counts.get(pid, 0) + 1
        
        # Map to structures
        counts = np.array([
            premise_counts.get(id_val, 0) 
            for id_val in structures["id"].values
        ])
        usage = 1.0 / (1.0 + np.log1p(counts))
        print(f"Using context counts: mean={usage.mean():.3f}, std={usage.std():.3f}")
    
    return usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", required=True, help="Path to structures.parquet")
    parser.add_argument("--graph_metrics", default="", help="Path to graph_metrics.parquet for usage signal")
    parser.add_argument("--contexts", default="", help="Path to contexts.jsonl for usage counting")
    parser.add_argument("--out_ckpt", default="outputs/use_cost_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load structural features
    print("Loading structures...")
    structures = pd.read_parquet(args.structures)
    
    # Select only numeric structural feature columns
    meta_cols = ["id", "name", "kind", "type", "use_cost", "binders", "conclusion_head"]
    all_cols = structures.columns.tolist()
    
    # Filter to numeric columns only
    feature_cols = []
    for col in all_cols:
        if col in meta_cols:
            continue
        # Check if column is numeric
        dtype = structures[col].dtype
        if dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
            feature_cols.append(col)
        elif dtype == object:
            # Try to convert boolean-like columns
            sample = structures[col].dropna().iloc[0] if len(structures[col].dropna()) > 0 else None
            if isinstance(sample, (bool, np.bool_)):
                feature_cols.append(col)
    
    print(f"Using {len(feature_cols)} structural features: {feature_cols[:5]}...")
    
    # Convert to numeric array, handling booleans
    X_list = []
    for col in feature_cols:
        values = structures[col].values
        if values.dtype == object:
            # Convert booleans to 0/1
            values = values.astype(float)
        X_list.append(values.reshape(-1, 1))
    
    X = np.hstack(X_list).astype(np.float32)
    
    # Handle any NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Compute usage-based target
    print("Computing usage signal...")
    graph_metrics = None
    if args.graph_metrics:
        graph_metrics = pd.read_parquet(args.graph_metrics)
    
    y = compute_usage_signal(
        structures, 
        graph_metrics,
        args.contexts if args.contexts else None
    )
    
    # Clip to reasonable range and handle any inf/nan
    y = np.clip(y, 0.01, 10.0)
    y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=0.01)
    
    # Split train/val
    n = len(X)
    n_val = int(n * args.val_split)
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()
    y_val = torch.from_numpy(y[val_idx]).float()
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Create model
    model = UseCostModel(in_dim=X.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint = None  # Initialize checkpoint
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = nn.functional.mse_loss(pred, batch_y)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}, batch {i//args.batch_size}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = nn.functional.mse_loss(val_pred, y_val).item()
        
        train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            checkpoint = {
                "state_dict": model.state_dict(),
                "config": {
                    "in_dim": X.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "feature_cols": feature_cols
                },
                "val_loss": val_loss,
                "epoch": epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    if checkpoint is None:
        # No good checkpoint, save current state
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "in_dim": X.shape[1],
                "hidden_dim": args.hidden_dim,
                "feature_cols": feature_cols
            },
            "val_loss": float('inf'),
            "epoch": epoch
        }
    
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out_ckpt)
    print(f"Saved model to {args.out_ckpt}")
    
    # Print some examples
    print("\nExample predictions (lower = easier to use):")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    with torch.no_grad():
        sample_idx = np.random.choice(len(X), size=min(10, len(X)), replace=False)
        sample_X = torch.from_numpy(X[sample_idx])
        sample_pred = model(sample_X).squeeze().numpy()
        sample_true = y[sample_idx]
        
        for i, idx in enumerate(sample_idx):
            name = structures.iloc[idx]["name"]
            print(f"  {name[:50]:50s} pred={sample_pred[i]:.3f} true={sample_true[i]:.3f}")


if __name__ == "__main__":
    main()