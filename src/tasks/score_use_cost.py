#!/usr/bin/env python3
"""Score use-cost for declarations using learned model."""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.tasks.train_use_cost_model import UseCostModel


def predict_use_cost(
    model: UseCostModel,
    structures: pd.DataFrame,
    feature_cols: list
) -> np.ndarray:
    """Predict use-cost for all declarations."""
    X = structures[feature_cols].values.astype(np.float32)
    X_tensor = torch.from_numpy(X)
    
    model.eval()
    with torch.no_grad():
        use_costs = model(X_tensor).squeeze().numpy()
    
    return use_costs


def predict_use_cost_from_similar(
    structures: pd.DataFrame,
    similar_ids: list,
    top_k: int = 10
) -> float:
    """Estimate use-cost from similar declarations (fallback)."""
    similar = structures[structures["id"].isin(similar_ids[:top_k])]
    
    if len(similar) == 0:
        return 1.0  # Default
    
    # Use learned use-cost if available, else computed use-cost
    if "learned_use_cost" in similar.columns:
        return float(similar["learned_use_cost"].median())
    elif "use_cost" in similar.columns:
        return float(similar["use_cost"].median())
    else:
        # Fallback to structural heuristic
        explicit = similar["num_explicit_premises"].median()
        return 1.0 + explicit * 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to trained use_cost_model.pt")
    parser.add_argument("--out", required=True, help="Output path for structures with learned use-cost")
    parser.add_argument("--similar_ids", default="", help="JSON file with similar declaration IDs for new statements")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint["config"]
    
    model = UseCostModel(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    
    # Load structures
    print(f"Loading structures from {args.structures}")
    structures = pd.read_parquet(args.structures)
    
    # Predict use-cost
    print("Predicting use-cost...")
    feature_cols = config["feature_cols"]
    use_costs = predict_use_cost(model, structures, feature_cols)
    
    # Add to dataframe
    structures["learned_use_cost"] = use_costs
    
    # If we have similar IDs for a new statement, predict its use-cost
    if args.similar_ids and Path(args.similar_ids).exists():
        with open(args.similar_ids) as f:
            similar_data = json.load(f)
            similar_ids = similar_data.get("similar_ids", [])
            new_use_cost = predict_use_cost_from_similar(structures, similar_ids)
            print(f"Predicted use-cost for new statement: {new_use_cost:.3f}")
            
            # Save to file for what-if to use
            similar_data["predicted_use_cost"] = float(new_use_cost)
            with open(args.similar_ids, "w") as f:
                json.dump(similar_data, f, indent=2)
    
    # Save enhanced structures
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    structures.to_parquet(args.out)
    print(f"Saved to {args.out}")
    
    # Print statistics
    print(f"\nUse-cost statistics:")
    print(f"  Mean: {use_costs.mean():.3f}")
    print(f"  Std:  {use_costs.std():.3f}")
    print(f"  Min:  {use_costs.min():.3f}")
    print(f"  Max:  {use_costs.max():.3f}")
    
    # Show examples of low and high cost
    print("\nEasiest to use (low cost):")
    easiest = structures.nsmallest(5, "learned_use_cost")
    for _, row in easiest.iterrows():
        print(f"  {row['name'][:60]:60s} cost={row['learned_use_cost']:.3f}")
    
    print("\nHardest to use (high cost):")
    hardest = structures.nlargest(5, "learned_use_cost")
    for _, row in hardest.iterrows():
        print(f"  {row['name'][:60]:60s} cost={row['learned_use_cost']:.3f}")


if __name__ == "__main__":
    main()