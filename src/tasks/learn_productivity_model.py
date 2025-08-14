#!/usr/bin/env python3
"""Learn to predict theorem productivity from structural features."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_productivity_data(structures_df: pd.DataFrame, graph_df: pd.DataFrame):
    """Prepare features and productivity labels."""
    
    # Merge to get productivity signal
    df = structures_df.merge(graph_df[['id', 'out_deg', 'in_deg']], on='id', how='inner')
    
    # Define productivity: how many theorems use this one
    df['productivity'] = df['out_deg']
    df['log_productivity'] = np.log1p(df['out_deg'])
    
    # Create productivity categories
    df['productivity_level'] = pd.cut(
        df['out_deg'],
        bins=[-0.5, 0.5, 10.5, 100.5, float('inf')],
        labels=['unused', 'low', 'medium', 'high']
    )
    
    print("\n" + "="*60)
    print("PRODUCTIVITY DISTRIBUTION")
    print("="*60)
    print(df['productivity_level'].value_counts().to_string())
    print(f"\nMean productivity: {df['productivity'].mean():.1f}")
    print(f"Median productivity: {df['productivity'].median():.1f}")
    print(f"Max productivity: {df['productivity'].max():,}")
    
    # Select features
    feature_cols = [
        'num_explicit_premises',
        'num_implicit_args', 
        'num_typeclass_constraints',
        'num_forall',
        'num_exists',
        'num_arrows',
        'max_nesting_depth',
        'namespace_depth',
        'conclusion_arity',
        'uses_classical',
        'is_polymorphic',
        'has_decidable_instances'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    
    # Add interaction features
    if 'num_explicit_premises' in df.columns and 'max_nesting_depth' in df.columns:
        df['premises_x_depth'] = df['num_explicit_premises'] * df['max_nesting_depth']
        available_features.append('premises_x_depth')
    
    if 'num_forall' in df.columns and 'num_arrows' in df.columns:
        df['forall_minus_arrows'] = df['num_forall'] - df['num_arrows']
        available_features.append('forall_minus_arrows')
    
    return df, available_features


def train_productivity_models(df: pd.DataFrame, features: list):
    """Train multiple models to predict productivity."""
    
    print("\n" + "="*60)
    print("TRAINING PRODUCTIVITY MODELS")
    print("="*60)
    
    # Prepare data
    X = df[features].fillna(0)
    y = df['productivity'].values
    y_log = df['log_productivity'].values
    
    # Split data
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Linear Regression on log(productivity)
    print("\n1. Linear Regression (log-transformed):")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_log_train)
    y_log_pred = lr.predict(X_test_scaled)
    y_pred = np.expm1(y_log_pred)  # Transform back
    
    results['linear'] = {
        'model': lr,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    print(f"  R² = {results['linear']['r2']:.3f}")
    print(f"  MAE = {results['linear']['mae']:.1f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': lr.coef_,
        'abs_coef': np.abs(lr.coef_)
    }).sort_values('abs_coef', ascending=False)
    
    print("\n  Top feature coefficients (for log productivity):")
    for _, row in feature_importance.head(10).iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        print(f"    {row['feature']:30} : {row['coefficient']:+.3f} {direction}")
    
    # 2. Poisson Regression (for count data)
    print("\n2. Poisson Regression:")
    poisson = PoissonRegressor(max_iter=1000)
    poisson.fit(X_train_scaled, y_train)
    y_pred_poisson = poisson.predict(X_test_scaled)
    
    results['poisson'] = {
        'model': poisson,
        'r2': r2_score(y_test, y_pred_poisson),
        'mae': mean_absolute_error(y_test, y_pred_poisson),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_poisson))
    }
    print(f"  R² = {results['poisson']['r2']:.3f}")
    print(f"  MAE = {results['poisson']['mae']:.1f}")
    
    # 3. Random Forest
    print("\n3. Random Forest:")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['random_forest'] = {
        'model': rf,
        'r2': r2_score(y_test, y_pred_rf),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf))
    }
    print(f"  R² = {results['random_forest']['r2']:.3f}")
    print(f"  MAE = {results['random_forest']['mae']:.1f}")
    
    # Feature importance from Random Forest
    feature_importance_rf = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top features by importance:")
    for _, row in feature_importance_rf.head(10).iterrows():
        print(f"    {row['feature']:30} : {row['importance']:.3f}")
    
    # 4. Gradient Boosting
    print("\n4. Gradient Boosting:")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    results['gradient_boosting'] = {
        'model': gb,
        'r2': r2_score(y_test, y_pred_gb),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb))
    }
    print(f"  R² = {results['gradient_boosting']['r2']:.3f}")
    print(f"  MAE = {results['gradient_boosting']['mae']:.1f}")
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['r2']:>8.3f} {metrics['mae']:>10.1f} {metrics['rmse']:>10.1f}")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest model: {best_model[0]} (R² = {best_model[1]['r2']:.3f})")
    
    return results, scaler


def analyze_productivity_patterns(df: pd.DataFrame, features: list):
    """Analyze what makes theorems productive."""
    
    print("\n" + "="*60)
    print("PRODUCTIVITY PATTERNS")
    print("="*60)
    
    # High productivity theorems
    high_prod = df[df['productivity'] > 100]
    low_prod = df[df['productivity'] == 0]
    
    print(f"\nHigh productivity (>100 uses): {len(high_prod):,} theorems")
    print(f"Zero productivity (unused): {len(low_prod):,} theorems")
    
    print("\n" + "-"*40)
    print("Feature comparison (high vs zero productivity):")
    print(f"{'Feature':<30} {'High Prod':>12} {'Zero Prod':>12} {'Ratio':>8}")
    print("-" * 65)
    
    for feat in features:
        if feat in df.columns:
            high_mean = high_prod[feat].mean()
            low_mean = low_prod[feat].mean()
            if low_mean > 0:
                ratio = high_mean / low_mean
            else:
                ratio = float('inf') if high_mean > 0 else 1.0
            print(f"{feat:<30} {high_mean:>12.2f} {low_mean:>12.2f} {ratio:>8.2f}x")
    
    # Productivity by number of premises
    print("\n" + "-"*40)
    print("Average productivity by number of premises:")
    premise_prod = df.groupby('num_explicit_premises')['productivity'].agg(['mean', 'median', 'count'])
    print(premise_prod.head(10).to_string())
    
    # Find "productivity sweet spot"
    print("\n" + "-"*40)
    print("Productivity Sweet Spot Analysis:")
    
    # Theorems with 1-3 premises and moderate depth
    sweet_spot = df[
        (df['num_explicit_premises'].between(1, 3)) & 
        (df['max_nesting_depth'].between(3, 8)) &
        (df['num_forall'] <= 3)
    ]
    
    print(f"Sweet spot criteria: 1-3 premises, depth 3-8, ≤3 foralls")
    print(f"  Theorems matching: {len(sweet_spot):,} ({100*len(sweet_spot)/len(df):.1f}%)")
    print(f"  Mean productivity: {sweet_spot['productivity'].mean():.1f}")
    print(f"  vs overall mean: {df['productivity'].mean():.1f}")
    print(f"  Productivity boost: {sweet_spot['productivity'].mean() / df['productivity'].mean():.2f}x")
    
    # Top productive theorems
    print("\n" + "-"*40)
    print("Top 10 most productive theorems:")
    top_prod = df.nlargest(10, 'productivity')[['name', 'productivity', 'num_explicit_premises', 'max_nesting_depth']]
    for i, row in enumerate(top_prod.itertuples(), 1):
        print(f"{i:2}. {row.name[:45]:45} | enables {row.productivity:6.0f} | premises={row.num_explicit_premises}")


def predict_new_theorem_productivity(
    models: dict,
    scaler: StandardScaler,
    features: list,
    num_premises: int = 2,
    max_depth: int = 5,
    num_forall: int = 1
):
    """Predict productivity for a hypothetical new theorem."""
    
    print("\n" + "="*60)
    print("PRODUCTIVITY PREDICTION FOR NEW THEOREM")
    print("="*60)
    
    # Create feature vector
    new_theorem = pd.DataFrame({
        'num_explicit_premises': [num_premises],
        'num_implicit_args': [2],
        'num_typeclass_constraints': [1],
        'num_forall': [num_forall],
        'num_exists': [0],
        'num_arrows': [num_premises],
        'max_nesting_depth': [max_depth],
        'namespace_depth': [3],
        'conclusion_arity': [1],
        'uses_classical': [0],
        'is_polymorphic': [1],
        'has_decidable_instances': [0]
    })
    
    # Add interaction features
    new_theorem['premises_x_depth'] = num_premises * max_depth
    new_theorem['forall_minus_arrows'] = num_forall - num_premises
    
    # Align with trained features
    for feat in features:
        if feat not in new_theorem.columns:
            new_theorem[feat] = 0
    
    # Create DataFrames with feature names to avoid sklearn warnings
    X_new_df = pd.DataFrame(new_theorem[features].values, columns=features)
    X_new_scaled = scaler.transform(X_new_df)
    
    print(f"\nTheorem characteristics:")
    print(f"  Explicit premises: {num_premises}")
    print(f"  Max nesting depth: {max_depth}")
    print(f"  Universal quantifiers: {num_forall}")
    
    print(f"\nPredicted productivity (downstream uses):")
    for name, result in models.items():
        model = result['model']
        if name == 'linear':
            # Linear model was trained on log
            pred_log = model.predict(X_new_scaled)[0]
            pred = np.expm1(pred_log)
        elif name in ['random_forest', 'gradient_boosting']:
            # Tree models don't need scaling but need DataFrame
            pred = model.predict(X_new_df)[0]
        else:
            pred = model.predict(X_new_scaled)[0]
        
        print(f"  {name:<20}: {pred:>6.1f} theorems")


def main():
    parser = argparse.ArgumentParser(description="Learn productivity from structural features")
    parser.add_argument("--structures", type=Path, required=True, help="Path to structures.parquet")
    parser.add_argument("--graph_metrics", type=Path, required=True, help="Path to graph_metrics.parquet")
    parser.add_argument("--save_model", type=Path, help="Save best model to file")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading structures from {args.structures}")
    structures_df = pd.read_parquet(args.structures)
    
    logger.info(f"Loading graph metrics from {args.graph_metrics}")
    graph_df = pd.read_parquet(args.graph_metrics)
    
    # Prepare data
    df, features = prepare_productivity_data(structures_df, graph_df)
    
    # Train models
    models, scaler = train_productivity_models(df, features)
    
    # Analyze patterns
    analyze_productivity_patterns(df, features)
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Test different theorem profiles
    profiles = [
        (0, 3, 0, "Simple constant/constructor"),
        (1, 5, 1, "Basic lemma"),
        (2, 6, 2, "Standard theorem"),
        (3, 8, 3, "Complex theorem"),
        (5, 10, 5, "Very complex theorem"),
    ]
    
    for premises, depth, foralls, description in profiles:
        print(f"\n{description}:")
        predict_new_theorem_productivity(
            models, scaler, features,
            num_premises=premises,
            max_depth=depth,
            num_forall=foralls
        )
    
    # Save best model if requested
    if args.save_model:
        best_name = max(models.items(), key=lambda x: x[1]['r2'])[0]
        best_model = models[best_name]['model']
        
        import joblib
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'features': features,
            'model_type': best_name,
            'r2_score': models[best_name]['r2']
        }, args.save_model)
        
        logger.info(f"Saved {best_name} model to {args.save_model}")
    
    print("\n" + "="*60)
    print("PRODUCTIVITY ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()