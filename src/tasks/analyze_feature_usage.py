#!/usr/bin/env python3
"""Analyze relationship between structural features and usage patterns."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_usage_patterns(structures_df: pd.DataFrame, graph_df: pd.DataFrame):
    """Analyze how structural features relate to usage."""
    
    # Merge data
    df = structures_df.merge(graph_df[['id', 'out_deg']], on='id', how='inner')
    
    # Create usage indicators
    df['is_used'] = (df['out_deg'] > 0).astype(int)
    df['is_frequently_used'] = (df['out_deg'] > 10).astype(int)
    df['is_heavily_used'] = (df['out_deg'] > 100).astype(int)
    
    # Print basic statistics
    print("\n" + "="*60)
    print("USAGE STATISTICS")
    print("="*60)
    print(f"Total declarations analyzed: {len(df):,}")
    print(f"Used at least once: {df['is_used'].sum():,} ({100*df['is_used'].mean():.1f}%)")
    print(f"Used >10 times: {df['is_frequently_used'].sum():,} ({100*df['is_frequently_used'].mean():.1f}%)")
    print(f"Used >100 times: {df['is_heavily_used'].sum():,} ({100*df['is_heavily_used'].mean():.1f}%)")
    print(f"\nOut-degree statistics:")
    print(f"  Mean: {df['out_deg'].mean():.1f}")
    print(f"  Median: {df['out_deg'].median():.1f}")
    print(f"  Max: {df['out_deg'].max():,}")
    
    return df


def analyze_premises_effect(df: pd.DataFrame):
    """Analyze how number of premises affects usage."""
    
    print("\n" + "="*60)
    print("USAGE BY NUMBER OF EXPLICIT PREMISES")
    print("="*60)
    
    # Group by exact premise count (up to 10)
    premise_stats = df[df['num_explicit_premises'] <= 10].groupby('num_explicit_premises').agg({
        'is_used': 'mean',
        'is_frequently_used': 'mean',
        'out_deg': ['mean', 'median', 'std'],
        'id': 'count'
    }).round(3)
    
    premise_stats.columns = ['prob_used', 'prob_freq_used', 'mean_usage', 'median_usage', 'std_usage', 'n_decls']
    print(premise_stats.to_string())
    
    # Statistical test
    print("\n" + "-"*40)
    # Create buckets for chi-square test
    df['premise_bucket'] = pd.cut(df['num_explicit_premises'], 
                                  bins=[-0.5, 0.5, 1.5, 2.5, 5.5, 100], 
                                  labels=['0', '1', '2', '3-5', '6+'])
    
    # Chi-square test
    contingency = pd.crosstab(df['premise_bucket'], df['is_used'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square test (premises vs usage):")
    print(f"  χ² = {chi2:.2f}, p-value = {p_value:.4e}")
    if p_value < 0.001:
        print("  ✓ HIGHLY SIGNIFICANT relationship between premises and usage")
    elif p_value < 0.05:
        print("  ✓ Significant relationship between premises and usage")
    else:
        print("  ✗ No significant relationship found")
    
    # ANOVA for continuous outcome
    groups = [group['out_deg'].values for name, group in df.groupby('premise_bucket')]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\nANOVA test (premises vs out-degree):")
    print(f"  F = {f_stat:.2f}, p-value = {p_anova:.4e}")
    
    return df


def analyze_feature_importance(df: pd.DataFrame):
    """Analyze which features predict usage using logistic regression."""
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE FOR PREDICTING USAGE")
    print("="*60)
    
    features = [
        'num_explicit_premises',
        'num_implicit_args', 
        'num_typeclass_constraints',
        'num_forall',
        'num_exists',
        'num_arrows',
        'max_nesting_depth',
        'namespace_depth',
        'conclusion_arity'
    ]
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['is_used']
    
    # Standardize for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    
    # Print coefficients
    print("\nLogistic Regression Coefficients (standardized):")
    print("(Positive = increases usage probability)")
    print("-"*40)
    
    coef_df = pd.DataFrame({
        'feature': available_features,
        'coefficient': lr.coef_[0],
        'abs_coef': np.abs(lr.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    
    for _, row in coef_df.iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        print(f"{row['feature']:30} : {row['coefficient']:+.3f} {direction}")
    
    print(f"\nModel Accuracy: {lr.score(X_scaled, y):.3f}")
    
    # Feature correlations with usage
    print("\n" + "-"*40)
    print("Raw Correlations with Usage:")
    correlations = []
    for feat in available_features:
        corr = df[feat].corr(df['out_deg'])
        correlations.append((feat, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in correlations:
        print(f"{feat:30} : {corr:+.3f}")
    
    return lr, scaler, available_features


def find_usage_patterns(df: pd.DataFrame):
    """Find interesting usage patterns."""
    
    print("\n" + "="*60)
    print("INTERESTING USAGE PATTERNS")
    print("="*60)
    
    # Most used declarations
    print("\nTop 10 Most Used Declarations:")
    print("-"*40)
    top_used = df.nlargest(10, 'out_deg')[['name', 'out_deg', 'num_explicit_premises', 
                                            'num_typeclass_constraints', 'max_nesting_depth']]
    for i, row in enumerate(top_used.itertuples(), 1):
        print(f"{i:2}. {row.name[:50]:50} | used={row.out_deg:6} | premises={row.num_explicit_premises}")
    
    # Sweet spot analysis
    print("\n" + "-"*40)
    print("'Sweet Spot' Theorems (1-2 premises, high usage):")
    sweet_spot = df[(df['num_explicit_premises'].between(1, 2)) & 
                    (df['out_deg'] > 100)].sort_values('out_deg', ascending=False).head(10)
    
    for i, row in enumerate(sweet_spot.itertuples(), 1):
        print(f"{i:2}. {row.name[:50]:50} | used={row.out_deg:6} | premises={row.num_explicit_premises}")
    
    # Complex but unused
    print("\n" + "-"*40)
    print("Complex but Unused (many premises, zero usage):")
    complex_unused = df[(df['num_explicit_premises'] >= 5) & 
                        (df['out_deg'] == 0)].head(10)
    
    for i, row in enumerate(complex_unused.itertuples(), 1):
        print(f"{i:2}. {row.name[:50]:50} | premises={row.num_explicit_premises} | depth={row.max_nesting_depth}")
    
    # By theorem type
    if 'kind' in df.columns:
        print("\n" + "-"*40)
        print("Usage by Declaration Kind:")
        kind_stats = df.groupby('kind').agg({
            'is_used': 'mean',
            'out_deg': 'mean',
            'id': 'count'
        }).round(3)
        kind_stats.columns = ['prob_used', 'mean_usage', 'count']
        print(kind_stats.sort_values('prob_used', ascending=False).to_string())


def predict_new_theorem_usage(
    df: pd.DataFrame, 
    lr_model: LogisticRegression,
    scaler: StandardScaler,
    features: list,
    num_premises: int = 2,
    max_depth: int = 5
):
    """Predict if a new theorem would be used based on its features."""
    
    print("\n" + "="*60)
    print("PREDICTION FOR NEW THEOREM")
    print("="*60)
    
    # Create example theorem features
    new_theorem = pd.DataFrame({
        'num_explicit_premises': [num_premises],
        'num_implicit_args': [2],  # typical
        'num_typeclass_constraints': [1],
        'num_forall': [1],
        'num_exists': [0],
        'num_arrows': [num_premises],
        'max_nesting_depth': [max_depth],
        'namespace_depth': [3],  # typical
        'conclusion_arity': [1]
    })
    
    # Align with trained features
    for feat in features:
        if feat not in new_theorem.columns:
            new_theorem[feat] = 0
    
    X_new = new_theorem[features].values
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    prob_used = lr_model.predict_proba(X_new_scaled)[0, 1]
    
    print(f"Theorem with {num_premises} premises and depth {max_depth}:")
    print(f"  Probability of being used: {prob_used:.1%}")
    
    # Compare to similar theorems
    similar = df[(df['num_explicit_premises'] == num_premises) & 
                 (df['max_nesting_depth'].between(max_depth-1, max_depth+1))]
    
    if len(similar) > 0:
        print(f"  Similar theorems usage rate: {similar['is_used'].mean():.1%}")
        print(f"  Similar theorems mean out-degree: {similar['out_deg'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature-usage relationships")
    parser.add_argument("--structures", type=Path, required=True, help="Path to structures.parquet")
    parser.add_argument("--graph_metrics", type=Path, required=True, help="Path to graph_metrics.parquet")
    parser.add_argument("--output", type=Path, help="Optional: save analysis results to file")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading structures from {args.structures}")
    structures_df = pd.read_parquet(args.structures)
    
    logger.info(f"Loading graph metrics from {args.graph_metrics}")
    graph_df = pd.read_parquet(args.graph_metrics)
    
    # Run analyses
    df = analyze_usage_patterns(structures_df, graph_df)
    df = analyze_premises_effect(df)
    lr_model, scaler, features = analyze_feature_importance(df)
    find_usage_patterns(df)
    
    # Example predictions
    for premises in [0, 1, 2, 3, 5]:
        predict_new_theorem_usage(df, lr_model, scaler, features, num_premises=premises)
    
    # Save results if requested
    if args.output:
        logger.info(f"Saving analysis results to {args.output}")
        # Could save the enriched dataframe or specific statistics
        df.to_parquet(args.output)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()