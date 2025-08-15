Testing 10 theorems in domain: number_theory
======================================================================

[1/10] Fermat's Little Theorem
    Type: ∀ p : ℕ, Prime p → ∀ a : ℕ, ¬(p ∣ a) → a ^ (p - 1) ≡ 1 [MOD ...
    Testing adoption rates... ✓ (0.66% @ k=10)
    Testing semantic productivity... ✓ (pred=1.9)

[2/10] Divisibility Transitivity
    Type: ∀ a b c : ℕ, a ∣ b → b ∣ c → a ∣ c...
    Testing adoption rates... ✓ (0.06% @ k=10)
    Testing semantic productivity... ✓ (pred=1.8)

[3/10] Infinitude of Primes
    Type: ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ p > n...
    Testing adoption rates... ✓ (0.11% @ k=10)
    Testing semantic productivity... ✓ (pred=1.7)

[4/10] Prime Oddness
    Type: ∀ p : ℕ, Prime p → p = 2 ∨ Odd p...
    Testing adoption rates... ✓ (0.11% @ k=10)
    Testing semantic productivity... ✓ (pred=1.0)

[5/10] Coprimality Divisibility
    Type: ∀ a b : ℕ, Coprime a b → ∀ c : ℕ, a ∣ c → b ∣ c → a * b ∣ c...
    Testing adoption rates... ✓ (0.00% @ k=10)
    Testing semantic productivity... ✓ (pred=2.0)

[6/10] Euclid's Lemma
    Type: ∀ p : ℕ, Prime p → ∀ a b : ℕ, p ∣ a * b → p ∣ a ∨ p ∣ b...
    Testing adoption rates... ✓ (0.39% @ k=10)
    Testing semantic productivity... ✓ (pred=2.1)

[7/10] GCD Commutativity
    Type: ∀ a b : ℕ, gcd a b = gcd b a...
    Testing adoption rates... ✓ (0.06% @ k=10)
    Testing semantic productivity... ✓ (pred=2.5)

[8/10] Prime Factorial
    Type: ∀ p : ℕ, Prime p → ∀ n : ℕ, n < p → ¬(p ∣ n!)...
    Testing adoption rates... ✓ (0.28% @ k=10)
    Testing semantic productivity... ✓ (pred=2.3)

[9/10] Wilson's Theorem
    Type: ∀ p : ℕ, Prime p ↔ (p - 1)! ≡ -1 [MOD p]...
    Testing adoption rates... ✓ (0.06% @ k=10)
    Testing semantic productivity... ✓ (pred=0.9)

[10/10] Bezout's Identity
    Type: ∀ a b : ℕ, ∃ x y : ℤ, a * x + b * y = gcd a b...
    Testing adoption rates... ✓ (0.06% @ k=10)
    Testing semantic productivity... ✓ (pred=2.7)

======================================================================

## COMPREHENSIVE PRODUCTIVITY ANALYSIS

| Theorem | Adoption@10 | Adoption@50 | Semantic Pred | Similarity | Category |
|---------|-------------|-------------|---------------|------------|----------|
| Bezout's Identity | 0.06% (L=0.12) | 0.44% (~8) | 2.7 | 0.995 | ◐ Moderate |
| GCD Commutativity | 0.06% (L=0.12) | 0.55% (~10) | 2.5 | 0.997 | ◐ Moderate |
| Prime Factorial | 0.28% (L=0.59) | 1.27% (~23) | 2.3 | 0.996 | ✓ Useful |
| Euclid's Lemma | 0.39% (L=0.82) | 1.44% (~26) | 2.1 | 0.997 | ✓ Useful |
| Coprimality Divisibility | 0% | 0.77% (~14) | 2.0 | 0.960 | ◐ Moderate |
| Fermat's Little Theorem | 0.66% (L=1.41) | 1.16% (~21) | 1.9 | 0.996 | ✓ Useful |
| Divisibility Transitivity | 0.06% (L=0.12) | 0.22% (~4) | 1.8 | 0.979 | ◐ Moderate |
| Infinitude of Primes | 0.11% (L=0.23) | 0.72% (~13) | 1.7 | 0.921 | ◐ Moderate |
| Prime Oddness | 0.11% (L=0.23) | 0.22% (~4) | 1.0 | 0.924 | ○ Low |
| Wilson's Theorem | 0.06% (L=0.12) | 0.33% (~6) | 0.9 | 0.951 | ○ Low |

## Key Insights

**Best by adoption**: Fermat's Little Theorem (0.66% @ k=10)
**Best by semantic prediction**: Bezout's Identity (predicted 2.7 downstream uses)
**High impact theorems**: 0/10

### Metric Explanations:
- **Adoption@k**: % of existing theorems that would use this (higher is better)
- **L=lift**: Performance vs random (>1.0 means better than random)
- **Semantic Pred**: Predicted number of downstream theorems based on similarity
- **Similarity**: How similar to existing theorems (1.0 = identical)
