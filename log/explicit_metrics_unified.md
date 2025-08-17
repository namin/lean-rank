# Theorem Importance Analysis - Explicit Metrics

*Generated: 2025-08-17 14:24*

## Unified Ranking Table

Sorted by Combined Score (most important first)

| Rank | Combined Score | Direct Usage | Transitive Deps | ∃ | Theorem | Statement |
|------|----------------|--------------|-----------------|---|---------|-----------|
| 1 | 1200.0 | 3 | 400 | - | `Nat.gcd_rec` | ∀ (m n : ℕ), m.gcd n = (n % m).gcd m |
| 2 | 930.0 | 2 | 465 | - | `Nat.mod_lt` | ∀ (x : ℕ) {y : ℕ}, y > 0 → x % y < y |
| 3 | 846.0 | 2 | 423 | - | `Nat.gcd_zero_right` | ∀ (n : ℕ), n.gcd 0 = n |
| 4 | 810.0 | 3 | 270 | - | `Nat.gcd_dvd` | ∀ (m n : ℕ), m.gcd n ∣ m ∧ m.gcd n ∣ n |
| 5 | 765.0 | 3 | 255 | - | `Nat.gcd_mul_left` | ∀ (m n k : ℕ), (m * n).gcd (m * k) = m * n.gcd k |
| 6 | 672.0 | 8 | 84 | - | `Nat.Coprime.gcd_mul_left_cancel` | ∀ {k n : ℕ} (m : ℕ), k.Coprime n → (k * m).gcd n = m.gcd n |
| 7 | 568.0 | 1 | 568 | - | `Nat.mod_eq` | ∀ (x y : ℕ), x % y = if 0 < y ∧ y ≤ x then (x - y) % y else x |
| 8 | 528.0 | 3 | 176 | - | `Nat.Prime.coprime_iff_not_dvd` | ∀ {p n : ℕ}, Nat.Prime p → (p.Coprime n ↔ ¬p ∣ n) |
| 9 | 506.0 | 1 | 506 | - | `Nat.mod_eq_of_lt` | ∀ {a b : ℕ}, a < b → a % b = a |
| 10 | 483.0 | 1 | 483 | - | `Nat.mod_eq_sub_mod` | ∀ {a b : ℕ}, a ≥ b → a % b = (a - b) % b |
| 11 | 472.0 | 2 | 236 | - | `Nat.Coprime.dvd_of_dvd_mul_right` | ∀ {k n m : ℕ}, k.Coprime n → k ∣ m * n → k ∣ m |
| 12 | 457.0 | 1 | 457 | - | `Nat.mod_zero` | ∀ (a : ℕ), a % 0 = a |
| 13 | 411.0 | 3 | 137 | - | `Nat.prime_of_mem_primeFactorsList` | ∀ {n p : ℕ}, p ∈ n.primeFactorsList → Nat.Prime p |
| 14 | 405.0 | 3 | 135 | - | `Prime.irreducible` | ∀ {M : Type u_1} [inst : CancelCommMonoidWithZero M] {p : M}, Prime p → Irreducible p |
| 15 | 393.0 | 1 | 393 | - | `Nat.gcd.induction` | ∀ {P : ℕ → ℕ → Prop} (m n : ℕ), (∀ (n : ℕ), P 0 n) → (∀ (m n : ℕ), 0 < m → P (n % m) m → P m n) → P m n |
| 16 | 332.0 | 2 | 166 | - | `Nat.gcd_comm` | ∀ (m n : ℕ), m.gcd n = n.gcd m |
| 17 | 316.0 | 2 | 158 | - | `Nat.div_lt_self` | ∀ {n k : ℕ}, 0 < n → 1 < k → n / k < n |
| 18 | 294.0 | 14 | 21 | - | `ArithmeticFunction.IsMultiplicative.mul` | ∀ {R : Type u_1} [inst : CommSemiring R] {f g : ArithmeticFunction R},   f.IsMultiplicative → g.IsMultiplicative → (f * g).IsMultiplicative |
| 19 | 292.0 | 2 | 146 | - | `Nat.factors_lemma` | ∀ {k : ℕ}, (k + 2) / (k + 2).minFac < k + 2 |
| 20 | 274.0 | 2 | 137 | - | `Nat.Prime.dvd_mul` | ∀ {p m n : ℕ}, Nat.Prime p → (p ∣ m * n ↔ p ∣ m ∨ p ∣ n) |
| 21 | 268.0 | 4 | 67 | - | `Nat.primeFactorsList_count_eq` | ∀ {n p : ℕ}, List.count p n.primeFactorsList = n.factorization p |
| 22 | 262.0 | 1 | 262 | - | `Nat.Prime.one_lt` | ∀ {p : ℕ}, Nat.Prime p → 1 < p |
| 23 | 258.0 | 1 | 258 | - | `Nat.gcd_dvd_left` | ∀ (m n : ℕ), m.gcd n ∣ m |
| 24 | 255.0 | 15 | 17 | - | `ArithmeticFunction.moebius_mul_coe_zeta` | ArithmeticFunction.moebius * ↑ArithmeticFunction.zeta = 1 |
| 25 | 241.0 | 1 | 241 | - | `Nat.gcd_dvd_right` | ∀ (m n : ℕ), m.gcd n ∣ n |
| 26 | 240.0 | 3 | 80 | - | `Nat.prime_iff` | ∀ {p : ℕ}, Nat.Prime p ↔ Prime p |
| 27 | 232.0 | 1 | 232 | - | `Nat.Coprime.dvd_of_dvd_mul_left` | ∀ {k m n : ℕ}, k.Coprime m → k ∣ m * n → k ∣ n |
| 28 | 214.0 | 2 | 107 | - | `Nat.gcd_dvd_gcd_of_dvd_left` | ∀ {m k : ℕ} (n : ℕ), m ∣ k → m.gcd n ∣ k.gcd n |
| 29 | 210.0 | 5 | 42 | - | `PrimeSpectrum.zeroLocus_vanishingIdeal_eq_closure` | ∀ {R : Type u} [inst : CommSemiring R] (t : Set (PrimeSpectrum R)),   PrimeSpectrum.zeroLocus ↑(PrimeSpectrum.vanishingIdeal t) = closure t |
| 30 | 204.0 | 4 | 51 | - | `Nat.primeFactorsList_prime` | ∀ {p : ℕ}, Nat.Prime p → p.primeFactorsList = [p] |
| 31 | 194.0 | 2 | 97 | - | `PrimeSpectrum.subset_zeroLocus_iff_le_vanishingIdeal` | ∀ {R : Type u} [inst : CommSemiring R] (t : Set (PrimeSpectrum R)) (I : Ideal R),   t ⊆ PrimeSpectrum.zeroLocus ↑I ↔ I ≤ PrimeSpectrum.vanishingIdeal t |
| 32 | 184.0 | 2 | 92 | - | `Nat.gcd_one_left` | ∀ (n : ℕ), Nat.gcd 1 n = 1 |
| 33 | 183.0 | 3 | 61 | - | `Nat.Prime.prime` | ∀ {p : ℕ}, Nat.Prime p → Prime p |
| 34 | 178.0 | 1 | 178 | - | `Nat.prime_dvd_prime_iff_eq` | ∀ {p q : ℕ}, Nat.Prime p → Nat.Prime q → (p ∣ q ↔ p = q) |
| 35 | 176.0 | 2 | 88 | - | `Nat.mod_add_div` | ∀ (m k : ℕ), m % k + k * (m / k) = m |
| 36 | 174.0 | 2 | 87 | - | `Nat.gcd_one_right` | ∀ (n : ℕ), n.gcd 1 = 1 |
| 37 | 172.0 | 2 | 86 | - | `Nat.gcd_assoc` | ∀ (m n k : ℕ), (m.gcd n).gcd k = m.gcd (n.gcd k) |
| 38 | 162.0 | 1 | 162 | - | `Nat.div_le_self` | ∀ (n k : ℕ), n / k ≤ n |
| 39 | 154.0 | 7 | 22 | - | `Nat.gcd_mul_gcd_of_coprime_of_mul_eq_mul` | ∀ {c d a b : ℕ}, c.Coprime d → a * b = c * d → a.gcd c * b.gcd c = c |
| 40 | 142.0 | 1 | 142 | - | `ZMod.charP` | ∀ (n : ℕ), CharP (ZMod n) n |
| 41 | 138.0 | 1 | 138 | - | `Ideal.IsPrime.mem_or_mem` | ∀ {α : Type u} [inst : Semiring α] {I : Ideal α}, I.IsPrime → ∀ {x y : α}, x * y ∈ I → x ∈ I ∨ y ∈ I |
| 42 | 138.0 | 2 | 69 | - | `PrimeSpectrum.gc_set` | ∀ (R : Type u) [inst : CommSemiring R],   GaloisConnection (fun s => PrimeSpectrum.zeroLocus s) fun t => ↑(PrimeSpectrum.vanishingIdeal t) |
| 43 | 138.0 | 3 | 46 | - | `Zsqrtd.norm_eq_mul_conj` | ∀ {d : ℤ} (n : ℤ√d), ↑n.norm = n * star n |
| 44 | 136.0 | 8 | 17 | - | `isPrimePow_iff_unique_prime_dvd` | ∀ {n : ℕ}, IsPrimePow n ↔ ∃! p, Nat.Prime p ∧ p ∣ n |
| 45 | 136.0 | 2 | 68 | - | `Nat.prime_def_minFac` | ∀ {p : ℕ}, Nat.Prime p ↔ 2 ≤ p ∧ p.minFac = p |
| 46 | 116.0 | 1 | 116 | - | `Nat.Coprime.symm` | ∀ {n m : ℕ}, n.Coprime m → m.Coprime n |
| 47 | 116.0 | 1 | 116 | - | `Ideal.IsPrime.ne_top` | ∀ {α : Type u} [inst : Semiring α] {I : Ideal α}, I.IsPrime → I ≠ ⊤ |
| 48 | 115.0 | 5 | 23 | - | `Nat.Coprime.gcd_mul` | ∀ {m n : ℕ} (k : ℕ), m.Coprime n → k.gcd (m * n) = k.gcd m * k.gcd n |
| 49 | 111.0 | 3 | 37 | - | `PrimeSpectrum.zeroLocus_empty_of_one_mem` | ∀ {R : Type u} [inst : CommSemiring R] {s : Set R}, 1 ∈ s → PrimeSpectrum.zeroLocus s = ∅ |
| 50 | 103.0 | 1 | 103 | - | `Nat.mod_one` | ∀ (x : ℕ), x % 1 = 0 |
| 51 | 101.0 | 1 | 101 | - | `PrimeSpectrum.mem_vanishingIdeal` | ∀ {R : Type u} [inst : CommSemiring R] (t : Set (PrimeSpectrum R)) (f : R),   f ∈ PrimeSpectrum.vanishingIdeal t ↔ ∀ x ∈ t, f ∈ x.asIdeal |
| 52 | 99.0 | 3 | 33 | - | `Nat.primeFactorsList_unique` | ∀ {n : ℕ} {l : List ℕ}, l.prod = n → (∀ p ∈ l, Nat.Prime p) → l.Perm n.primeFactorsList |
| 53 | 92.0 | 2 | 46 | - | `Ideal.IsPrime.comap` | ∀ {R : Type u} {S : Type v} {F : Type u_1} [inst : Semiring R] [inst_1 : Semiring S] [inst_2 : FunLike F R S] (f : F)   {K : Ideal S} [inst_3 : RingHomClass F R S] [hK : K.IsPrime], (Ideal.comap f K).IsPrime |
| 54 | 91.0 | 1 | 91 | - | `Nat.Prime.pos` | ∀ {p : ℕ}, Nat.Prime p → 0 < p |
| 55 | 91.0 | 1 | 91 | - | `PrimeSpectrum.gc` | ∀ (R : Type u) [inst : CommSemiring R],   GaloisConnection (fun I => PrimeSpectrum.zeroLocus ↑I) fun t => PrimeSpectrum.vanishingIdeal t |
| 56 | 87.0 | 3 | 29 | - | `IsLocalization.isPrime_iff_isPrime_disjoint` | ∀ {R : Type u_1} [inst : CommSemiring R] (M : Submonoid R) (S : Type u_2) [inst_1 : CommSemiring S]   [inst_2 : Algebra R S] [inst_3 : IsLocalization M S] (J : Ideal S),   J.IsPrime ↔ (Ideal.comap (algebraMap R S) J).IsPrime ∧ Disjoint ↑M ↑(Ideal.comap (algebraMap R S) J) |
| 57 | 85.0 | 1 | 85 | - | `Zsqrtd.ext_iff` | ∀ {d : ℤ} {x y : ℤ√d}, x = y ↔ x.re = y.re ∧ x.im = y.im |
| 58 | 85.0 | 1 | 85 | - | `Nat.gcd_dvd_gcd_mul_left` | ∀ (m n k : ℕ), m.gcd n ∣ (k * m).gcd n |
| 59 | 81.0 | 1 | 81 | - | `Nat.Prime.ne_one` | ∀ {p : ℕ}, Nat.Prime p → p ≠ 1 |
| 60 | 80.0 | 2 | 40 | - | `PrimeSpectrum.basicOpen_eq_zeroLocus_compl` | ∀ {R : Type u} [inst : CommSemiring R] (r : R), ↑(PrimeSpectrum.basicOpen r) = (PrimeSpectrum.zeroLocus {r})ᶜ |
| 61 | 80.0 | 4 | 20 | - | `Nat.gcd_self` | ∀ (n : ℕ), n.gcd n = n |
| 62 | 80.0 | 5 | 16 | - | `ZMod.val_coe_unit_coprime` | ∀ {n : ℕ} (u : (ZMod n)ˣ), (↑u).val.Coprime n |
| 63 | 78.0 | 3 | 26 | - | `ArithmeticFunction.cardFactors_eq_one_iff_prime` | ∀ {n : ℕ}, ArithmeticFunction.cardFactors n = 1 ↔ Nat.Prime n |
| 64 | 78.0 | 6 | 13 | - | `PrimeSpectrum.isClosed_singleton_iff_isMaximal` | ∀ {R : Type u} [inst : CommSemiring R] (x : PrimeSpectrum R), IsClosed {x} ↔ x.asIdeal.IsMaximal |
| 65 | 76.0 | 4 | 19 | - | `ArithmeticFunction.isMultiplicative_moebius` | ArithmeticFunction.moebius.IsMultiplicative |
| 66 | 75.0 | 1 | 75 | - | `Ideal.IsPrime.mul_mem_iff_mem_or_mem` | ∀ {α : Type u} [inst : CommSemiring α] {I : Ideal α}, I.IsPrime → ∀ {x y : α}, x * y ∈ I ↔ x ∈ I ∨ y ∈ I |
| 67 | 74.0 | 2 | 37 | - | `Prime.dvd_of_dvd_pow` | ∀ {M : Type u_1} [inst : CommMonoidWithZero M] {p : M}, Prime p → ∀ {a : M} {n : ℕ}, p ∣ a ^ n → p ∣ a |
| 68 | 72.0 | 2 | 36 | - | `Zsqrtd.norm_eq_one_iff_mem_unitary` | ∀ {d : ℤ} {a : ℤ√d}, a.norm = 1 ↔ a ∈ unitary (ℤ√d) |
| 69 | 72.0 | 2 | 36 | - | `IsLocalization.AtPrime.isLocalRing` | ∀ {R : Type u_1} [inst : CommSemiring R] (S : Type u_2) [inst_1 : CommSemiring S] [inst_2 : Algebra R S] (P : Ideal R)   [hp : P.IsPrime] [inst : IsLocalization.AtPrime S P], IsLocalRing S |
| 70 | 72.0 | 3 | 24 | - | `Nat.factors_eq` | ∀ (n : ℕ), UniqueFactorizationMonoid.normalizedFactors n = ↑n.primeFactorsList |
| 71 | 70.0 | 2 | 35 | - | `Nat.Coprime.coprime_dvd_right` | ∀ {n m k : ℕ}, n ∣ m → k.Coprime m → k.Coprime n |
| 72 | 70.0 | 1 | 70 | - | `Nat.Coprime.mul` | ∀ {m k n : ℕ}, m.Coprime k → n.Coprime k → (m * n).Coprime k |
| 73 | 69.0 | 3 | 23 | - | `ZMod.eq_iff_modEq_nat` | ∀ (n : ℕ) {a b : ℕ}, ↑a = ↑b ↔ a ≡ b [MOD n] |
| 74 | 69.0 | 3 | 23 | - | `ArithmeticFunction.coe_mul_zeta_apply` | ∀ {R : Type u_1} [inst : Semiring R] {f : ArithmeticFunction R} {x : ℕ},   (f * ↑ArithmeticFunction.zeta) x = ∑ i ∈ x.divisors, f i |
| 75 | 68.0 | 4 | 17 | - | `Nat.div_eq_zero_iff` | ∀ {a b : ℕ}, a / b = 0 ↔ b = 0 ∨ a < b |
| 76 | 68.0 | 2 | 34 | - | `Ideal.IsPrime.mem_of_pow_mem` | ∀ {α : Type u} [inst : Semiring α] {I : Ideal α}, I.IsPrime → ∀ {r : α} (n : ℕ), r ^ n ∈ I → r ∈ I |
| 77 | 68.0 | 2 | 34 | - | `Pell.is_pell_solution_iff_mem_unitary` | ∀ {d : ℤ} {a : ℤ√d}, a.re ^ 2 - d * a.im ^ 2 = 1 ↔ a ∈ unitary (ℤ√d) |
| 78 | 66.0 | 2 | 33 | - | `Nat.Prime.dvd_of_dvd_pow` | ∀ {p m n : ℕ}, Nat.Prime p → p ∣ m ^ n → p ∣ m |
| 79 | 65.0 | 5 | 13 | - | `ArithmeticFunction.vonMangoldt_sum` | ∀ {n : ℕ}, ∑ i ∈ n.divisors, ArithmeticFunction.vonMangoldt i = Real.log ↑n |
| 80 | 65.0 | 5 | 13 | - | `PrimeSpectrum.localization_specComap_range` | ∀ {R : Type u} (S : Type v) [inst : CommSemiring R] [inst_1 : CommSemiring S] [inst_2 : Algebra R S] (M : Submonoid R)   [inst_3 : IsLocalization M S], Set.range (algebraMap R S).specComap = {p \| Disjoint ↑M ↑p.asIdeal} |
| 81 | 64.0 | 4 | 16 | - | `Zsqrtd.divides_sq_eq_zero` | ∀ {d : ℕ} [dnsq : Zsqrtd.Nonsquare d] {x y : ℕ}, x * x = d * y * y → x = 0 ∧ y = 0 |
| 82 | 63.0 | 3 | 21 | - | `Zsqrtd.smul_val` | ∀ {d : ℤ} (n x y : ℤ), ↑n * { re := x, im := y } = { re := n * x, im := n * y } |
| 83 | 63.0 | 3 | 21 | - | `Pell.isPell_pellZd` | ∀ {a : ℕ} (a1 : 1 < a) (n : ℕ), Pell.IsPell (Pell.pellZd a1 n) |
| 84 | 62.0 | 2 | 31 | - | `Nat.Coprime.mul_dvd_of_dvd_of_dvd` | ∀ {m n a : ℕ}, m.Coprime n → m ∣ a → n ∣ a → m * n ∣ a |
| 85 | 62.0 | 1 | 62 | - | `ZMod.intCast_eq_intCast_iff` | ∀ (a b : ℤ) (c : ℕ), ↑a = ↑b ↔ a ≡ b [ZMOD ↑c] |
| 86 | 62.0 | 2 | 31 | - | `PrimeSpectrum.zeroLocus_empty_iff_eq_top` | ∀ {R : Type u} [inst : CommSemiring R] {I : Ideal R}, PrimeSpectrum.zeroLocus ↑I = ∅ ↔ I = ⊤ |
| 87 | 62.0 | 2 | 31 | - | `PrimeSpectrum.vanishingIdeal_zeroLocus_eq_radical` | ∀ {R : Type u} [inst : CommSemiring R] (I : Ideal R),   PrimeSpectrum.vanishingIdeal (PrimeSpectrum.zeroLocus ↑I) = I.radical |
| 88 | 60.0 | 6 | 10 | - | `PrimeSpectrum.isIrreducible_zeroLocus_iff_of_radical` | ∀ {R : Type u} [inst : CommSemiring R] (I : Ideal R),   I.IsRadical → (IsIrreducible (PrimeSpectrum.zeroLocus ↑I) ↔ I.IsPrime) |
| 89 | 59.0 | 1 | 59 | - | `isCoprime_comm` | ∀ {R : Type u} [inst : CommSemiring R] {x y : R}, IsCoprime x y ↔ IsCoprime y x |
| 90 | 57.0 | 3 | 19 | - | `Nat.factorization_pow` | ∀ (n k : ℕ), (n ^ k).factorization = k • n.factorization |
| 91 | 57.0 | 3 | 19 | - | `ArithmeticFunction.moebius_apply_prime_pow` | ∀ {p k : ℕ}, Nat.Prime p → k ≠ 0 → ArithmeticFunction.moebius (p ^ k) = if k = 1 then -1 else 0 |
| 92 | 56.0 | 1 | 56 | - | `ZMod.val_natCast` | ∀ {n : ℕ} (a : ℕ), (↑a).val = a % n |
| 93 | 56.0 | 4 | 14 | - | `ArithmeticFunction.vonMangoldt_apply_prime` | ∀ {p : ℕ}, Nat.Prime p → ArithmeticFunction.vonMangoldt p = Real.log ↑p |
| 94 | 56.0 | 4 | 14 | - | `PrimeSpectrum.isTopologicalBasis_basic_opens` | ∀ {R : Type u} [inst : CommSemiring R],   TopologicalSpace.IsTopologicalBasis (Set.range fun r => ↑(PrimeSpectrum.basicOpen r)) |
| 95 | 56.0 | 2 | 28 | - | `IsLocalization.isPrime_of_isPrime_disjoint` | ∀ {R : Type u_1} [inst : CommSemiring R] (M : Submonoid R) (S : Type u_2) [inst_1 : CommSemiring S]   [inst_2 : Algebra R S] [inst_3 : IsLocalization M S] (I : Ideal R),   I.IsPrime → Disjoint ↑M ↑I → (Ideal.map (algebraMap R S) I).IsPrime |
| 96 | 56.0 | 7 | 8 | - | `ArithmeticFunction.sum_eq_iff_sum_smul_moebius_eq` | ∀ {R : Type u_1} [inst : AddCommGroup R] {f g : ℕ → R},   (∀ n > 0, ∑ i ∈ n.divisors, f i = g n) ↔     ∀ n > 0, ∑ x ∈ n.divisorsAntidiagonal, ArithmeticFunction.moebius x.1 • g x.2 = f n |
| 97 | 55.0 | 5 | 11 | - | `PrimeSpectrum.localization_away_comap_range` | ∀ {R : Type u} [inst : CommSemiring R] (S : Type v) [inst_1 : CommSemiring S] [inst_2 : Algebra R S] (r : R)   [inst_3 : IsLocalization.Away r S], Set.range ⇑(PrimeSpectrum.comap (algebraMap R S)) = ↑(PrimeSpectrum.basicOpen r) |
| 98 | 54.0 | 2 | 27 | - | `Nat.Coprime.gcd_left` | ∀ {m n : ℕ} (k : ℕ), m.Coprime n → (k.gcd m).Coprime n |
| 99 | 54.0 | 3 | 18 | - | `EulerProduct.summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum` | ∀ {R : Type u_1} [inst : NormedCommRing R] {f : ℕ → R} [inst_1 : CompleteSpace R],   f 1 = 1 →     (∀ {m n : ℕ}, m.Coprime n → f (m * n) = f m * f n) →       (∀ {p : ℕ}, Nat.Prime p → Summable fun n => ‖f (p ^ n)‖) →         ∀ (s : Finset ℕ),           (Summable fun m => ‖f ↑m‖) ∧             HasSum (fun m => f ↑m) (∏ p ∈ Finset.filter (fun p => Nat.Prime p) s, ∑' (n : ℕ), f (p ^ n)) |
| 100 | 52.0 | 4 | 13 | - | `Nat.Prime.emultiplicity_factorial` | ∀ {p : ℕ}, Nat.Prime p → ∀ {n b : ℕ}, Nat.log p n < b → emultiplicity p n.factorial = ↑(∑ i ∈ Finset.Ico 1 b, n / p ^ i) |

### Metric Explanations

- **Combined Score**: `(direct × transitive) / (∃ + 1)` - Overall importance metric
- **Direct Usage**: Number of theorems that directly call this theorem
- **Transitive Deps**: Number of theorems that depend on this (full closure)
- **∃**: Count of existential quantifiers in the statement

### Summary Statistics

- Total theorems analyzed: 2121
- Theorems with zero direct usage: 312 (14.7%)
- Average direct usage: 1.64
- Average transitive dependencies: 11.44

---
*All metrics are explainable and can be manually verified.*