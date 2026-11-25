# rh-falsifier

Standalone RH/GRH falsifier from pure sieve arithmetic — single file,
numpy only.

On a prime chain {p^k}, the smoothed von Mangoldt fluctuation residue

    c_k = M(e^{u/2}+e^{-u/2}) - P(u) - Arch(u),   u = k ln p

is a bounded sum of real cosines 2 Σ_γ w(γ) cos(γ k ln p) iff RH holds
(von Koch equivalence). An off-line zero β+iγ injects exponential growth
p^{k(β-1/2)}, which the Padé approximant of C_p(z) = Σ c_k z^k detects
as a pole inside the "traitor annulus" p^{-1/2} < |z| < 1:

    real zero γ        -> pole at z = e^{±iγ ln p}   (|z| = 1)
    archimedean decay  -> pole at |z| = p^{-1/2}
    TRAITOR β+iγ       -> pole at |z| = p^{-(β-1/2)} (annulus)

Two-level logic: Padé localizes candidates, `--cert` certifies them
(rigorous per-term error bounds + certified exponential-growth test).
An interior pole without certified growth is a short-series artifact.

## Usage

    python3 rh_falsifier.py chain                    # {2^k}, Pade [8/8]
    python3 rh_falsifier.py chain --cert             # + rigorous bounds
    python3 rh_falsifier.py chain -p 3 -K 13 -d 4    # other prime chains
    python3 rh_falsifier.py inject --beta 0.75 --gamma 25   # calibration
    python3 rh_falsifier.py matrix -N 12             # closed-form K_N (demo)
    python3 rh_falsifier.py chain --chi 4            # L(s,chi_4), experimental

## Verified behaviour

- ζ chains {2^k},{3^k},{5^k}: |c_k| bounded, traitor annulus EMPTY;
  on-line poles at γ̂ ≈ 13.1 / 21.0 / 30.1 (true: 14.13 / 21.02 / 30.42)
- injected traitor β=0.75, γ=25 → detected pole → β̂=0.749,
  CERTIFIED GROWTH at 3σ
- sensitivity: detection band β ∈ (≈0.53, ≈0.97), amplitude floor ~1e-4
  at 21 terms
- certified error bound on every c_k: ε ≈ 2×10⁻⁷

## Honest scope

One-sided test: a certified interior pole would falsify RH/GRH; an empty
annulus proves nothing. Detection requires a candidate whose contribution
is not drowned by truncation — scale with more terms for weaker signals.
Raw Dirac atoms are not admissible test functions (r=1 pole); the
admissible object is the smoothed chain residue.
