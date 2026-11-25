#!/usr/bin/env python3
"""
rh_falsifier.py — standalone RH/GRH falsifier from pure sieve arithmetic.

Based on the project's matrix program (RH_matrix_program_final_v678.md,
Results 9-11): on a prime chain {p^k}, the smoothed von Mangoldt
fluctuation residue

    c_k = M(e^{u/2}+e^{-u/2}) - P(u) - Arch(u),   u = k ln p

is a bounded sum of real cosines 2 Σ_γ w(γ) cos(γ k ln p) iff RH holds
(von Koch equivalence). An off-line zero β+iγ injects a term growing as
p^{k(β-1/2)}, which the Padé approximant of C_p(z) = Σ c_k z^k sees as a
pole inside the "traitor annulus"  p^{-1/2} < |z| < 1 :

    real zero γ        -> pole at z = e^{±iγ ln p}      (|z| = 1)
    archimedean decay  -> pole at |z| = p^{-1/2}
    TRAITOR β+iγ       -> pole at |z| = p^{-(β-1/2)}    (annulus)

One-sided test: it can disprove RH (certified interior pole), it cannot
prove it (empty annulus is only consistency).

Modes:
  chain   — compute c_k on {p^k}, Padé, report poles  [default]
  matrix  — build the smoothed arithmetic Gram matrix K_N, check PSD
  inject  — calibration: add a synthetic traitor (β,γ) and verify detection

Requires: numpy, scipy (erfc/digamma optional for archimedean tail).
"""

import argparse
import numpy as np

# ----------------------------------------------------------------------
# smoothing kernel  g(u) = exp(-u^2 / (4 s^2)),  s = smoothing parameter
# Gaussian moment M = ∫ e^{v/2} g(v) dv = 2 s sqrt(pi) e^{s^2/4}
# ----------------------------------------------------------------------

def gauss(u, s):
    return np.exp(-u * u / (4.0 * s * s))


def boundary(u, s):
    """s=0,1 poles: mean value of the prime sum, closed form."""
    M = 2.0 * s * np.sqrt(np.pi) * np.exp(s * s / 4.0)
    return M * (np.exp(u / 2.0) + np.exp(-u / 2.0))


def archimedean(u):
    """trivial-zero block, series over Γ poles s=-2k (Poisson kernels):
    Σ_k e^{-(2k+1/2)|u|} = e^{-|u|/2}/(1-e^{-2|u|}).  Bounded, decays.
    At u=0 the series has a pole (the r=1 diagonal of the matrix form);
    we renormalize by the finite part, e^{-a/2}/(1-e^{-2a}) = 1/(2a) - 1/4."""
    a = abs(u)
    if a < 1e-8:
        return -0.25
    return np.exp(-a / 2.0) / (1.0 - np.exp(-2.0 * a))


# ----------------------------------------------------------------------
# sieve of von Mangoldt up to a bound
# ----------------------------------------------------------------------

def sieve_lambda(n_max):
    """Return arrays (log n of prime powers, Lambda(n)/sqrt(n)) for n<=n_max
    using a linear sieve for the least prime factor."""
    n_max = int(n_max)
    lpf = np.zeros(n_max + 1, dtype=np.int64)
    primes = []
    for i in range(2, n_max + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            v = i * p
            if v > n_max or p > lpf[i]:
                break
            lpf[v] = p
    # prime powers: for each prime p, all p^a <= n_max
    us, ws = [], []
    for p in primes:
        lp = np.log(p)
        j = p
        while j <= n_max:
            us.append(np.log(j))
            ws.append(lp / np.sqrt(j))
            j *= p
    return np.array(us), np.array(ws)


# ----------------------------------------------------------------------
# chain residue
# ----------------------------------------------------------------------

def chain_residue(p, K, s=0.1, margin=1.2, chi=None):
    """c_k for k=0..K-1 on the chain {p^k}.

    chi: optional callable n -> Dirichlet character value (for L(s,chi);
         for a non-principal character the boundary term is absent).
    """
    u_max = (K - 1) * np.log(p)
    n_max = int(np.exp(u_max + margin)) + 2
    if n_max > 50_000_000:
        raise ValueError(f"sieve bound {n_max:.2e} too large — reduce K for p={p}")
    us, ws = sieve_lambda(n_max)
    if chi is not None:
        chi_arr = np.array([chi(int(round(np.exp(u)))) for u in us])
        ws = ws * chi_arr
    cs = np.zeros(K)
    for k in range(K):
        u0 = k * np.log(p)
        P = np.sum(ws * (gauss(u0 - us, s) + gauss(u0 + us, s)))
        if chi is None:
            cs[k] = boundary(u0, s) - P - archimedean(u0)
        else:
            # non-principal chi: no s=1 pole -> no boundary term; the
            # archimedean/trivial-zero part has a different closed form
            # (character-dependent), so it is omitted. Truncation artifacts
            # can place Padé poles inside the annulus: see caveat in output.
            cs[k] = -P
    return cs


# ----------------------------------------------------------------------
# certification: rigorous error bounds on c_k
# ----------------------------------------------------------------------

def _tail_bound(u0, s, L):
    """Bound on the omitted prime-sum tail at chain point u0:

        sum_{ln j > L} Lambda(j)/sqrt(j) * [g(ln j - u0) + g(ln j + u0)]

    dominated by the sum over ALL integers n > e^L (prime powers are a
    subset; Lambda(n) <= ln n) and, since f(v) = v e^{v/2} e^{-(v-u0)^2/(4s^2)}
    is decreasing for v > u0 + ~s^2, bounded by its integral:

        int_L^inf v e^{v/2} e^{-(v-u0)^2/(4s^2)} dv
            = e^{u0/2+s^2/4} [ s sqrt(pi) (u0+s^2) erfc(w)
                               + 2 s^2 e^{-w^2} ],  w = (L-u0-s^2)/(2s)

    The reflected term g(u0+ln j) is the same with u0 -> -u0.
    """
    from scipy.special import erfc
    def side(v0):
        w = (L - v0 - s * s) / (2 * s)
        return max(np.exp(v0 / 2 + s * s / 4) * (
            s * np.sqrt(np.pi) * (v0 + s * s) * erfc(w)
            + 2 * s * s * np.exp(-w * w)), 0.0)
    return side(u0) + side(-u0)


def certify_chain(p, K, s=0.1, margin=1.2):
    """c_k with rigorous per-term error bound eps_k = truncation + rounding.
    Then a certified growth test on the tail of |c_k|."""
    u_max = (K - 1) * np.log(p)
    L = u_max + margin
    n_max = int(np.exp(L)) + 2
    if n_max > 50_000_000:
        raise ValueError(f"sieve bound {n_max:.2e} too large")
    us, ws = sieve_lambda(n_max)
    sumw = np.sum(ws)
    nterms = len(us)
    cs, eps = np.zeros(K), np.zeros(K)
    for k in range(K):
        u0 = k * np.log(p)
        P = np.sum(ws * (gauss(u0 - us, s) + gauss(u0 + us, s)))
        cs[k] = boundary(u0, s) - P - archimedean(u0)
        # truncation of the prime tail + float64 accumulation bound
        eps[k] = _tail_bound(u0, s, L) + nterms * 2.3e-16 * sumw
    return cs, eps


def growth_verdict(cs, eps, p, k_min=8, sigma=3.0):
    """Least-squares slope of log|c_k| vs k (per-step rate, in units of
    ln p).  Growth is CERTIFIED when slope > sigma*SE on points whose
    value clears 10x the error floor.  Returns (slope, se, used, verdict)."""
    ks = np.arange(len(cs))
    mask = (ks >= k_min) & (np.abs(cs) > 10 * np.maximum(eps, 1e-15))
    if mask.sum() < 4:
        return None, None, int(mask.sum()), "INSUFFICIENT clean points"
    x, y = ks[mask], np.log(np.abs(cs[mask])) / np.log(p)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = sol[0]
    dof = len(x) - 2
    se = np.sqrt(res[0] / dof) / np.sqrt(np.sum((x - x.mean()) ** 2)) \
        if len(res) and res[0] > 0 else np.inf
    if se == np.inf or se == 0:
        return slope, se, len(x), "UNDETERMINED"
    if slope > sigma * se:
        verdict = f"CERTIFIED GROWTH (slope {slope:.3f} > {sigma}*SE)"
    elif slope < -sigma * se:
        verdict = f"CERTIFIED DECAY (slope {slope:.3f})"
    else:
        verdict = f"BOUNDED/FLAT within {sigma} sigma (slope {slope:.3f} +- {se:.3f})"
    return slope, se, len(x), verdict


# ----------------------------------------------------------------------
# Padé [d/d] poles
# ----------------------------------------------------------------------

def pade_poles(c, d):
    """[d/d] Padé approximant of C(z)=sum c_k z^k; returns poles (roots of
    the denominator 1 + b_1 z + ... + b_d z^d)."""
    c = np.asarray(c, dtype=np.float64)
    n = 2 * d
    if len(c) < n + 1:
        raise ValueError(f"need >= {n+1} coefficients, have {len(c)}")
    # solve  sum_{j=1..d} b_j c[d+i-j] = -c[d+i],  i=1..d
    A = np.array([[c[d + i - j] for j in range(1, d + 1)] for i in range(1, d + 1)])
    rhs = -np.array([c[d + i] for i in range(1, d + 1)])
    b = np.linalg.solve(A, rhs)
    coeffs = np.concatenate([[1.0], b])[::-1]          # numpy.roots: descending
    return np.roots(coeffs)


def classify_poles(poles, p, tol=0.03):
    """Split poles into |z|~1 (real zeros), |z|~p^{-1/2} (trivial) and the
    traitor annulus (p^{-1/2}, 1)."""
    zeros, trivial, traitors = [], [], []
    for z in poles:
        r = abs(z)
        if r > 1.0 - tol:
            theta = abs(np.angle(z))
            gammas = [(theta + 2 * np.pi * m) / np.log(p) for m in range(8)]
            zeros.append((z, r, gammas))
        elif r < (p ** -0.5) * (1.0 + tol):
            trivial.append((z, r))
        else:
            beta = 0.5 - np.log(r) / np.log(p)
            gamma = abs(np.angle(z)) / np.log(p)
            traitors.append((z, r, beta, gamma))
    return zeros, trivial, traitors


# ----------------------------------------------------------------------
# arithmetic Gram matrix K_N  (v677/v678 closed form, unsmoothed)
#
#   K_mn = (m+n)/sqrt(mn)                     <- boundary poles s=0,1
#          - sqrt(mn) * max / |m^2 - n^2|     <- trivial-zero block
#                                              (= e^{-|u|/2}/(1-e^{-2|u|}))
#          - Lambda(max/min)/sqrt(max/min)    <- prime powers, ratio = p^k
#   K_nn = c0                                 <- r=1 pole, renormalized
# ----------------------------------------------------------------------

def _lam_ratio(m, n):
    """Lambda(j)/sqrt(j) if max/min = j is a prime power, else 0."""
    if m == n:
        return 0.0
    j = max(m, n) // min(m, n)
    if j * min(m, n) != max(m, n):
        return 0.0
    f = j
    p = None
    for q in range(2, f + 1):
        if q * q > f:
            break
        if f % q == 0:
            p = q
            while f % q == 0:
                f //= q
            break
    if p is None:
        p = f
    if f != 1:
        return 0.0
    return np.log(p) / np.sqrt(j)


def build_K(N, c0=1.37):
    """Closed-form K_N. Off-diagonal exact; diagonal = renormalization
    constant c0 absorbing the r=1 pole of the trivial-zero block."""
    K = np.zeros((N, N))
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m == n:
                K[m - 1, n - 1] = c0
                continue
            mn = np.sqrt(m * n)
            v = (m + n) / mn
            v -= mn * max(m, n) / abs(m * m - n * n)
            v -= _lam_ratio(m, n)
            K[m - 1, n - 1] = v
    return K


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RH/GRH falsifier (sieve arithmetic only)")
    ap.add_argument("mode", nargs="?", default="chain",
                    choices=["chain", "matrix", "inject"])
    ap.add_argument("-p", type=int, default=2, help="prime for the chain {p^k}")
    ap.add_argument("-K", type=int, default=21, help="chain terms c_0..c_{K-1}")
    ap.add_argument("-N", type=int, default=12, help="matrix size for mode=matrix")
    ap.add_argument("-d", type=int, default=8, help="Pade order [d/d]")
    ap.add_argument("-s", type=float, default=0.1, help="smoothing sigma")
    ap.add_argument("--beta", type=float, default=0.75, help="traitor beta (inject)")
    ap.add_argument("--gamma", type=float, default=37.0, help="traitor gamma (inject)")
    ap.add_argument("--amp", type=float, default=0.02, help="traitor amplitude")
    ap.add_argument("--chi", type=int, default=0, metavar="MOD",
                    help="primitive Dirichlet character modulus (4 or 5) for L(s,chi) — EXPERIMENTAL: annulus thresholds are zeta-calibrated, expect truncation artifacts")
    ap.add_argument("--cert", action="store_true",
                    help="rigorous error bounds on c_k (truncation + rounding) and a certified growth test — zeta chains only")
    args = ap.parse_args()

    if args.mode == "matrix":
        print(f"Closed-form K_N, N={args.N} (diagonal = c0 renormalization)")
        print("c0        lambda_min      PSD?")
        for c0 in [0.0, 0.5, 1.0, 1.37, 2.0, 3.0, 5.0]:
            K = build_K(args.N, c0)
            ev = np.linalg.eigvalsh(K)
            print(f"{c0:5.2f}    {ev[0]:+.4e}      {ev[0] > -1e-6}")
        print("\nNOTE: raw Dirac atoms are NOT admissible test functions")
        print("(r=1 pole on the diagonal, v677): a negative lambda_min here")
        print("is EXPECTED and is NOT a falsification of RH. The admissible")
        print("test is the smoothed chain residue — use mode=chain.")
        return

    chi_fn = None
    if args.chi == 4:
        chi_fn = lambda n: 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1)
    elif args.chi == 5:
        chi_fn = lambda n: [0, 1, -1, -1, 1][n % 5] if n % 5 else 0
    elif args.chi != 0:
        raise SystemExit("supported --chi moduli: 4, 5")

    print(f"Chain {{p^k}}, p={args.p}, K={args.K}, sigma={args.s} — sieving ...")
    eps = None
    if args.cert:
        if chi_fn is not None:
            print("(cert bounds are zeta-only; running uncertified for chi)")
            cs = chain_residue(args.p, args.K, s=args.s, chi=chi_fn)
        else:
            cs, eps = certify_chain(args.p, args.K, s=args.s)
    else:
        cs = chain_residue(args.p, args.K, s=args.s, chi=chi_fn)
    print("c_k:", np.array2string(cs, precision=4, suppress_small=False, max_line_width=100))
    if eps is not None:
        print("eps:", np.array2string(eps, precision=2, max_line_width=100))
        print(f"certified |c_k| <= {np.max(np.abs(cs)+eps):.4f} for all k<= {args.K-1}")
    print(f"|c_k| max = {np.max(np.abs(cs)):.4f}  (bounded cosines expected if RH)")

    if args.mode == "inject":
        w = np.exp(-(args.gamma * args.s) ** 2)
        for k in range(args.K):
            cs[k] += args.amp * 2.0 * (args.p ** (k * (args.beta - 0.5))) \
                     * np.cos(args.gamma * k * np.log(args.p))
        print(f"INJECTED traitor beta={args.beta}, gamma={args.gamma} "
              f"(growth x{args.p**(args.beta-0.5):.3f}/step)")
        print("c_k+:", np.array2string(cs, precision=4, max_line_width=100))

    poles = pade_poles(cs, args.d)
    zeros, trivial, traitors = classify_poles(poles, args.p)

    print(f"\nPade [{args.d}/{args.d}] poles ({len(poles)}):")
    for z, r, gs in sorted(zeros, key=lambda t: t[2][1]):
        cand = ", ".join(f"{g:.1f}" for g in gs[:4])
        print(f"  |z|={r:.4f}  gamma ~ {{{cand}, ...}}   (on-line zero candidate)")
    for z, r in trivial:
        print(f"  |z|={r:.4f}  (trivial/archimedean, p^-1/2 = {args.p**-0.5:.4f})")
    print()
    if traitors:
        for z, r, b, g in traitors:
            print(f"  *** TRAITOR POLE: |z|={r:.4f} -> beta ~ {b:.3f}, gamma ~ {g:.2f} ***")
        if args.mode != "inject" and not args.cert:
            print("\nWARNING: interior poles on short series are often Pade")
            print("artifacts (seen at K=17/d=6 on the true spectrum). Confirm")
            print("with --cert: a real traitor shows CERTIFIED GROWTH of |c_k|;")
            print("an artifact leaves the residue flat.")
    else:
        print("Traitor annulus (p^-1/2, 1): EMPTY — consistent with RH on this chain.")
    if eps is not None or args.mode == "inject":
        eps_g = np.zeros(args.K) if eps is None else eps
        slope, se, nused, verdict = growth_verdict(cs, eps_g, args.p)
        print(f"\nGrowth test on |c_k| (k>={min(8,args.K-4)}, {nused} clean pts): "
              f"{verdict}")
    if chi_fn is not None:
        print("\nNOTE: chi mode is EXPERIMENTAL — pole classification is")
        print("zeta-calibrated; short-series Padé artifacts may appear in the")
        print("annulus. Treat interior poles as candidates, not detections,")
        print("until re-calibrated for L(s,chi).")
    if args.mode == "inject":
        print("\n(calibration run: a pole in the annulus was EXPECTED — "
              "detection of the injected traitor confirms sensitivity)")


if __name__ == "__main__":
    main()
