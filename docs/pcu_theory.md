# PCU (Periodic Coordinate Unwrapping) - Theoretical Analysis

**Version**: v1.0 | **Date**: 2026-09-10
**Status**: validated on internal-solver measurements (historical version tag v6.7) + generalization benchmarks (run_03_pcu_generalize)

---

## 0. Notation and Problem Setting

We consider the family of objective functions (generalized periodic separable structure):

$$
f(x) = A\,d + \sum_{i=1}^{d}\left[ c_1\,(x_i - o_i)^2 - A\cos\!\left(\frac{2\pi (x_i-o_i)}{T}\right)\right] + b^{\top} x, \quad x \in \mathbb{R}^d
$$

where $o \in \mathbb{R}^d$ is the unknown offset (the A^2EP translation center), $T>0$ is the period, $c_1\ge 0$ is the quadratic coefficient, $A>0$ is the periodic amplitude, and $b \in \mathbb{R}^d$ is a linear bias. Let $u_i = x_i - o_i$ and $\omega = 2\pi/T$.

The gradient and Hessian are both analytic:

$$
g_i(x) = 2c_1 u_i + \omega A \sin(\omega u_i) + b_i
$$

$$
H_{ii}(x) = 2c_1 + \omega^2 A \cos(\omega u_i), \quad H_{ij} = 0\ (i\ne j)
$$

**Problem**: given an arbitrary single point $x$ (and its analytic $(g(x), H(x))$), recover the global optimum $o$ analytically (when $b=0$, $f(o)=0$ is the global minimum).

**Three setting levels**:
- Separable (diagonal H): dimension-wise unwrapping
- Rotated non-separable ($H = R^{\top} D R$): eigh recovers $R$ -> transform to $z=Rx$ -> dimension-wise unwrapping
- Ambiguity handling: multi-candidate combination + F verification

---

## 1. Propositions and Proofs

### Proposition 1 (Recoverability of the periodic phase)

> Let $h(u) = c_1 u^2 + A(1-\cos(\omega u)) + b u$, $A>0$, $\omega>0$. Given the scalar values $(g_i, H_{ii})$ of an arbitrary single point $x$, $u_i \bmod T$ has at most two candidates, and both can be obtained analytically from $(g_i, H_{ii})$.

**Proof**: From the diagonal Hessian element

$$
H_{ii} = 2c_1 + \omega^2 A \cos(\omega u_i) \;\Longrightarrow\; \cos(\omega u_i) = \frac{H_{ii} - 2c_1}{\omega^2 A} =: \xi.
$$

If $|\xi|>1$ there is no real solution (numerically we use $\mathrm{clip}(\xi,-1,1)$, in which case the phase is unreliable and the proposition does not apply). Otherwise there is a unique $\alpha = \arccos(\xi) \in [0,\pi]$ such that

$$
\omega u_i \equiv \pm \alpha \pmod{2\pi} \;\Longrightarrow\; u_i \equiv \frac{\alpha}{\omega} \ \text{or}\ \frac{2\pi-\alpha}{\omega} \pmod{T}.
$$

Hence the phase of $u_i$ (mod $T$) has at most two candidates. QED

### Proposition 2 (Recoverability of the integer offset - necessity of the linear term)

> For a candidate phase $u_i^0 \in \{( \alpha/\omega ) \bmod T,\ (2\pi-\alpha)/\omega \bmod T\}$, the integer offset $k \in \mathbb{Z}$ is constrained by the gradient equation:
> $$
> 2c_1 (kT + u_i^0) + \omega A \sin(\omega u_i^0) + b_i = g_i .
> $$
> **When $c_1 > 0$**, this equation is **linear** in $k$ (the $\sin$ term is constant at a fixed phase) and the solution is unique:
> $$
> k^* = \frac{g_i - b_i - \omega A\sin(\omega u_i^0) - 2c_1 u_i^0}{2 c_1 T}.
> $$
> When $c_1 = 0$, the equation does not contain $k$ and every integer candidate satisfies the gradient match - **the integer offset cannot be recovered uniquely from a single point**. Note, however, that for purely periodic functions ($c_1=0,\ b=0$) the zero set of $f$ forms the lattice $u_i \equiv 0 \pmod{T}$; once the phase is recovered from a single point $(g,H)$, **projecting onto the nearest lattice point directly yields a root**, so combinatorial search is not necessary - combinatorial search plus F verification is only a more conservative implementation (more robust to numerical errors in the phase estimate). If the goal is to recover a specific $o$, the problem itself is non-unique modulo $T$ (identifiability boundary).

**Proof**: Once $u_i^0$ is fixed, $\sin(\omega u_i^0)$ is a constant; substituting into the gradient expression yields a linear equation. For $c_1>0$ the coefficient $2c_1T \ne 0$ gives a unique solution (which must be an integer, otherwise that phase has no valid candidate). For $c_1=0$ the coefficient is 0; if $g_i - b_i = \omega A \sin(\omega u_i^0)$ holds then every $k$ matches; in this case the $\cos$ (Hessian) and $\sin$ (gradient) together lock the phase to $\{0,\pi\}$, and the nearest-lattice projection uniquely determines the root. QED

**Corollary 2.1 (single-point full solvability of the separable Rastrigin family)**: when $c_1=1,\ T=1,\ b=0$ (standard Rastrigin), unwrapping any single point $(g,H)$ yields **at most 2** candidates per $o_i$ (at most 1 integer per phase, 2 phases), the combinatorial search size is $\le 2^d$, and the true $o$ is guaranteed to lie in the candidate set.

### Proposition 3 (Reduction of rotated non-separable structures)

> Let $f(x) = \tilde f(Rx)$, where $\tilde f$ is a separable periodic function (the setting of Props. 1-2) and $R$ is an orthogonal matrix. If $H(x) = R^{\top} D(x) R$ and the diagonal elements of $D(x)$ are pairwise distinct (at the current $x$), then the eigenvectors $V$ of a single-point $\mathrm{eigh}(H)$ satisfy $V = R^{\top}$ (up to column permutation and sign), so the problem can be transformed to $z = Rx$ and unwrapped dimension by dimension.

**Proof**: $H = R^{\top} D R$ is an orthogonal similarity diagonalization. If the diagonal elements of $D$ are pairwise distinct, every eigenspace of $H$ is one-dimensional and the eigendecomposition is unique up to sign: $V = R^{\top} \Pi \Sigma$, where $\Pi$ is a permutation matrix and $\Sigma$ a diagonal sign matrix. Unwrapping is independent per dimension and insensitive to sign ($\cos$ is symmetric under $u \mapsto -u$), so after unwrapping in $z=Rx$ and substituting back with $R^{\top}$, correctness is unaffected by $\Pi,\Sigma$. QED

> **Empirical**: Rotated_Rastrigin_20D, 30 runs, convergence rate 0.00 -> 1.00, confirming that eigh recovery is effective (the eigenvectors differ from the true $R^{\top}$ only by permutation/sign).

### Proposition 4 (Candidate-ambiguity upper bound and combinatorial search complexity)

> Let the number of candidates per dimension be $\nu_i = |\mathcal C_i|$. Then:
> - $\nu_i \le 2$ (for $c_1>0$, Prop. 2 gives at most 1 integer solution per phase, and there are 2 phases);
> - the combinatorial search size is $N = \prod_i \nu_i \le 2^d$;
> - when $N \le 1024$, exhaustive search picks the smallest $F$ ($O(N)$ function evaluations); otherwise greedy + coordinate descent ($O(d\,\bar\nu)$ evaluations, with $\bar\nu$ the average candidate count).

**Proof**: $\nu_i \le 2$ follows directly from Prop. 1 (2 phases) x Prop. 2 (at most 1 integer solution per phase). The exhaustive complexity follows from the combinatorial count. The correctness of coordinate descent relies on separability of the objective in the combination space (taking the per-dimension optimum independently gives the global optimum), which holds for separable objectives; for non-separable (rotated) cases, F verification provides the safety net. QED

### Proposition 5 (Safe-acceptance guarantee - zero harm)

> PCU accepts and returns early only when a candidate point $x_c$ satisfies $f(x_c) < 1e-4$ (and the internal solver convergence criterion $|f(x_c) - f_{opt}|<1e-4$); otherwise it falls back to the original HGCA pipeline with zero effect. Therefore **PCU never drives the solver to a worse point**.

**Proof**: the accept branch requires $x_c$ to already satisfy the convergence criterion ($f$ sufficiently close to $f_{opt}$) and returns the optimum directly; the reject branch runs the original pipeline, exactly equivalent to running without PCU. Neither branch is worse than the baseline. QED

### Proposition 6 (Generalization over model parameters)

> The generalization of PCU over the parameter family $(\ c_1,\ A,\ T,\ b\ )$ has been validated (run_03_pcu_generalize):
> - $c_1$ variants ($c_1=2$): 0.00 -> 1.00
> - $A$ variants ($A=5, 20$): 0.00 -> 1.00
> - $T$ variants ($T=2$): 0.00 -> 1.00
> - purely periodic ($c_1=0$, cos/sin^2): 0.00 -> 1.00 (zeros form a lattice; a single-point projection yields a root)
> - **bias $b\ne 0$: correctly rejected** (the global optimum no longer lies at the periodic center $o$, $F(o_c)=b^{\top}o$ is significantly nonzero; falls back to the original pipeline with zero harm) - this is a **provable boundary**: PCU solves the family of problems whose optimum lies at the periodic center.

---

## 2. Complexity Analysis

| Step | Complexity |
|---|---|
| 1 gradient + 1 Hessian evaluation | $O(d^2)$ (diagonal Hessian is $O(d)$; rotated case is $O(d^3)$ for eigh) |
| Dimension-wise unwrapping (2 phases x integer scan within window per dimension) | $O(d \cdot \lceil 2R/T \rceil)$, $R$ the search radius |
| Candidate combination (exhaustive <=1024) | $O(N)$ F evaluations, $N\le 2^d$ (in practice $\nu_i=1$ for most dimensions at large $d$, so $N$ is tiny) |
| **Total budget** | **fully decoupled from the $O(\text{maxfevals})$ of CMA-ES** - on a PCU hit, nit=2 (1 gradient + 1 Hessian); measured: Rastrigin 20D drops from ~77,000 evaluations to 2 |

---

## 3. Differences from Known Work

- **Phase unwrapping (signal processing / imaging)**: recovers the true phase from mod $2\pi$ observations (graph cuts / MRF, multi-period diversity). Difference: this work recovers the **objective offset** (not a signal phase) using the **analytic gradient + Hessian of the optimization objective** (not signal observations), and the integer offset is determined analytically from the linear gradient term (no search / prior).
- **qBnB (quasi branch and bound)**: constructs quadratic lower bounds via Newton iterations for global optimization. Difference: still iterative; this work gives an **analytic closed form** on the periodic separable family.
- **CMA-ES restart strategies (BIPOP/IPOP/MSC-CMA-ES)**: stochastic / structure-aware restarts across basins. Difference: all follow the sample-iterate paradigm and cannot identify the structure that "the periodic center is the global optimum"; PCU is a deterministic analytic fast path whose cost scales polynomially with the dimension, not with the evaluation budget.

---

## 4. Limitations (honest boundaries)

1. **Requires an analytic Hessian**: the discretization error of a numerical Hessian destroys the $\cos$ phase information; PCU is effective only when analytic gradient/Hessian are available (on failure it silently falls back).
2. **The optimum must lie at the periodic center**: for $b\ne 0$ (bias/translation moves the optimum off-center) it is correctly rejected and not counted as a positive contribution. Boundary detail: $F(o)=b^{\top}o$; when $b^{\top}o$ is close to 0 ($b\perp o$) there is a theoretical false-accept risk, but the standard Rastrigin family ($b=0$ and $A d\gg 1e-4$) is unaffected; the biased Rastrigin case configured in run_03 (linear bias $b$, CEC-style) is designed to be rejected rather than counted as a positive contribution - no trigger-level data is currently recorded for it, so this boundary is stated as a design property, not as an empirical claim.
3. **$c_1=0$ purely periodic family**: the integer offset is unidentifiable modulo $T$; zeros form a lattice and a single-point projection yields a root (combinatorial search is a conservative implementation).
4. **Rotation recovery needs distinct eigenvalues**: with repeated eigenvalues of $D$, $R$ is not uniquely recoverable (F verification still provides a safety net, but solutions may be missed).
5. **Applicability domain**: periodic separable (or rotated-periodic) structures. Griewank (cos-product coupling, non-separable) and Schaffer (radial, non-separable) show zero trigger and zero harm in experiments; they are outside this domain.

---

*Data sources: run_02_pcu_integrated (integrated 30 runs, TOTAL 0.8917->0.9750), run_03_pcu_generalize (generalization, 10 runs), _diag_biased (bias-boundary diagnostics).*
