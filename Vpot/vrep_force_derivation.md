# V'_rep from DFT forces — derivation and validity

Reference note for `Hamiltonian.py:585-599` (`Vrep_fit.__init__`, the `derivF` pool).
Atomic units throughout; `collect_traj` converts eV -> Ha and eV/Ang -> Ha/a0.

```python
u = X[min(i+1,len(keep)-1)] - X[max(i-1,0)]
u /= np.linalg.norm(u)
g = self.get_g(fr,d,bond)      # grad_X sum_bonds r_IJ ; N*dRds = g.u
NdRds = float(g.ravel()@u.ravel())
dEds = self.finite_diff(self.E_wr(fr.coords+s*u,fr.atoms,q,alpha),
                        self.E_wr(fr.coords-s*u,fr.atoms,q,alpha), s)
Vp.append(-(float((fr.F*u).sum()) + dEds)/NdRds)
```

`E_wr` = `E_elec` + `E_other`, where `E_other` adds the C-C and H-H repulsion read from
`Vpot/par/EurPhysJD_67_38_2013/{C_C,H_H}.par`, reproducing hotbit's
`tab={'CH':...,'rest':'default'}` so the residual is pure C-H. On this dataset it is a
measured no-op (H-H's table ends at 1.767 a0, well inside methane's ~3.4 a0 H-H
distances; C-C is frozen along both scans that contain it).

---

## 1. The model

For a configuration $X \in \mathbb{R}^{3n}$ (all atomic coordinates stacked),

$$E_{\rm DFTB}(X) \;=\; E_{\rm elec}(X) \;+\; \sum_{I<J} V_{\rm rep}\!\left(r_{IJ}(X)\right)$$

`E_elec` is `self.E_elec` (band energy + SCC Coulomb, via `H.E_tot`). The `bond` mask
selects the $N$ pairs of the type being fitted that lie inside `Rcut`; every other pair
is assumed to contribute nothing (either $r > R_{\rm cut}$, or a different element pair
fitted elsewhere).

The `spread > tol` guard enforces $r_{IJ} = R$ for all $N$ selected pairs of a frame, so

$$\sum_{I<J} V_{\rm rep}(r_{IJ}) \;=\; N\,V_{\rm rep}(R), \qquad
R = \bar r = \frac1N \sum_{(I,J)\in\text{bond}} r_{IJ}$$

One geometry -> one unknown $R$. That is what makes the pointwise inversion possible.

## 2. Two independent fit equations

**Energy condition** (`Ediff_tot`, line 587) — pool `deriv`:

$$V_{\rm rep}(R) = \frac{E_{\rm DFT} - E_{\rm elec}}{N}$$

differentiated afterwards by spline (`F_R = E_R(R,1)`).

**Force condition** — pool `derivF`, the snippet. Require the gradients to match too:

$$\nabla_X E_{\rm DFTB} \;=\; \nabla_X E_{\rm DFT} \;=\; -\,\mathbf F^{\rm DFT}$$

This gives $V'_{\rm rep}$ pointwise, with no differentiation of a fitted curve.

## 3. Projection on a direction

A gradient in $3n$ dimensions cannot be inverted component-wise for the single scalar
$V'_{\rm rep}$, so contract both sides with a unit direction $u$ and use the chain rule
along $s \mapsto X + s u$:

$$\frac{d}{ds} E_{\rm DFTB}(X+su)\Big|_0
= \underbrace{\nabla E_{\rm elec}\cdot u}_{\texttt{dEds}}
+ \sum_{I<J} V'_{\rm rep}(r_{IJ})\,\frac{dr_{IJ}}{ds}$$

At the base point all selected $r_{IJ}$ equal $R$, so $V'_{\rm rep}$ factors out of the
sum **even though the individual $dr_{IJ}/ds$ differ**:

$$\sum_{(I,J)} V'_{\rm rep}(r_{IJ})\,\frac{dr_{IJ}}{ds}
= V'_{\rm rep}(R) \sum_{(I,J)} \frac{dr_{IJ}}{ds}
= N\,V'_{\rm rep}(R)\, \underbrace{\frac{d\bar r}{ds}}_{\texttt{dRds}}$$

This is what licenses differentiating the **mean** bond length. The DFT side is

$$\frac{d}{ds} E_{\rm DFT}(X+su)\Big|_0 = \nabla E_{\rm DFT}\cdot u
= -\sum_I \mathbf F_I \cdot u_I = -\,\texttt{(fr.F*u).sum()}$$

(the elementwise product summed over atoms *and* Cartesian components is the full
$3n$-dimensional inner product). Equating the two:

$$-\,\mathbf F\cdot u \;=\; \frac{dE_{\rm elec}}{ds} + N\,V'_{\rm rep}(R)\,\frac{d\bar r}{ds}$$

$$\boxed{\;V'_{\rm rep}(R) \;=\; -\,\frac{\mathbf F\cdot u + \dfrac{dE_{\rm elec}}{ds}}
{N\,\dfrac{d\bar r}{ds}}\;}$$

which is line 598 verbatim.

## 4. The choice of u

`X` is the stack of kept geometries in trajectory order, so
`X[i+1] - X[i-1]` is a central difference in *frame index*: the local tangent of the
trajectory in configuration space, one-sided at the endpoints via the `min`/`max` clamp.

- The formula divides by `dRds`, so $u$ **must** have a component along bond stretching.
  The trajectory tangent is the direction in which the scan actually changes $R$, so
  `dRds` is $O(1)$. There is no guard against `dRds ~ 0`: a trajectory whose consecutive
  frames differ mostly by rotation or bending would blow up here.
- It is also the best-conditioned direction numerically — the displacement the data
  itself explores.

`u /= norm(u)` **cancels** in the final ratio (`F·u`, `dEds`, `dRds` are all linear in
$u$ to first order). It is there only so that `h*u` is a displacement of physical size
$h = 0.02\,a_0$, i.e. so that $s$ is arc length and the finite-difference step is
controlled.

Translational and rigid-rotational components of $u$ are harmless: both give zero on the
left ($\sum_I \mathbf F_I = 0$; zero net contribution from rotation follows from
rotational invariance of $E_{\rm DFT}$) and zero on the right ($E_{\rm elec}$ and
$\bar r$ are invariant). Only the internal part of $u$ carries information.

## 5. One finite difference, one analytic derivative

```
dEds = (E_wr(X+su) - E_wr(X-su)) / (2s)      # numerical, s = 0.02 a0
NdRds = g . u                                 # analytic
```

`dEds` is a central difference, error $O(s^2)$; each costs a full SCF, hence two extra
`E_wr` calls per frame. Numerical because there is no analytic $\nabla E_{\rm elec}$ yet
— that needs Hellmann-Feynman + Pulay terms from $\partial H^0/\partial R$ and
$\partial S/\partial R$.

`N\,d\bar r/ds` is **exact**, not differenced. Since $\partial r_{IJ}/\partial X_I =
\hat{\mathbf n}_{IJ}$ with $\hat{\mathbf n}_{IJ} = (X_I-X_J)/r_{IJ}$,

$$\mathbf g \;\equiv\; \nabla_X \!\!\sum_{(I,J)\in\text{bond}}\!\! r_{IJ},
\qquad \mathbf g_K = \sum_J \hat{\mathbf n}_{KJ},
\qquad N\,\frac{d\bar r}{ds} = \mathbf g\cdot u$$

which is `get_g` followed by one inner product — $O(N_{\rm bonds})$, no divide-guard, and
it holds the pair set fixed by construction (the mask is an argument, so no pair can cross
`Rcut` mid-difference the way it could with a displaced-geometry `self.bonds` call).
Verified against the finite difference it replaced to $5\times10^{-10}$, and against the
equivalent matrix form $\sum \hat{\mathbf n}\cdot\Delta u$ to $2\times10^{-16}$.

**$\mathbf g$ is not $u$, and $\hat{\mathbf g}$ must not be substituted for it.** $u$ is
data — the direction the reference trajectory actually displaces along; $\mathbf g$ is
geometry — a property of the bond set at that configuration. Measured angles between them:
45.0° (CH-), 48.2° (methane), 65.9° (ethyne), 70.8° (benzene). CH- shows why: only H moves
along the scan, so $u$ has support on one atom while $\mathbf g$ has support on both
($\mathbf g_C = -\hat x$, $\mathbf g_H = +\hat x$, $|\mathbf g| = \sqrt2$,
$\mathbf g\cdot u = 1$). Setting $u = \hat{\mathbf g}$ replaces the physical displacement
with one that also moves C-C — see the appendix.

## 6. Downstream

`o = R.argsort()` sorts frames by bond length; `np.array(Vp)[o]` is reordered with the
same permutation, so `derivF` holds $(R_k, V'_{\rm rep}(R_k), w)$ triples.
`integration()` merges coincident $R$, appends the anchor $V'_{\rm rep}(R_{\rm cut}) = 0$
with a large weight, smooths with `splrep(..., s=lam)` and integrates back:

$$V_{\rm rep}(r) = -\int_r^{R_{\rm cut}} V'_{\rm rep}(r')\,dr' = A(r) - A(R_{\rm cut})$$

`deriv` and `derivF` are two independent routes to the same $V'_{\rm rep}$ — one from
differentiating the energy residual curve, one from the DFT forces frame by frame.
Comparing them is a genuine consistency check.

Measured at `Rcut = 3.0236` (hotbit's 1.6 Ang): the two integrated curves agree to
**max 0.0053 Ha, rms 0.0014** — several times tighter than either route's distance to
`Vpot/par/C_H_repulsion.par` (0.0389 and 0.0442). Point by point, mean $-0.0026$,
rms $0.0083$, max $0.029$ Ha/a0, and every large deviation sits at a **trajectory
endpoint** (1.3228 / 3.0236 for CH-, 1.6563 / 2.9977 for benzene) where the energy
route's `CubicSpline` hits its not-a-knot end condition and the force route's tangent
degenerates to a one-sided difference. Interior points agree to ~0.001.

---

# Appendix — on the "chain rule ratio" argument

A plausible-looking derivation of the same formula runs: write the total differentials
$dE = \nabla E\cdot d\mathbf r$ and $dR = \nabla R\cdot d\mathbf r$ with
$d\mathbf r = u\,ds$; take the ratio; decompose
$\nabla E = \frac{dE}{dR}\nabla R + \nabla E_\perp$; assume
$\nabla E_\perp \cdot u \approx 0$ because "$R$ dominates the displacement"; conclude
that $dE/dR = (dE/ds)/(dR/ds)$ is the exact physical derivative.

The directional-derivative algebra is fine, and the observation that the ratio kills the
*magnitude* of $u$ is correct. The rest is not.

**(a) The code never forms `dEds/dRds`.** Line 598 has
$-(\mathbf F\cdot u + \texttt{dEds})/(N\,\texttt{dRds})$. The numerator is not
$dE_{\rm elec}/ds$; it is the directional derivative of the **residual**

$$E_{\rm res}(X) \equiv E_{\rm DFT}(X) - E_{\rm elec}(X), \qquad
\frac{dE_{\rm res}}{ds} = -\,\mathbf F\cdot u - \texttt{dEds}$$

so what is computed is

$$V'_{\rm rep}(R) = \frac{1}{N}\,\frac{dE_{\rm res}/ds}{d\bar r/ds}$$

That distinction is the whole point: $E_{\rm DFT}$ or $E_{\rm elec}$ *alone* is not a
function of $R$ — it depends on every internal coordinate. $E_{\rm res}$ is a function of
$R$ alone by construction of the model.

**(b) The argument contradicts itself.** It assumes $\nabla E_\perp\cdot u \approx 0$ and
then claims the *exact* derivative. Both cannot hold. As a statement about the total
energy the assumption is simply false: nothing makes the component of
$\nabla E_{\rm DFT}$ orthogonal to $\nabla R$ orthogonal to the trajectory tangent.

**(c) The decomposition is circular.** Writing
$\nabla E = \frac{dE}{dR}\nabla R + \nabla E_\perp$ *defines* $dE/dR$ as the projection
coefficient of $\nabla E$ onto $\nabla R$; recovering it by dividing is the definition
read backwards, not a proof. It coincides with the chain-rule $dE/dR$ only when $E$
genuinely factors through $R$.

**(d) $|u|$ cancels; the direction of $u$ does not.** The answer is independent of the
*direction* of $u$ only if $\nabla E_{\rm res} \parallel \nabla \bar r$ exactly. That is
not a theorem — it is the model ansatz.

**The correct version.** No perpendicular-gradient approximation is needed. Since

$$E_{\rm res}(X) \;\overset{!}{=}\; \sum_{I<J} V_{\rm rep}(r_{IJ})
\;\overset{\text{equal bonds}}{=}\; N\,V_{\rm rep}(\bar r)$$

$E_{\rm res}$ is a composite function of $X$ through $\bar r$ alone, so by the chain rule,
*exactly*,

$$\nabla_X E_{\rm res} = N\,V'_{\rm rep}(R)\,\nabla_X \bar r
\;\;\Longrightarrow\;\;
\frac{dE_{\rm res}}{ds} = N\,V'_{\rm rep}(R)\,\frac{d\bar r}{ds}
\quad\text{for \emph{any} } u$$

Here $\nabla E_\perp \equiv 0$ identically — not "small". The method inverts a *model*,
it does not extract a physical derivative. If the ansatz is imperfect (three-body
character in the true residual, another pair type inside `Rcut`, unequal bonds), then
$\nabla E_{\rm res}$ acquires a genuine component off $\nabla\bar r$ and the answer *does*
depend on which $u$ was picked. That $u$-dependence is fit error made visible.

So the real assumption is not about the trajectory at all: it is that the residual is
pairwise, short-ranged, and confined to the masked pairs. The trajectory tangent enters
only for conditioning — $d\bar r/ds$ must stay away from zero.

**Two caveats dropped by that argument.** Nothing is exact numerically: both finite
differences carry $O(h^2)$ error at $h = 0.02\,a_0$, and $E_{\rm elec}$ is an SCF result,
so the SCF threshold (`1e-5`) sets a noise floor on `dEds` that is amplified by
$1/(N\,\texttt{dRds})$. And $\mathbf F$ and $E_{\rm DFT}$ must come from the same
functional / basis / geometry, or $\nabla E_{\rm res}$ is not the gradient of the
$E_{\rm res}$ fitted in `Ediff_tot`.

## A cheap check — done, 2026-09-07/08

If the ansatz holds, $V'_{\rm rep}$ is invariant to the choice of $u$. The check was run
with $u = \hat{\mathbf g}$, the pure bond-stretch direction, against the trajectory
tangent. It does **not** agree, and the spread is diagnostic:

| $u$ | max \|V - hotbit\| | rms |
|---|---|---|
| trajectory tangent | **0.0429** | **0.0102** |
| $\hat{\mathbf g}$ | 0.1901 | 0.1237 |

The tangent wins because it is *accidentally* orthogonal to $\nabla r_{CC}$ in the ethyne
and benzene scans — both freeze C-C. With $u = \hat{\mathbf g}$ the C-C coefficient is
exactly $-1/2$, giving constant per-molecule offsets of $+0.34$ (ethyne, $r_{CC}=2.2761$)
and $+0.126$ (benzene, $r_{CC}=2.6363$). That is the $u$-dependence this section predicts:
fit error made visible, from a second pair type inside `Rcut`.

Methane's C-H and H-H stretch gradients are collinear to $0.28^\circ$, so no choice of $u$
can separate them; a Gram-Schmidt projection leaves 0.5% of the denominator and a 2x2
channel solve has $\mathrm{cond}(A)\approx200$. It turned out not to matter: hotbit's H-H
repulsion table terminates at 1.767 a0 while methane's H-H distances are ~3.4 a0, so
$V'_{HH}\equiv0$ there and the coefficient multiplies zero.

**What this check does not measure.** The residual gap to hotbit is *not* fit error of
this kind. Our SK integrals, on-site energies and Hubbard $U$ reproduce hotbit's to
$10^{-4}$ / 2 mHa, and substituting hotbit's own H and S tables changes $V'_{\rm rep}$ by
0.09% across the whole CH- scan. `C_H_repulsion.par` is exactly a degree-4 polynomial
(residual $9\times10^{-9}$), i.e. its $V'_{\rm rep}$ is a single global cubic with zero
interior knots, which cannot follow a steepening short-range repulsion. Our points sitting
13-15% steeper below 1.6 a0 is the reference being over-smoothed, not our error —
recomputing the compressed CH- point from the DFT *forces* rather than the energy spline
makes it steeper still ($-1.163$ vs $-1.133$; hotbit's curve says $-0.984$).
