# Mathematical Contract

This document fixes the equations and interpretations that implementation must follow.

## 1. Coordinates and relative geometry

UAV position:
\[
p_u \in \mathbb{R}^3
\]

Obstacle position:
\[
p_o \in \mathbb{R}^3
\]

Relative mean:
\[
\mu = p_o - p_u
\]

Distance:
\[
d = \|\mu\|
\]

For \(d > \varepsilon_d\):
\[
n = \frac{\mu}{d}
\]

Thus \(n\) points from UAV to obstacle.

If \(d \le \varepsilon_d\), the geometry is degenerate. Do not manufacture a random direction.

## 2. Collision radius

\[
R_{\mathrm{collision}}
=
r_u+r_o+d_{\mathrm{margin}}
\]

## 3. Relative covariance

Assuming independent UAV and obstacle errors:
\[
\Sigma^r = \Sigma^o + \Sigma^u
\]

## 4. Directional uncertainty

\[
q = n^T \Sigma^r n
\]

After PSD-tolerance handling:
\[
\sigma = \sqrt{q}
\]

Do not inject `eps_sigma` into the physical variance.

## 5. Confidence coefficient

For per-evaluation risk tolerance \(\epsilon\):

\[
\beta = \Phi^{-1}(1-\epsilon)
\]

Recommended validation:
\[
0 < \epsilon < 0.5
\]

## 6. Chance-constrained safety margin

\[
c = d-\beta\sigma-R_{\mathrm{collision}}
\]

Interpretation:

- \(c>0\): conservative sufficient safety condition satisfied;
- \(c=0\): chance-constraint boundary;
- \(c<0\): sufficient chance constraint violated.

Do not claim \(c<0\) proves exact collision probability exceeds \(\epsilon\).

## 7. Safety-margin gradient

For fixed covariance and non-degenerate geometry:

\[
\nabla_{p_u}c
=
-n
+
\beta
\frac{(I-nn^T)\Sigma^r n}{d\sigma}.
\]

If \(\sigma \le \varepsilon_\sigma\):

\[
\nabla_{p_u}c = -n.
\]

If covariance is isotropic:

\[
\Sigma^r=\sigma_0^2I
\Rightarrow
\nabla_{p_u}c=-n.
\]

A strong test identity is:

\[
n^T \nabla_{p_u}c = -1
\]

for the non-degenerate full-gradient branch.

## 8. Attractive potential and force

Goal distance:
\[
d_g=\|p_u-p_g\|
\]

Potential:
\[
U_{\mathrm{att}}=
\begin{cases}
\frac12 k_a d_g^2,&d_g\le d_a\\
k_a d_a d_g-\frac12k_a d_a^2,&d_g>d_a
\end{cases}
\]

Force:
\[
F_{\mathrm{att}}=
\begin{cases}
k_a(p_g-p_u),&d_g\le d_a\\
k_a d_a\frac{p_g-p_u}{d_g},&d_g>d_a
\end{cases}
\]

At the goal, return zero force.

## 9. Constant-velocity state transition

State:
\[
x=[p^T,v^T]^T
\]

\[
A(\tau)
=
\begin{bmatrix}
I&\tau I\\
0&I
\end{bmatrix}
\]

## 10. Continuous white-acceleration process noise

Let:
\[
S_a=q_aI_3
\]

Then:
\[
Q(\tau)
=
\begin{bmatrix}
\frac{\tau^3}{3}S_a&
\frac{\tau^2}{2}S_a\\
\frac{\tau^2}{2}S_a&
\tau S_a
\end{bmatrix}
\]

`q_a` is acceleration-noise spectral density.

## 11. Kalman covariance update

Use Joseph form:

\[
P=
(I-KH)P^-(I-KH)^T
+
KRK^T.
\]

Then:
\[
P\leftarrow \frac12(P+P^T).
\]

## 12. Prediction grid

The safety/risk time grid includes:
\[
\tau=0
\]
and:
\[
\tau=T_p.
\]

If the nominal step does not exactly land on \(T_p\), append \(T_p\).

At zero horizon:
\[
A(0)=I,\qquad Q(0)=0.
\]

## 13. UAV nominal future position

Initial implementation:
\[
\hat p_u(t+\tau)
=
p_u(t)+v_u(t)\tau.
\]

If UAV uncertainty is disabled:
\[
\Sigma_u(\tau)=0.
\]

## 14. CPA

Current estimated relative position:
\[
r=p_o-p_u
\]

Current estimated relative velocity:
\[
v_r=v_o-v_u
\]

For non-negligible \(\|v_r\|^2\):

\[
t_{\mathrm{CPA}}
=
\mathrm{clip}
\left(
-\frac{r^Tv_r}{\|v_r\|^2},
0,
T_p
\right)
\]

If relative speed is below threshold:
\[
t_{\mathrm{CPA}}=0.
\]

Analytic diagnostic distance:
\[
d_{\mathrm{CPA}}
=
\|r+t_{\mathrm{CPA}}v_r\|.
\]

The final chance-constrained CPA margin must be evaluated using the prediction distributions at \(t_{\mathrm{CPA}}\).

## 15. Chance-constraint TTC

\[
TTC_{\mathrm{cc}}
=
\inf
\{\tau\in[0,T_p]:c(\tau)\le0\}.
\]

If:
\[
c(0)\le0,
\]
then:
\[
TTC_{\mathrm{cc}}=0.
\]

For a first sampled crossing:
\[
c_m>0,\quad c_{m+1}\le0,
\]
use:
\[
\alpha=\frac{c_m}{c_m-c_{m+1}}
\]
and:
\[
TTC_{\mathrm{cc}}
=
\tau_m+\alpha(\tau_{m+1}-\tau_m).
\]

If no crossing occurs:
\[
TTC_{\mathrm{cc}}=\infty.
\]

## 16. CPA risk

\[
R_{\mathrm{CPA}}
=
\frac{1}
{1+\exp((c_{\mathrm{CPA}}-\rho_c)/s_c)}.
\]

Use numerically stable evaluation.

## 17. TTC risk

\[
R_{\mathrm{TTC}}
=
\begin{cases}
\exp(-TTC_{\mathrm{cc}}/T_c),&TTC_{\mathrm{cc}}<\infty\\
0,&TTC_{\mathrm{cc}}=\infty
\end{cases}
\]

## 18. Combined dynamic risk

\[
\mathcal R
=
w_cR_{\mathrm{CPA}}
+w_tR_{\mathrm{TTC}}
\]

with:
\[
w_c,w_t\ge0,\qquad w_c+w_t=1.
\]

## 19. Temporal weights

\[
\omega_m
=
\frac{
e^{-\lambda_\tau\tau_m}
}{
\sum_j e^{-\lambda_\tau\tau_j}
}.
\]

Use a numerically stable normalized exponential implementation.

## 20. Frozen risk gain

At a control cycle:

\[
\eta
=
\eta_0(1+\lambda_R\mathcal R).
\]

Evaluate it and freeze it while constructing the local predictive field.

## 21. Predictive repulsive potential

\[
U_{\mathrm{rep}}
=
\eta
\sum_m
\omega_m
\exp(-c_m/\ell_r).
\]

Engineering exponent clipping may be applied in code.

## 22. Predictive repulsive force

With frozen risk/weights/covariance evaluation:

\[
F_{\mathrm{rep}}
=
\frac{\eta}{\ell_r}
\sum_m
\omega_m
\exp(-c_m/\ell_r)
\nabla_{p_u}c_m.
\]

There is **no additional leading minus sign** in this final implementation formula.

For isotropic covariance the force contribution is approximately in direction \(-n\).

## 23. Damping

Reference velocity when no global path exists:

\[
v_{\mathrm{ref}}
=
v_{\mathrm{des}}
\frac{p_g-p_u}{\|p_g-p_u\|}
\]

for nonzero goal distance.

\[
F_d=-k_v(v_u-v_{\mathrm{ref}}).
\]

## 24. Total command

\[
F_{\mathrm{cmd}}
=
F_{\mathrm{att}}
+
\sum_iF_{\mathrm{rep},i}
+
F_d.
\]

Then apply norm saturation.

The point-mass model converts it using:
\[
a_{\mathrm{raw}}=k_FF_{\mathrm{cmd}}
\]
followed by acceleration saturation.

## 25. Traditional APF baseline

Clearance:
\[
s=d-R_{\mathrm{collision}}.
\]

For \(0<s<s_0\):
\[
U_{\mathrm{trad}}
=
\frac12\eta_{\mathrm{trad}}
(1/s-1/s_0)^2.
\]

For \(s\ge s_0\):
\[
U_{\mathrm{trad}}=0.
\]

Force:
\[
F_{\mathrm{trad}}
=
-\eta_{\mathrm{trad}}
(1/s-1/s_0)s^{-2}n.
\]

If \(s\le0\), simulation collision logic takes precedence.
A minimum numerical clearance may protect standalone function evaluation.

## 26. Known theoretical/algorithmic limitations

1. Perfectly symmetric collinear head-on geometry with isotropic covariance may produce no lateral avoidance component.
2. The chance constraint is conservative and not the exact 3D collision probability.
3. Fixed-Q/R Kalman covariance does not automatically inflate when an unexpected maneuver creates a large residual.
4. Exponent clipping and force saturation break exact global potential-gradient equivalence.
5. `margin_gradient` is not a full gradient of the composite CPA/TTC risk.
