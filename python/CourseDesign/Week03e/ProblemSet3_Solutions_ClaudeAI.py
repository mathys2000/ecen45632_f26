# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-py
# ---

# %% [markdown]
# # ECEN 4/5632 Problem Set 3 Solutions
#
# from ClaudeAI
# ---
#

# %% [markdown]
# ## 1. Ideal Sinc Reconstruction
#
# Given: $x[n] = \delta[n] + 2\delta[n-1]$, sample period $T$.
#
# ### a) Reconstructed signal
#
# Ideal bandlimited reconstruction replaces each sample with a shifted, scaled sinc:
#
# $$x_r(t) = \sum_n x[n]\,\text{sinc}\!\left(\frac{t-nT}{T}\right) = \text{sinc}\!\left(\frac{t}{T}\right) + 2\,\text{sinc}\!\left(\frac{t-T}{T}\right)$$
#
# ### b) Verify sample values
#
# $$x_r(0) = \text{sinc}(0) + 2\,\text{sinc}(-1) = 1 + 2(0) = 1 \checkmark$$
#
# (using $\text{sinc}(-1) = \sin(-\pi)/(-\pi) = 0$)
#
# $$x_r(T) = \text{sinc}(1) + 2\,\text{sinc}(0) = 0 + 2(1) = 2 \checkmark$$
#
# This confirms the key sinc property: $\text{sinc}(k)=0$ for nonzero integers $k$, and $\text{sinc}(0)=1$, so the reconstruction passes exactly through each sample.
#
# ### c) Value at $t = T/2$
#
# $$x_r(T/2) = \text{sinc}(0.5) + 2\,\text{sinc}(-0.5)$$
#
# $$\text{sinc}(0.5) = \frac{\sin(\pi/2)}{\pi/2} = \frac{2}{\pi}, \qquad \text{sinc}(-0.5) = \frac{\sin(-\pi/2)}{-\pi/2} = \frac{2}{\pi}$$
#
# $$x_r(T/2) = \frac{2}{\pi} + 2\left(\frac{2}{\pi}\right) = \frac{6}{\pi} \approx 1.9099$$
#
# ---
#
# ## 2. Zero-Order Hold Frequency Response
#
# Given: $h_{ZOH}(t) = u(t) - u(t-T)$, a rectangular pulse of height 1 on $[0,T]$.
#
# ### a) Direct integration
#
# $$H_{ZOH}(j\Omega) = \int_0^T e^{-j\Omega t}\,dt = \left.\frac{e^{-j\Omega t}}{-j\Omega}\right|_0^T = \frac{1 - e^{-j\Omega T}}{j\Omega}$$
#
# ### b) Separate into linear-phase and magnitude terms
#
# Factor out the "half-delay" phase:
#
# $$H_{ZOH}(j\Omega) = e^{-j\Omega T/2}\cdot\frac{e^{j\Omega T/2}-e^{-j\Omega T/2}}{j\Omega} = e^{-j\Omega T/2}\cdot\frac{2\sin(\Omega T/2)}{\Omega}$$
#
# Writing the magnitude term as a normalized sinc:
#
# $$\boxed{H_{ZOH}(j\Omega) = T\,\text{sinc}\!\left(\frac{\Omega T}{2\pi}\right)e^{-j\Omega T/2}}$$
#
# - **Linear-phase term:** $e^{-j\Omega T/2}$ -- phase $= -\Omega T/2$, linear in $\Omega$, corresponding to a pure delay of $T/2$ (half a sample period), the group delay of the hold.
# - **Magnitude term:** $T\,\text{sinc}(\Omega T/2\pi)$ -- a sinc-shaped rolloff that suppresses high frequencies (the familiar "sinc droop" of a DAC).
#
# ### c) Normalized magnitude in dB
#
# Let $\omega = \Omega T$. Then $|H_{ZOH}|/T = |\text{sinc}(\omega/2\pi)|$.
#
# **At $\omega = 0.5\pi$:**
#
# $$\frac{\omega}{2\pi} = 0.25, \qquad \text{sinc}(0.25) = \frac{\sin(0.25\pi)}{0.25\pi} = \frac{0.7071}{0.7854} = 0.9003$$
#
# $$20\log_{10}(0.9003) \approx -0.91\text{ dB}$$
#
# **At $\omega = \pi$:**
#
# $$\frac{\omega}{2\pi} = 0.5, \qquad \text{sinc}(0.5) = \frac{2}{\pi} = 0.6366$$
#
# $$20\log_{10}(0.6366) \approx -3.92\text{ dB}$$
#
# (This is the well-known ZOH droop of about $-3.9$ dB at the Nyquist frequency.)
#
# ---
#
# ## 3. Uniform Quantization and SQNR
#
# Range: $\approx -1$ V to $+1$ V, full-scale span $= 2$ V.
#
# ### a) Step size for $B = 8$
#
# $$\Delta = \frac{2}{2^B} = \frac{2}{256} = \frac{1}{128} = 0.0078125\text{ V}$$
#
# ### b) Error variance and RMS error
#
# $$\sigma_e^2 = \frac{\Delta^2}{12} = \frac{(0.0078125)^2}{12} = 5.086\times10^{-6}\text{ V}^2$$
#
# $$\sigma_e = \sqrt{\sigma_e^2} = \frac{\Delta}{\sqrt{12}} = 0.002255\text{ V}\ (\approx 2.26\text{ mV})$$
#
# ### c) SQNR for 8-bit full-scale sine
#
# Standard result: for a full-scale sinusoid quantized with $B$ bits (uniform-error model),
#
# $$\text{SQNR(dB)} = 6.02\,B + 1.76$$
#
# $$\text{SQNR}_{8} = 6.02(8) + 1.76 = 48.16+1.76 = 49.92\text{ dB}$$
#
# ### d) SQNR for 16-bit full-scale sine
#
# $$\text{SQNR}_{16} = 6.02(16)+1.76 = 96.32+1.76 = 98.08\text{ dB}$$
#
# ### e) Reconciling "96 dB" vs "98 dB" for 16-bit audio
#
# These numbers are not contradictory because they answer **different questions**:
#
# - **"~96 dB"** is the common rule-of-thumb figure $6.02\,B \approx 96.3$ dB. It represents the *dynamic range* of the quantizer in a loose sense -- essentially $20\log_{10}(2^B)$, the ratio of the full-scale range to the quantization step, without including the $+1.76$ dB correction factor. It's often quoted as a quick "6 dB per bit" estimate and is sometimes meant as a conservative, generic dynamic-range spec rather than a precise SQNR for one specific signal.
# - **"~98 dB"** is the *exact* SQNR formula result for the specific case of a **full-scale sinusoidal** test signal, which includes the $+1.76$ dB term that comes from the ratio of a sine wave's mean-square power to its peak value ($V_{rms}^2 = V_{pk}^2/2$).
#
# So the 96 dB figure is a rounded/approximate dynamic-range statement (or omits the sine-specific correction), while 98 dB is the precise theoretical SQNR for a full-scale sine tone. Both are legitimate -- they simply correspond to slightly different definitions/assumptions (generic dynamic range vs. exact full-scale-sinusoid SQNR), and real-world 16-bit audio numbers can also be affected by dither, noise shaping, and measurement weighting, which further explains why quoted "16-bit" figures vary in the literature.
#
# ---
#
# ## 4. Finite Sequence z-Transform
#
# Given: $x[n] = 2\delta[n+1] - \delta[n] + 3\delta[n-2]$
#
# ### z-Transform
#
# $$X(z) = \sum_n x[n]z^{-n} = 2z^{1} - 1 + 3z^{-2}$$
#
# $$\boxed{X(z) = 2z - 1 + 3z^{-2}}$$
#
# ### ROC
#
# The sequence has both a positive-time term ($2z^1$, i.e. nonzero at $n=-1$) and negative-time-index terms extending to $n=2$ ($3z^{-2}$). Because there is a term $2z$ that diverges as $|z|\to\infty$, and a term $3z^{-2}$ that diverges as $|z|\to 0$, the ROC is the entire $z$-plane except the origin and infinity:
#
# $$\text{ROC: } 0 < |z| < \infty$$
#
# ### Does the DTFT exist?
#
# Since $x[n]$ has **finite support** (only 3 nonzero samples), the sum defining $X(e^{j\omega})$ is a finite sum and converges for every $\omega$. The unit circle $|z|=1$ lies within the ROC, so **yes, the DTFT exists** (it is simply $X(e^{j\omega}) = 2e^{j\omega} - 1 + 3e^{-j2\omega}$).
#
# ---
#
# ## 5. Right-Sided Exponential and Convergence
#
# Given: $x[n] = (1.2)^n u[n]$
#
# ### z-Transform via geometric series
#
# $$X(z) = \sum_{n=0}^{\infty}(1.2)^n z^{-n} = \sum_{n=0}^\infty (1.2\,z^{-1})^n$$
#
# This geometric series converges when $|1.2\,z^{-1}| < 1$, i.e. $|z| > 1.2$, giving:
#
# $$\boxed{X(z) = \frac{1}{1-1.2\,z^{-1}}, \qquad \text{ROC: } |z|>1.2}$$
#
# ### Pole
#
# $$\text{Pole at } z = 1.2$$
#
# ### Does the DTFT exist?
#
# No. The ROC $|z|>1.2$ does **not** include the unit circle $|z|=1$ (since $1.2>1$). Equivalently, $(1.2)^n u[n]$ grows without bound and is not absolutely (or even square) summable, so the DTFT does not converge.
#
# ---
#
# ## 6. Same Algebra, Different ROC
#
# Given: $X(z) = \dfrac{1}{1-0.75\,z^{-1}}$
#
# ### a) ROC: $|z| > 0.75$
#
# This ROC is the exterior of the pole, associated with a **right-sided (causal)** sequence:
#
# $$x[n] = (0.75)^n u[n]$$
#
# ### b) ROC: $|z| < 0.75$
#
# This ROC is the interior of the pole, associated with a **left-sided (anticausal)** sequence:
#
# $$x[n] = -(0.75)^n u[-n-1]$$
#
# ### Why the same rational expression gives two different answers
#
# The algebraic expression $X(z)$ alone only fixes the poles/zeros; the **ROC** determines which side of each pole the power series is expanded on (causal vs. anticausal), so specifying the ROC is essential to uniquely invert a rational $X(z)$.
#
# ---
#
# ## 7. Partial Fractions with Two ROCs
#
# Given: $X(z) = \dfrac{1}{(1-0.5z^{-1})(1-0.8z^{-1})}$
#
# ### Partial fraction expansion
#
# Write $X(z) = \dfrac{A}{1-0.5z^{-1}} + \dfrac{B}{1-0.8z^{-1}}$.
#
# Substituting $x = z^{-1}$: $1 = A(1-0.8x) + B(1-0.5x)$
#
# - At $x = 1/0.5 = 2$: $\ 1 = A(1-1.6) = -0.6A \ \Rightarrow\ A = -\dfrac{5}{3}$
# - At $x = 1/0.8 = 1.25$: $\ 1 = B(1-0.625) = 0.375B \ \Rightarrow\ B = \dfrac{8}{3}$
#
# (Check: $A+B = -5/3+8/3 = 1$ âœ“, matching $X(z)|_{x=0}=1$.)
#
# $$X(z) = \frac{-5/3}{1-0.5z^{-1}} + \frac{8/3}{1-0.8z^{-1}}$$
#
# Poles at $z=0.5$ and $z=0.8$.
#
# ### a) ROC: $|z| > 0.8$ (outside both poles -- causal/stable-looking, right-sided)
#
# Both terms are inverted as right-sided sequences:
#
# $$\boxed{x[n] = -\frac{5}{3}(0.5)^n u[n] + \frac{8}{3}(0.8)^n u[n]}$$
#
# ### b) ROC: $0.5 < |z| < 0.8$ (between the two poles)
#
# This ROC lies **outside** the pole at $z=0.5$ (so that term is right-sided) but **inside** the pole at $z=0.8$ (so that term is left-sided/anticausal):
#
# $$\boxed{x[n] = -\frac{5}{3}(0.5)^n u[n] \;-\; \frac{8}{3}(0.8)^n u[-n-1]}$$
#
# This is a genuinely **two-sided** sequence -- decaying for $n\to+\infty$ (from the $0.5$-pole term) and also decaying for $n\to-\infty$ (from the $0.8$-pole term), which is exactly what an annular ROC between two poles signals.
#

# %%
