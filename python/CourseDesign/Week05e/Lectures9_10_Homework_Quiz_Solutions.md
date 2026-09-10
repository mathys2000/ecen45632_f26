# Lectures 9–10 Homework and Weekly Quiz

All mathematical expressions use raw `$...$` and `$$...$$` delimiters.

## Homework — Lecture 9

### Problem 1 — Pole-zero geometry
For

$$H(z)=\frac{1-0.8z^{-1}}{1-0.5z^{-1}},$$

(a) identify the zero and pole; (b) explain geometrically why $|H(e^{j\omega})|$ is the product of zero-vector lengths divided by the product of pole-vector lengths; (c) predict qualitatively whether the magnitude is larger near $\omega=0$ or $\omega=\pi$.

### Problem 2 — Phase delay and group delay
Let

$$\phi(\omega)=-3\omega+0.4\sin\omega.$$

Compute the phase delay

$$\tau_p(\omega)=-\frac{\phi(\omega)}{\omega}$$

for $\omega\neq 0$ and the group delay

$$\tau_g(\omega)=-\frac{d\phi(\omega)}{d\omega}.$$

Explain why the system is not linear phase.

### Problem 3 — First-order all-pass
For real $|a|<1$,

$$H_{ap}(z)=\frac{z^{-1}-a}{1-az^{-1}}.$$

Show that

$$|H_{ap}(e^{j\omega})|=1$$

for all $\omega$. Locate the pole and zero and explain the reciprocal relationship.

### Problem 4 — ML feature distortion
An ECG classifier uses QRS peak timing and onset slope as features. Explain how a causal IIR with nonlinear phase can alter those features even if its magnitude response removes only an irrelevant high-frequency noise band. Propose one offline and one streaming-safe mitigation.

## Homework — Lecture 10

### Problem 5 — Symmetry and linear phase
A real FIR filter has length $N=M+1$ and satisfies

$$h[n]=h[M-n].$$

Show that its frequency response can be written as

$$H(e^{j\omega})=e^{-j\omega M/2}A(\omega),$$

where $A(\omega)$ is real. State the group delay.

### Problem 6 — FIR Types I–IV
For each case below, identify the FIR type and state any forced zeros at $\omega=0$ or $\omega=\pi$:

1. $N=9$, symmetric.
2. $N=8$, symmetric.
3. $N=9$, antisymmetric.
4. $N=8$, antisymmetric.

Then explain why a Type II filter cannot realize a high-pass response with nonzero gain at $\omega=\pi$.

### Problem 7 — Four-zero constellation
A real linear-phase FIR has a zero at

$$z_0=0.7e^{j0.6}.$$

List the other zeros implied by real coefficients and linear-phase reciprocal symmetry.

### Problem 8 — Symmetry-constrained Conv1d
Describe a PyTorch parameterization that guarantees a trainable odd-length Conv1d kernel remains symmetric during training. Explain what inductive bias this imposes and what flexibility it removes.

---

# Weekly Quiz — Student Version

### Q1
For

$$H(z)=K\frac{\prod_i(z-z_i)}{\prod_k(z-p_k)},$$

what geometric quantities determine $|H(e^{j\omega})|$ when $z=e^{j\omega}$?

### Q2
State the definitions of phase delay and group delay.

### Q3
What property distinguishes an all-pass filter from an ordinary magnitude-shaping filter?

### Q4
Why can two filters with nearly identical magnitude responses produce different transient waveforms?

### Q5
A real FIR has $N=21$ symmetric taps. What is its group delay?

### Q6
Which FIR type is even-length and symmetric, and what forced endpoint zero does it have?

### Q7
A complex zero $z_0$ of a real linear-phase FIR is neither real nor on the unit circle. What other three zeros accompany it?

### Q8
Does a randomly initialized `nn.Conv1d` kernel automatically have linear phase? Briefly justify.

---

# Solutions

## Homework Solutions

### Solution 1
The zero is at $z=0.8$ and the pole is at $z=0.5$. Evaluating on the unit circle gives

$$H(e^{j\omega})=K\frac{\prod_i(e^{j\omega}-z_i)}{\prod_k(e^{j\omega}-p_k)}.$$

Therefore

$$|H(e^{j\omega})|=|K|\frac{\prod_i|e^{j\omega}-z_i|}{\prod_k|e^{j\omega}-p_k|}.$$

Each magnitude is a Euclidean vector length in the $z$-plane. Near $\omega=0$, the unit-circle point is $z=1$, which is closer to the zero at $0.8$ than to the pole at $0.5$, so the numerator is relatively small. Near $\omega=\pi$, the point is $z=-1$ and the distance ratio is less suppressed. Direct checks give

$$|H(1)|=\frac{0.2}{0.5}=0.4,$$

$$|H(-1)|=\frac{1.8}{1.5}=1.2.$$

Thus the magnitude is larger near $\omega=\pi$.

### Solution 2

$$\tau_p(\omega)=3-0.4\frac{\sin\omega}{\omega}.$$

Also,

$$\frac{d\phi}{d\omega}=-3+0.4\cos\omega,$$

so

$$\tau_g(\omega)=3-0.4\cos\omega.$$

Because $\tau_g(\omega)$ varies with $\omega$, the phase is not linear in frequency.

### Solution 3
On the unit circle,

$$H_{ap}(e^{j\omega})=\frac{e^{-j\omega}-a}{1-ae^{-j\omega}}.$$

The squared numerator magnitude is

$$|e^{-j\omega}-a|^2=1+a^2-2a\cos\omega,$$

and the squared denominator magnitude is

$$|1-ae^{-j\omega}|^2=1+a^2-2a\cos\omega.$$

Hence

$$|H_{ap}(e^{j\omega})|=1.$$

The pole is at $z=a$ and the zero is at $z=1/a$. For $|a|<1$, the pole is inside the unit circle and the reciprocal zero is outside.

### Solution 4
Nonlinear phase means frequency-dependent group delay, so different spectral components of the QRS transient can move by different amounts. That can change peak time, onset slope, and morphology even when the magnitude response removes only nominal noise. Offline mitigation: use a validated forward-backward zero-phase method while accounting for the squared magnitude response. Streaming-safe mitigation: use a linear-phase FIR or train/evaluate with the exact causal preprocessing that will be available at deployment.

### Solution 5
Pair symmetric samples about $M/2$. The frequency response can be factored as

$$H(e^{j\omega})=e^{-j\omega M/2}A(\omega),$$

where $A(\omega)$ is real. Therefore the phase is linear apart from possible $\pi$ jumps caused by sign changes in $A(\omega)$, and the group delay is

$$\tau_g=\frac{M}{2}=\frac{N-1}{2}.$$

### Solution 6
1. $N=9$, symmetric: Type I; no endpoint zero is forced by symmetry alone.
2. $N=8$, symmetric: Type II; forced zero at $\omega=\pi$, equivalently $z=-1$.
3. $N=9$, antisymmetric: Type III; forced zeros at $\omega=0$ and $\omega=\pi$, equivalently $z=1$ and $z=-1$.
4. $N=8$, antisymmetric: Type IV; forced zero at $\omega=0$, equivalently $z=1$.

A Type II response must satisfy $H(e^{j\pi})=0$, so it cannot have nonzero high-pass gain at Nyquist.

### Solution 7
Given

$$z_0=0.7e^{j0.6},$$

real coefficients imply the conjugate

$$z_0^*=0.7e^{-j0.6}.$$

Linear-phase reciprocal symmetry also implies

$$\frac{1}{z_0}=\frac{1}{0.7}e^{-j0.6}$$

and

$$\frac{1}{z_0^*}=\frac{1}{0.7}e^{j0.6}.$$

Thus the four-zero constellation is

$$z_0,\quad z_0^*,\quad \frac{1}{z_0},\quad \frac{1}{z_0^*}.$$

### Solution 8
For odd length $N=2R+1$, learn only $R+1$ independent parameters and construct

$$h=[a_0,a_1,\ldots,a_{R-1},a_R,a_{R-1},\ldots,a_1,a_0].$$

In PyTorch, concatenate the learned half, center tap, and a reversed copy. The resulting kernel is symmetric for every optimizer step, so it has exact generalized linear phase. This imposes a phase-preserving inductive bias and roughly halves the number of independent coefficients, but it prevents the model from learning arbitrary nonlinear-phase kernels.

## Weekly Quiz Solutions

### A1

$$|H(e^{j\omega})|=|K|\frac{\prod_i|e^{j\omega}-z_i|}{\prod_k|e^{j\omega}-p_k|}.$$

The relevant geometric quantities are the vector lengths from each zero and pole to the unit-circle point $e^{j\omega}$.

### A2

$$\tau_p(\omega)=-\frac{\angle H(e^{j\omega})}{\omega},$$

$$\tau_g(\omega)=-\frac{d}{d\omega}\angle H(e^{j\omega}).$$

### A3
An all-pass filter satisfies

$$|H(e^{j\omega})|=1$$

for all $\omega$ while changing phase and group delay.

### A4
Their phase responses can differ. Frequency-dependent group delay changes the relative arrival times of frequency components and can reshape a transient even when magnitudes are nearly the same.

### A5

$$\tau_g=\frac{21-1}{2}=10\text{ samples}.$$

### A6
Type II. It is even-length and symmetric and has a forced zero at

$$\omega=\pi,$$

or $z=-1$.

### A7

$$z_0^*,\quad \frac{1}{z_0},\quad \frac{1}{z_0^*}.$$

### A8
No. A generic `nn.Conv1d` learns unconstrained kernel coefficients, so symmetry or antisymmetry is not guaranteed. Exact linear phase must be imposed by parameterization, constraints, or a suitable regularizer/architecture.
