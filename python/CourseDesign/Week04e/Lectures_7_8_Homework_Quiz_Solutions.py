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
# # Lectures_7_8_Homework_Quiz_Solutions
#
# Lectures 7&8 Homework and Weekly Quiz: Instructor Version
#

# %% [markdown]
# ## Homework
#
# ### Problem 1: Ideal lowpass FIR and causal realization
# An ideal discrete-time lowpass filter has
#
# $$
# H_d(e^{j\omega})=
# \begin{cases}
# 1, & |\omega|\le \omega_c,\\
# 0, & \omega_c<|\omega|\le \pi.
# \end{cases}
# $$
#
# 1. Use the inverse DTFT to show that
#
# $$
# h_d[n]=\frac{\sin(\omega_c n)}{\pi n},\qquad n\ne 0,
# $$
#
# with
#
# $$
# h_d[0]=\frac{\omega_c}{\pi}.
# $$
#
# 2. Explain why this ideal impulse response cannot be implemented exactly as a real-time FIR filter.
# 3. For an odd-length filter with $N=41$ taps, state the shift required to make a symmetric truncated version causal and give its constant group delay.
#
# #### Solution
# The inverse DTFT gives
#
# $$
# h_d[n]=\frac{1}{2\pi}\int_{-\omega_c}^{\omega_c}e^{j\omega n}\,d\omega.
# $$
#
# For $n\ne 0$,
#
# $$
# h_d[n]
# =\frac{1}{2\pi}\left[\frac{e^{j\omega n}}{jn}\right]_{-\omega_c}^{\omega_c}
# =\frac{e^{j\omega_c n}-e^{-j\omega_c n}}{2\pi j n}
# =\frac{\sin(\omega_c n)}{\pi n}.
# $$
#
# For $n=0$,
#
# $$
# h_d[0]=\frac{1}{2\pi}\int_{-\omega_c}^{\omega_c}1\,d\omega
# =\frac{\omega_c}{\pi}.
# $$
#
# The response is infinite in duration and two-sided, so it is neither finite-length nor causal. For $N=41$, the symmetry center is
#
# $$
# M=\frac{N-1}{2}=20.
# $$
#
# Shift the truncated sequence right by $20$ samples. A symmetric Type-I FIR then has linear phase with constant group delay
#
# $$
# \tau_g=20\ \text{samples}.
# $$
#
# ---
#
# ### Problem 2: Window tradeoff
# A student designs two $51$-tap lowpass filters with the same cutoff. Filter A uses a rectangular window; Filter B uses a Hamming window.
#
# 1. Which filter should have the narrower transition band?
# 2. Which filter should have lower stopband sidelobes?
# 3. Give approximate first-sidelobe levels for the two windows.
# 4. Explain why merely increasing the rectangular-window length does not remove the Gibbs overshoot near a discontinuity.
#
# #### Solution
# The rectangular window has the narrowest main lobe for a given length, so Filter A typically has the narrower transition. The Hamming window suppresses sidelobes much more strongly, so Filter B has better stopband rejection.
#
# Typical first-sidelobe levels are approximately
#
# $$
# A_{\text{rect}}\approx -13\ \text{dB},
# \qquad
# A_{\text{Hamming}}\approx -43\ \text{dB}.
# $$
#
# Increasing the rectangular-window length narrows the oscillation region in frequency, but the peak overshoot near the discontinuity approaches a nonzero limiting value. This is the Gibbs phenomenon: the ripple becomes spatially narrower rather than disappearing in amplitude.
#
# ---
#
# ### Problem 3: Symmetry and linear phase
# Let a real FIR filter have length $N$ and satisfy
#
# $$
# h[n]=h[N-1-n].
# $$
#
# Show that its frequency response can be written in the form
#
# $$
# H(e^{j\omega})=e^{-j\omega(N-1)/2}A(\omega),
# $$
#
# where $A(\omega)$ is real. What is the group delay where $A(\omega)\ne 0$? Why can this be useful in a feature-extraction pipeline?
#
# #### Solution
# Pair symmetric terms about the midpoint. Factoring the common delay term gives
#
# $$
# H(e^{j\omega})
# =e^{-j\omega(N-1)/2}A(\omega),
# $$
#
# where the remaining paired cosine terms make $A(\omega)$ real. Therefore the phase is, apart from possible $\pi$-jumps caused by the sign of $A(\omega)$,
#
# $$
# \angle H(e^{j\omega})=-\omega\frac{N-1}{2}.
# $$
#
# Hence
#
# $$
# \tau_g(\omega)
# =-\frac{d}{d\omega}\angle H(e^{j\omega})
# =\frac{N-1}{2}.
# $$
#
# A constant group delay preserves relative temporal alignment among frequency components. In ML preprocessing this can be valuable when transient timing or waveform shape is itself an informative feature. It is useful, not universally mandatory: some models can tolerate or learn phase variation.
#
# ---
#
# ### Problem 4: FIR convolution in PyTorch
# A causal FIR filter has taps
#
# $$
# h=[0.2,\ 0.5,\ 0.3]
# $$
#
# and the input is
#
# $$
# x=[1,\ 2,\ 0,\ -1].
# $$
#
# 1. Compute the first four causal output samples using
#
# $$
# y[n]=\sum_{k=0}^{2}h[k]x[n-k],
# $$
#
# with $x[n]=0$ for $n<0$.
# 2. Explain why `torch.nn.functional.conv1d` requires the tap order to be reversed if you want textbook DSP convolution with left zero-padding.
#
# #### Solution
# The output is
#
# $$
# y[0]=0.2(1)=0.2,
# $$
#
# $$
# y[1]=0.2(2)+0.5(1)=0.9,
# $$
#
# $$
# y[2]=0.2(0)+0.5(2)+0.3(1)=1.3,
# $$
#
# $$
# y[3]=0.2(-1)+0.5(0)+0.3(2)=0.4.
# $$
#
# Thus
#
# $$
# \boxed{y=[0.2,\ 0.9,\ 1.3,\ 0.4]}.
# $$
#
# PyTorch `Conv1d` uses the deep-learning cross-correlation convention: it multiplies the kernel by samples in the same order as they appear in the local window. Textbook convolution uses $x[n-k]$, which reverses one sequence. Therefore a fixed DSP FIR implemented with `conv1d` normally uses a flipped tap vector together with appropriate left padding.
#
# ---
#
# ### Problem 5: Butterworth magnitude response
# For an $N=4$ analog Butterworth lowpass filter,
#
# $$
# |H_c(j\Omega)|^2=\frac{1}{1+(\Omega/\Omega_c)^{2N}}.
# $$
#
# Compute $|H_c(j\Omega)|$ in dB at:
#
# 1. $\Omega=\Omega_c$
# 2. $\Omega=2\Omega_c$
#
# #### Solution
# At cutoff,
#
# $$
# |H_c(j\Omega_c)|^2=\frac{1}{2},
# $$
#
# so
#
# $$
# |H_c(j\Omega_c)|=\frac{1}{\sqrt{2}}
# $$
#
# and
#
# $$
# 20\log_{10}|H_c(j\Omega_c)|\approx -3.01\ \text{dB}.
# $$
#
# At $\Omega=2\Omega_c$,
#
# $$
# |H_c(j2\Omega_c)|^2
# =\frac{1}{1+2^8}
# =\frac{1}{257}.
# $$
#
# Thus
#
# $$
# |H_c(j2\Omega_c)|=\frac{1}{\sqrt{257}}\approx 0.0624,
# $$
#
# or
#
# $$
# 20\log_{10}(0.0624)\approx -24.1\ \text{dB}.
# $$
#
# ---
#
# ### Problem 6: Stability under the bilinear transform
# The bilinear transform is
#
# $$
# z=\frac{1+sT/2}{1-sT/2}.
# $$
#
# For $T=0.1$ s and analog pole
#
# $$
# s_p=-2+j3,
# $$
#
# compute the mapped digital pole and verify that it lies inside the unit circle.
#
# #### Solution
# With $T/2=0.05$,
#
# $$
# z_p=\frac{1+0.05(-2+j3)}{1-0.05(-2+j3)}
# =\frac{0.9+j0.15}{1.1-j0.15}.
# $$
#
# Multiplying numerator and denominator by $1.1+j0.15$,
#
# $$
# z_p
# \approx 0.785+ j0.243.
# $$
#
# Therefore
#
# $$
# |z_p|\approx \sqrt{0.785^2+0.243^2}\approx 0.822<1.
# $$
#
# The stable analog pole in the left-half plane maps to a stable digital pole inside the unit circle.
#
# ---
#
# ### Problem 7: Prewarping
# A desired digital cutoff is
#
# $$
# \omega_c=0.4\pi\ \text{rad/sample}
# $$
#
# with $T=1$. Find the analog frequency to use when designing the prototype before applying the bilinear transform.
#
# #### Solution
# Use
#
# $$
# \Omega_c=\frac{2}{T}\tan\left(\frac{\omega_c}{2}\right).
# $$
#
# Therefore
#
# $$
# \Omega_c
# =2\tan(0.2\pi)
# \approx \boxed{1.4531\ \text{rad/s}}.
# $$
#
# Designing the analog prototype at this prewarped frequency causes the bilinear transform to place the corresponding digital cutoff at approximately $0.4\pi$.
#
# ---
#
# ### Problem 8: Forward-backward IIR filtering
# An offline preprocessing pipeline applies an IIR filter forward and then backward using a `filtfilt`-style operation.
#
# 1. If the one-pass frequency response is $H(e^{j\omega})$, what is the magnitude response of the two-pass operation?
# 2. What happens to phase?
# 3. Name two reasons this operation is inappropriate for a real-time deployment pipeline.
# 4. Give one ML-specific risk of using forward-backward filtering during dataset preparation.
#
# #### Solution
# Ignoring finite-record edge handling, the forward-backward magnitude becomes
#
# $$
# |H_{\text{fb}}(e^{j\omega})|
# =|H(e^{j\omega})|^2.
# $$
#
# The forward and reverse phase responses cancel, giving zero net phase distortion. In dB, the one-pass attenuation approximately doubles.
#
# It is not suitable for real-time causal operation because the backward pass requires future samples and the entire data block is generally needed. Edge initialization can also create boundary transients.
#
# An ML risk is train/deployment mismatch: a model trained on zero-phase, future-aware preprocessing may perform differently when deployed with a causal real-time filter. The preprocessing choice must match the intended inference setting.
#
# ---
#
# # Weekly Quiz: 10-12 Minutes
#
# ## Questions
#
# 1. A real FIR filter has $N=31$ symmetric taps. What is its constant group delay?
# 2. Which has better stopband sidelobe suppression for equal length: rectangular or Hamming window? Which usually has the narrower main lobe?
# 3. State one sentence explaining the Gibbs phenomenon in windowed FIR design.
# 4. In PyTorch, why do fixed FIR coefficients generally need to be reversed before using `Conv1d` to reproduce textbook convolution?
# 5. For a Butterworth lowpass, what is the magnitude at $\Omega=\Omega_c$ in dB?
# 6. State the bilinear-transform frequency relation between $\Omega$ and $\omega$.
# 7. A digital cutoff is $\omega_c=0.6\pi$. Should the analog prototype be designed directly at $\Omega_c=0.6\pi/T$? Explain briefly.
# 8. What is the main benefit and the main deployment limitation of forward-backward IIR filtering for ML preprocessing?
#
# ## Quiz Solutions
#
# 1. 
# $$
# \tau_g=\frac{N-1}{2}=15\ \text{samples}.
# $$
#
# 2. Hamming has much lower sidelobes; rectangular has the narrower main lobe for the same length.
#
# 3. Abrupt truncation of an ideal response creates oscillatory ripple near the frequency discontinuity; increasing length narrows the ripple region but does not drive the peak overshoot to zero.
#
# 4. `Conv1d` uses cross-correlation ordering, whereas textbook convolution reverses one sequence through the $x[n-k]$ indexing.
#
# 5. 
# $$
# |H_c(j\Omega_c)|=\frac{1}{\sqrt{2}},
# \qquad
# 20\log_{10}|H_c(j\Omega_c)|\approx -3.01\ \text{dB}.
# $$
#
# 6. 
# $$
# \boxed{\Omega=\frac{2}{T}\tan\left(\frac{\omega}{2}\right)}.
# $$
#
# 7. No. The bilinear transform warps frequency. Prewarp first:
#
# $$
# \Omega_c=\frac{2}{T}\tan\left(\frac{0.6\pi}{2}\right).
# $$
#
# 8. Benefit: zero net phase distortion in offline data. Limitation: it is noncausal/future-aware and therefore cannot be reproduced directly in a true streaming deployment.
#

# %%
