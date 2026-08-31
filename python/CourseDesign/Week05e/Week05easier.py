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
# # Week 5 easier
#
# **Week 5: Filter Properties.** Time and frequency domain responses. Understanding linear phase, generalized linear phase, and group delay.

# %% [markdown]
# ## Week 5: Filter Properties
#
# **Lecture 9: Time/Frequency Responses and Group Delay**
# *Oppenheim & Schafer: Chapter 5 (5.1–5.4)*
#
# * **0:00 – 0:15 | Frequency Response from Poles and Zeros:** Geometric evaluation of $\vert{}H(e^{j\omega})\vert{}$ and $\angle H(e^{j\omega})$ directly from the $z$-plane. Using vector lengths from zeros (numerator) and poles (denominator) to the unit circle.
# * **0:15 – 0:40 | Phase Distortion and Group Delay:**
# * Phase Delay: $\tau_p(\omega) = -\frac{\angle H(e^{j\omega})}{\omega}$.
# * Group Delay: $\tau_g(\omega) = -\frac{d}{d\omega} \angle H(e^{j\omega})$.
# * Demonstrating how non-linear phase causes frequency-dependent group delay, leading to time-domain waveform dispersion.
#
#
# * **0:40 – 1:00 | All-Pass Systems:** Systems where $\vert{}H(e^{j\omega})\vert{} = 1$ for all $\omega$. Exploring conjugate reciprocal pole-zero symmetry and their use in phase equalization to flatten overall group delay.
# * **1:00 – 1:15 | ML Context – Feature Distortion:** How phase distortion impacts downstream machine learning. If a neural network relies on peak timing or onset shape (e.g., ECG anomaly detection), IIR filter dispersion destroys the predictive features before the network even sees them.
#
# ---
#
# **Lecture 10: Linear Phase, Generalized Phase, and ML Intersections**
# *Oppenheim & Schafer: Chapter 5 (5.7) | Raschka: 1D Convolutions*
#
# * **0:00 – 0:25 | Exact and Generalized Linear Phase:** Proving why only Finite Impulse Response (FIR) filters achieve exact linear phase. Defining generalized linear phase $\angle H(e^{j\omega}) = \beta - \alpha\omega$.
# * **0:25 – 0:50 | Symmetry Constraints and the 4 FIR Types:**
# * Proving $h[n]$ must be symmetric ($h[n] = h[M-n]$) or anti-symmetric ($h[n] = -h[M-n]$).
# * Types I–IV (Even/Odd length, Symmetric/Anti-symmetric) and their inherent frequency constraints (e.g., Type II cannot be high-pass).
# * Zero locations: The 4-zero constellation ($r$, $1/r$, $r^*$, $1/r^*$).
#
#
# * **0:50 – 1:10 | DSP to ML Bridge – Convolutions as Filters:** Framing PyTorch `nn.Conv1d` as a trainable FIR filter. Discussing inductive bias: randomly initialized CNNs do not learn linear phase by default.
# * **1:10 – 1:15 | Transition to Multi-rate:** Setting up the necessity of exact linear phase (FIR) for anti-aliasing in the sample-rate conversions covered next week.
#
# ---------------------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well. For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# ChatGPT: (Medium)
#
# I’ll build these as a matched pair: Lecture 9 will move from pole-zero geometry to phase/group delay and all-pass equalization, while Lecture 10 will turn that phase discussion into the symmetry constraints behind the four linear-phase FIR types and then connect those constraints to trainable Conv1d kernels. I’ll keep every equation in the requested raw $...$ / $$...$$ form throughout the slides, homework, and quiz materials.
#
# I’m also tightening two technical nuances while building the decks: an all-pass section will distinguish “unit magnitude” from “trivial phase,” and the multirate transition will say that linear phase is often highly desirable for alignment but is not mathematically required for alias suppression itself. That keeps the DSP claims precise for a senior/graduate audience.
#
# I’ll build these as a matched pair: Lecture 9 will move from pole-zero geometry to phase/group delay and all-pass equalization, while Lecture 10 will turn that phase discussion into the symmetry constraints behind the four linear-phase FIR types and then connect those constraints to trainable `Conv1d` kernels. I’ll keep every equation in the requested raw `$...$` / `$$...$$` form throughout the slides, homework, and quiz materials.
#
#
# I’m also tightening two technical nuances while building the decks: an all-pass section will distinguish “unit magnitude” from “trivial phase,” and the multirate transition will say that linear phase is often highly desirable for alignment but is not mathematically required for alias suppression itself. That keeps the DSP claims precise for a senior/graduate audience.
#
#
# Created. I built Lectures 9–10 as a matched senior/graduate-level DSP/ML pair, with all mathematical expressions written using raw `$...$` and `$$...$$` delimiters as requested.
#
# ### Files
#
# * [Complete Lectures 9–10 instructor package](sandbox:/mnt/data/week5_lectures9_10/Week5_Lectures9_10_DSP_ML_Instructor_Package.zip)
# * [Lecture 9 — Microsoft PowerPoint version](sandbox:/mnt/data/week5_lectures9_10/Lecture9_Time_Frequency_Responses_Group_Delay_Microsoft_PowerPoint.pptx)
# * [Lecture 10 — Microsoft PowerPoint version](sandbox:/mnt/data/week5_lectures9_10/Lecture10_Linear_Phase_Generalized_Phase_ML_Microsoft_PowerPoint.pptx)
# * [Lecture 9 Python code](sandbox:/mnt/data/week5_lectures9_10/lecture9_time_frequency_group_delay.py)
# * [Lecture 10 Python code](sandbox:/mnt/data/week5_lectures9_10/lecture10_linear_phase_conv1d.py)
# * [Lecture 9 Jupyter notebook](sandbox:/mnt/data/week5_lectures9_10/notebooks/Lecture9_DSP_ML.ipynb)
# * [Lecture 9 executed notebook](sandbox:/mnt/data/week5_lectures9_10/notebooks/Lecture9_DSP_ML_executed.ipynb)
# * [Lecture 10 Jupyter notebook](sandbox:/mnt/data/week5_lectures9_10/notebooks/Lecture10_DSP_ML.ipynb)
# * [Lecture 10 executed notebook](sandbox:/mnt/data/week5_lectures9_10/notebooks/Lecture10_DSP_ML_executed.ipynb)
# * [Homework, weekly quiz, and complete solutions](sandbox:/mnt/data/week5_lectures9_10/Lectures9_10_Homework_Quiz_Solutions.md)
# * [README / teaching flow](sandbox:/mnt/data/week5_lectures9_10/README.txt)
#
# The Microsoft PowerPoint versions passed slide-overflow checks. Lecture 9 contains **24 slides** and Lecture 10 contains **31 slides**.
#
# ---
#

# %% [markdown]
# ### Lecture 9 highlights
#
# The pole-zero section develops the geometric interpretation
#
# $$|H(e^{j\omega})|=
# |K|
# \frac{\prod_i |e^{j\omega}-z_i|}
# {\prod_k |e^{j\omega}-p_k|}$$
#
# and the corresponding phase-angle sum, then uses actual pole-zero geometry plots.
#
# The phase section carefully distinguishes
#
# $$\tau_p(\omega)=
# -\frac{\angle H(e^{j\omega})}{\omega}$$
#
# from
#
# $$\tau_g(\omega)=
# -\frac{d}{d\omega}
# \angle H(e^{j\omega}).$$
#
# The code includes a transient-dispersion demonstration comparing a causal nonlinear-phase IIR with zero-phase offline filtering.
#
# The all-pass section uses
#
# $$H_{ap}(z) =
# \frac{z^{-1}-a^*}
# {1-az^{-1}}$$
#
# and demonstrates numerically that
#
# $$|H_{ap}(e^{j\omega})|=1$$
#
# while the group delay can vary substantially.
#
# The ML discussion uses ECG, audio-onset, and vibration examples to make the key point that phase distortion can destroy predictive timing structure **before the classifier ever sees the data**.
#
# ### Lecture 10 highlights
#
# The deck begins from generalized linear phase,
#
# $$\angle H(e^{j\omega}) =
# \beta-\alpha\omega,$$
#
# and develops the standard FIR factorization
#
# $$H(e^{j\omega}) =
# e^{-j\omega M/2}A(\omega).$$
#
# It derives symmetry and antisymmetry and then treats all four FIR types:
#
# * Type I: odd length, symmetric.
# * Type II: even length, symmetric, with a forced zero at $\omega=\pi$.
# * Type III: odd length, antisymmetric, with forced zeros at $\omega=0$ and $\omega=\pi$.
# * Type IV: even length, antisymmetric, with a forced zero at $\omega=0$.
#
# The pole-zero discussion includes the conjugate-reciprocal constellation
#
# $$z_0,\qquad
# z_0^*,\qquad
# \frac{1}{z_0},\qquad
# \frac{1}{z_0^*}.$$
#
# The PyTorch section shows that a generic `Conv1d` kernel does **not** automatically have linear phase, then demonstrates a symmetry-constrained parameterization in which only half of the taps are independently learned and the other half are generated by reflection.
#
# I also made the multirate transition precise: **anti-alias suppression requires adequate magnitude selectivity; exact linear phase is not mathematically required for alias suppression, although linear-phase FIR filters are often highly desirable when waveform timing and alignment must be preserved.**
#
# The homework contains eight problems across the two lectures, and the weekly quiz contains eight questions with complete worked solutions.
#
# ------------------------------
#

# %% [markdown]
# ## Week 5, Lecture 9
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT:
#
# Yes. Lecture 9 is really about turning pole-zero geometry into a physical timing interpretation.
#
# The underlying chain of ideas is
#
# $$
# \text{pole-zero locations}
# \rightarrow
# H(e^{j\omega})
# \rightarrow
# |H(e^{j\omega})|,\angle H(e^{j\omega})
# \rightarrow
# \tau_g(\omega)
# \rightarrow
# \text{waveform dispersion}
# \rightarrow
# \text{ML feature distortion}
# $$
#
# The deck is organized around exactly that progression: first read magnitude and phase directly from the $z$-plane, then interpret phase slope as delay, then use all-pass systems to modify phase without changing magnitude, and finally ask whether preprocessing preserves the timing information an ML model actually needs. 
#
# # Slides 3–5: Frequency response from pole-zero geometry
#
# The starting point is
#
# $$
# H(e^{j\omega}) =
# H(z)\big|_{z=e^{j\omega}}.
# $$
#
# That simply means that to obtain the frequency response of a discrete-time LTI system, evaluate its $z$-transform on the unit circle. 
#
# If
#
# $$
# H(z) =
# K
# \frac{\prod_i (z-z_i)}
# {\prod_k (z-p_k)},
# $$
#
# then on the unit circle,
#
# $$
# z=e^{j\omega},
# $$
#
# so
#
# $$
# H(e^{j\omega}) =
# K
# \frac{\prod_i (e^{j\omega}-z_i)}
# {\prod_k (e^{j\omega}-p_k)}.
# $$
#
# Taking magnitude gives
#
# $$
# |H(e^{j\omega})| =
# |K|
# \frac{\prod_i |e^{j\omega}-z_i|}
# {\prod_k |e^{j\omega}-p_k|}.
# $$
#
# That is the distance-ratio interpretation emphasized in the deck. 
#
# ## Physical interpretation
#
# For any chosen frequency $\omega$, mark the point
#
# $$
# e^{j\omega}
# $$
#
# on the unit circle.
#
# Now draw vectors from every zero and every pole to that point.
#
# Then:
#
# * zeros contribute numerator distances,
# * poles contribute denominator distances.
#
# So a zero near the unit circle at angle $\omega_0$ tends to suppress the response near $\omega_0$.
#
# A pole near the unit circle at angle $\omega_0$ tends to enhance the response near $\omega_0$.
#
# That gives a powerful qualitative rule:
#
# $$
# \boxed{
# \text{zero near }e^{j\omega_0}
# \Rightarrow
# \text{dip near }\omega_0
# }
# $$
#
# and
#
# $$
# \boxed{
# \text{pole near }e^{j\omega_0}
# \Rightarrow
# \text{peak near }\omega_0.
# }
# $$
#
# This is often more useful than immediately evaluating an algebraic expression.
#
# # A useful student question
#
# Suppose a pole moves radially from radius $0.5$ to radius $0.95$ while keeping the same angle $\omega_0$.
#
# What happens?
#
# A weak answer is:
#
# > The response increases.
#
# A good answer is:
#
# > Near $\omega_0$, the distance from the unit-circle point to the pole becomes much smaller, so the denominator decreases and the magnitude response becomes much larger.
#
# A strong answer adds:
#
# > The phase also changes more rapidly near $\omega_0$, so the pole contributes a more concentrated group-delay feature.
#
# That is exactly the graduate-level extension in the deck. 
#
# # Phase from geometry
#
# The phase follows from the same vectors.
#
# For
#
# $$
# H(e^{j\omega}) =
# K
# \frac{\prod_i (e^{j\omega}-z_i)}
# {\prod_k (e^{j\omega}-p_k)},
# $$
#
# the phase is
#
# $$
# \angle H(e^{j\omega}) =
# \angle K +
# \sum_i
# \angle(e^{j\omega}-z_i) -
# \sum_k
# \angle(e^{j\omega}-p_k).
# $$
#
# So:
#
# * every zero contributes a positive vector angle,
# * every pole contributes a negative vector angle.
#
# The deck emphasizes that phase can change rapidly even when the magnitude response looks fairly smooth. 
#
# That is important because students often focus entirely on magnitude plots and overlook phase.
#
# # Slide 6: Worked geometry example
#
# The deck uses
#
# $$
# H(z) =
# \frac{1-0.8z^{-1}}
# {1-0.5z^{-1}}.
# $$
#
# There is a zero at
#
# $$
# z=0.8
# $$
#
# and a pole at
#
# $$
# z=0.5.
# $$
#
# At
#
# $$
# \omega=0,
# $$
#
# the unit-circle point is
#
# $$
# z=1.
# $$
#
# Distance from the zero:
#
# $$
# |1-0.8| =
# 0.2.
# $$
#
# Distance from the pole:
#
# $$
# |1-0.5| =
# 0.5.
# $$
#
# Therefore,
#
# $$
# |H(1)| =
# \frac{0.2}{0.5} =
# \boxed{0.4}.
# $$
#
# At
#
# $$
# \omega=\pi,
# $$
#
# the unit-circle point is
#
# $$
# z=-1.
# $$
#
# Now the distances are
#
# $$
# |-1-0.8| =
# 1.8
# $$
#
# and
#
# $$
# |-1-0.5| =
# 1.5.
# $$
#
# Therefore,
#
# $$
# |H(-1)| =
# \frac{1.8}{1.5} =
# \boxed{1.2}.
# $$
#
# So before plotting anything, we already know:
#
# $$
# |H(e^{j0})|
# <
# |H(e^{j\pi})|.
# $$
#
# That suggests a high-frequency tilt.
#
# The deck uses exactly this geometric prediction. 
#
# # Good answer to “Why is this useful?”
#
# A good answer is:
#
# > Pole-zero geometry lets us predict peaks, notches, and qualitative spectral shape without evaluating the transfer function at hundreds of frequencies.
#
# A stronger answer is:
#
# > It also connects numerical filter behavior to structural parameters. Moving a pole or zero has an immediately interpretable geometric effect.
#
# # Slides 8–9: Phase delay versus group delay
#
# The deck deliberately distinguishes two quantities:
#
# $$
# \tau_p(\omega) =
# -\frac{\angle H(e^{j\omega})}{\omega}
# $$
#
# and
#
# $$
# \tau_g(\omega) =
# -\frac{d}{d\omega}
# \angle H(e^{j\omega}).
# $$
#
#
#
# These answer different questions.
#
# ## Phase delay
#
# Phase delay asks:
#
# > For a sinusoid at frequency $\omega$, what pure time delay would produce the same phase shift?
#
# If a system gives phase
#
# $$
# \phi(\omega),
# $$
#
# then a pure delay $n_0$ would produce
#
# $$
# \phi(\omega) =
# -\omega n_0.
# $$
#
# So
#
# $$
# n_0 =
# -\frac{\phi(\omega)}{\omega}.
# $$
#
# That is phase delay.
#
# # Group delay
#
# Group delay asks:
#
# > How does a narrowband wave packet or envelope propagate through the system?
#
# It depends on the local phase slope:
#
# $$
# \tau_g(\omega) =
# -\frac{d\phi(\omega)}{d\omega}.
# $$
#
# This is usually the more useful quantity when discussing transient dispersion.
#
# # When are they equal?
#
# Suppose
#
# $$
# \phi(\omega) =
# -\alpha\omega.
# $$
#
# Then
#
# $$
# \tau_p(\omega) =
# -\frac{-\alpha\omega}{\omega}
# =\alpha
# $$
#
# and
#
# $$
# \tau_g(\omega) =
# -\frac{d}{d\omega}(-\alpha\omega) =
# \alpha.
# $$
#
# Therefore,
#
# $$
# \boxed{
# \tau_p(\omega)=\tau_g(\omega)=\alpha
# }
# $$
#
# for pure linear phase through the origin.
#
# But for generalized linear phase
#
# $$
# \phi(\omega) =
# \beta-\alpha\omega,
# $$
#
# we get
#
# $$
# \tau_g(\omega) =
# \alpha,
# $$
#
# while
#
# $$
# \tau_p(\omega) =
# \alpha-\frac{\beta}{\omega}.
# $$
#
# So they are generally not equal.
#
# This is a good reason not to conflate phase delay and group delay.
#
# # Good answer to “Which one describes waveform dispersion?”
#
# $$
# \boxed{\text{Group delay}}
# $$
#
# because dispersion arises when different nearby frequency components experience different local delay.
#
# # Slide 9: Linear phase
#
# If
#
# $$
# \angle H(e^{j\omega}) =
# \beta-\alpha\omega,
# $$
#
# then
#
# $$
# \tau_g(\omega) =
# -\frac{d}{d\omega}
# \left(
# \beta-\alpha\omega
# \right) =
# \boxed{\alpha}.
# $$
#
# The group delay is constant.
#
# The deck notes an important subtlety: the constant $\beta$ changes phase offset but not the group-delay slope. 
#
# Another subtle issue is phase wrapping.
#
# If a real amplitude term changes sign, the displayed phase may jump by
#
# $$
# \pi.
# $$
#
# That jump does not necessarily mean a real physical dispersion event.
#
# Students should learn to distinguish:
#
# $$
# \text{phase wrapping/sign changes}
# $$
#
# from
#
# $$
# \text{true nonlinear phase curvature}.
# $$
#
# # Slides 10–12: Nonlinear phase and waveform dispersion
#
# Suppose
#
# $$
# \phi(\omega)
# $$
#
# is curved.
#
# Then
#
# $$
# \frac{d\phi}{d\omega}
# $$
#
# varies with frequency.
#
# Therefore,
#
# $$
# \tau_g(\omega)
# $$
#
# varies with frequency too.
#
# The deck makes the physical connection: a transient contains many frequencies, and if those components arrive at different times, the waveform cannot reassemble exactly. 
#
# The inverse DTFT reminds us why:
#
# $$
# x[n] =
# \frac{1}{2\pi}
# \int_{-\pi}^{\pi}
# X(e^{j\omega})e^{j\omega n}d\omega.
# $$
#
# A waveform is built from the coherent superposition of many frequency components.
#
# If filtering changes the relative phase relationships, the reconstructed waveform changes.
#
# Possible consequences include:
#
# $$
# \text{peak shift},
# $$
#
# $$
# \text{ringing},
# $$
#
# $$
# \text{smearing},
# $$
#
# and
#
# $$
# \text{altered onset slope}.
# $$
#
# # A simple two-tone example
#
# Suppose
#
# $$
# x[n] =
# \cos(0.2\pi n)
# +
# \cos(0.7\pi n).
# $$
#
# Imagine the low-frequency component is delayed by
#
# $$
# 3
# $$
#
# samples while the high-frequency component is delayed by
#
# $$
# 12
# $$
#
# samples.
#
# The output becomes approximately
#
# $$
# y[n] =
# \cos(0.2\pi(n-3))
# +
# \cos(0.7\pi(n-12)).
# $$
#
# Even if both frequency components have the same magnitudes as before, their relative timing has changed.
#
# So the waveform shape changes.
#
# This is the essence of phase distortion.
#
# # Good answer to “Can two filters have the same magnitude response and different outputs?”
#
# Yes.
#
# The deck has this exact concept check. 
#
# A good answer is:
#
# > Yes. If the phase responses differ, the spectral components recombine differently in time, so transients and waveform shape can differ even when the magnitude spectra are identical.
#
# # Slide 12: Causal IIR versus zero-phase filtering
#
# The deck compares a causal IIR with forward-backward filtering. 
#
# The causal IIR may:
#
# * delay the transient,
# * smear it,
# * alter its local shape.
#
# Forward-backward filtering can restore timing much better because its net phase is approximately zero.
#
# But this does not mean it is “free.”
#
# As discussed in Lecture 8,
#
# $$
# H_{\text{fb}}(e^{j\omega}) =
# |H(e^{j\omega})|^2.
# $$
#
# So the magnitude response changes too.
#
# And it is noncausal.
#
# A strong answer to
#
# > “Why not always use zero-phase filtering?”
#
# is:
#
# > Because it requires future samples and therefore is only available offline. It also changes the effective magnitude response and can create a mismatch with real-time deployment.
#
# # Slides 13–15: All-pass systems
#
# An all-pass system satisfies
#
# $$
# \boxed{
# |H_{ap}(e^{j\omega})| =
# 1
# \quad
# \forall\omega.
# }
# $$
#
# But that does not mean
#
# $$
# H_{ap}(e^{j\omega})=1.
# $$
#
# It can have nontrivial phase.
#
# That is the major conceptual point.
#
# The deck explicitly says that all-pass systems modify phase and group delay while leaving magnitude unchanged. 
#
# # First-order all-pass
#
# The deck uses
#
# $$
# H_{ap}(z) =
# \frac{z^{-1}-a^*}
# {1-az^{-1}},
# \qquad
# |a|<1.
# $$
#
# The pole is at
#
# $$
# z=a,
# $$
#
# inside the unit circle.
#
# The zero is at
#
# $$
# z=\frac{1}{a^*},
# $$
#
# outside the unit circle at the reciprocal-conjugate location. 
#
# # Why is the magnitude one?
#
# Evaluate on the unit circle:
#
# $$
# z=e^{j\omega}.
# $$
#
# For a real first-order parameter $a$, the familiar form is
#
# $$
# H_{ap}(z) =
# \frac{z^{-1}-a}
# {1-az^{-1}}.
# $$
#
# Then
#
# $$
# H_{ap}(e^{j\omega}) =
# \frac{e^{-j\omega}-a}
# {1-ae^{-j\omega}}.
# $$
#
# The squared numerator magnitude is
#
# $$
# |e^{-j\omega}-a|^2 =
# 1+a^2-2a\cos\omega.
# $$
#
# The squared denominator magnitude is
#
# $$
# |1-ae^{-j\omega}|^2 =
# 1+a^2-2a\cos\omega.
# $$
#
# Thus,
#
# $$
# |H_{ap}(e^{j\omega})|^2 =
# 1,
# $$
#
# so
#
# $$
# \boxed{
# |H_{ap}(e^{j\omega})|=1.
# }
# $$
#
# This is also the solution structure given in the homework appendix. 
#
# # Important conceptual point
#
# A magnitude-only plot would show:
#
# $$
# |H_{ap}(e^{j\omega})|=1.
# $$
#
# That could make someone think the filter does nothing.
#
# But its phase can vary strongly.
#
# Therefore:
#
# $$
# \boxed{
# \text{flat magnitude does not imply unchanged waveform}.
# }
# $$
#
# The deck makes exactly this point. 
#
# # Slide 16: Phase equalization
#
# Suppose an existing system has response
#
# $$
# H(z)
# $$
#
# with acceptable magnitude but undesirable group delay.
#
# Cascade an all-pass:
#
# $$
# H_{\text{tot}}(z) =
# H(z)H_{ap}(z).
# $$
#
# Magnitude becomes
#
# $$
# |H_{\text{tot}}(e^{j\omega})| =
# |H(e^{j\omega})|
# |H_{ap}(e^{j\omega})|.
# $$
#
# Since
#
# $$
# |H_{ap}(e^{j\omega})| =
# 1,
# $$
#
# we obtain
#
# $$
# \boxed{
# |H_{\text{tot}}(e^{j\omega})| =
# |H(e^{j\omega})|.
# }
# $$
#
# The group delays add:
#
# $$
# \boxed{
# \tau_{g,\text{tot}}(\omega) =
# \tau_{g,H}(\omega)
# +
# \tau_{g,ap}(\omega).
# }
# $$
#
# So the all-pass can be selected to flatten the total group delay over a desired band. 
#
# # Very important limitation
#
# An all-pass cannot fix a bad magnitude response.
#
# It cannot repair:
#
# $$
# \text{a deep magnitude notch},
# $$
#
# $$
# \text{poor stopband attenuation},
# $$
#
# or
#
# $$
# \text{insufficient bandwidth}.
# $$
#
# Its magnitude is fixed at one.
#
# The deck also makes the deeper point that a stable causal all-pass generally contributes nonnegative group delay. So equalization usually means **adding delay where the original system was too fast**, not subtracting delay where it was too slow. 
#
# That is a good graduate-level point.
#
# # Good answer to “Can an all-pass make the system faster at one frequency?”
#
# Generally not in the causal stable sense students might imagine.
#
# A better answer is:
#
# > A causal stable all-pass is normally used to add frequency-dependent delay so that frequencies with too little original delay are slowed down to match the more delayed frequencies.
#
# # Slides 18–19: ML feature distortion
#
# The machine-learning connection is very strong here.
#
# The deck gives several examples: ECG, audio event detection, and vibration signals. 
#
# The important idea is:
#
# $$
# \boxed{
# \text{the classifier cannot recover information that preprocessing has already destroyed}.
# }
# $$
#
# # ECG example
#
# An ECG QRS complex contains useful information in:
#
# $$
# \text{peak timing},
# $$
#
# $$
# \text{peak width},
# $$
#
# $$
# \text{slope},
# $$
#
# and perhaps
#
# $$
# \text{relative timing between waves}.
# $$
#
# If an IIR preprocessing filter introduces frequency-dependent group delay, the QRS morphology can change.
#
# The classifier is not seeing the original event anymore.
#
# It is seeing
#
# $$
# \text{filtered morphology}.
# $$
#
# That may be fine if training and deployment always use exactly the same preprocessing.
#
# But it is dangerous if preprocessing changes later.
#
# # Audio onset example
#
# Suppose a classifier distinguishes:
#
# * kick drum,
# * snare,
# * cymbal.
#
# Onset shape can be highly informative.
#
# An IIR can redistribute timing among frequencies.
#
# Then features such as:
#
# $$
# \text{spectral flux},
# $$
#
# $$
# \text{attack time},
# $$
#
# or local spectrogram structure may shift.
#
# # Vibration example
#
# Mechanical faults may create periodic impulsive events.
#
# Useful features can include:
#
# $$
# \text{impulse spacing},
# $$
#
# $$
# \text{envelope timing},
# $$
#
# and
#
# $$
# \text{ring-down structure}.
# $$
#
# Phase distortion can modify those patterns.
#
# # Good answer to “Does nonlinear phase always hurt ML?”
#
# No.
#
# This is an important nuance.
#
# A weak answer is:
#
# > Yes, because phase distortion is bad.
#
# A strong answer is:
#
# > It depends on which information the task uses. If classification depends mainly on long-term spectral magnitude statistics, nonlinear phase may have little effect. If timing or morphology is predictive, phase distortion can be important.
#
# That is exactly the engineering question posed in the deck. 
#
# # A useful decision framework
#
# Before preprocessing, ask:
#
# > What information should remain invariant?
#
# If the task mainly needs
#
# $$
# |X(e^{j\omega})|,
# $$
#
# then phase distortion may be tolerable.
#
# If the task needs
#
# $$
# \text{relative timing},
# $$
#
# $$
# \text{waveform shape},
# $$
#
# or
#
# $$
# \text{transient morphology},
# $$
#
# then group-delay distortion deserves much more attention.
#
# # Slide 20: Python response analysis
#
# The deck uses:
#
# ```python
# b = np.array([1.0, -0.8])
# a = np.array([1.0, -0.5])
#
# w, H = signal.freqz(b, a, worN=2048)
# phase = np.unwrap(np.angle(H))
# w_gd, gd = signal.group_delay((b, a), w=2048)
# ```
#
#
#
# Each line corresponds to a mathematical object.
#
# `freqz` computes samples of
#
# $$
# H(e^{j\omega}).
# $$
#
# Then
#
# ```python
# np.abs(H)
# ```
#
# gives
#
# $$
# |H(e^{j\omega})|.
# $$
#
# And
#
# ```python
# np.angle(H)
# ```
#
# gives wrapped phase.
#
# `np.unwrap(...)` removes artificial $2\pi$ jumps so that differentiation is meaningful.
#
# Finally,
#
# ```python
# signal.group_delay(...)
# ```
#
# computes
#
# $$
# \tau_g(\omega).
# $$
#
# # Why unwrap phase before differentiating?
#
# Suppose displayed phase jumps from approximately
#
# $$
# +\pi
# $$
#
# to
#
# $$
# -\pi.
# $$
#
# That is often just a representation jump of
#
# $$
# 2\pi,
# $$
#
# not a real rapid physical phase change.
#
# Differentiating the wrapped phase directly would produce huge artificial spikes.
#
# Therefore, use a continuous unwrapped phase before computing
#
# $$
# -\frac{d\phi}{d\omega}.
# $$
#
# This is a good practical question for students.
#
# # Homework 1: Pole-zero geometry
#
# For
#
# $$
# H(z) =
# \frac{1-0.8z^{-1}}
# {1-0.5z^{-1}},
# $$
#
# the expected geometric checkpoints are:
#
# At
#
# $$
# \omega=0,
# $$
#
# $$
# |H(1)|=0.4.
# $$
#
# At
#
# $$
# \omega=\pi,
# $$
#
# $$
# |H(-1)|=1.2.
# $$
#
# A strong explanation:
#
# > The zero at $0.8$ is closer to $z=1$ than the pole at $0.5$, suppressing DC. At $z=-1$, their distance ratio instead exceeds one, so the high-frequency response is larger.
#
# The solution sketch confirms these values. 
#
# # Homework 2: Phase delay and group delay
#
# The deck gives
#
# $$
# \phi(\omega) =
# -3\omega
# +
# 0.4\sin\omega.
# $$
#
# Phase delay is
#
# $$
# \tau_p(\omega) =
# -\frac{\phi(\omega)}{\omega}.
# $$
#
# Therefore,
#
# $$
# \tau_p(\omega) =
# -\frac{-3\omega+0.4\sin\omega}{\omega}
# $$
#
# so
#
# $$
# \boxed{
# \tau_p(\omega) =
# 3 -
# 0.4\frac{\sin\omega}{\omega}.
# }
# $$
#
# Group delay is
#
# $$
# \tau_g(\omega) =
# -\frac{d\phi}{d\omega}.
# $$
#
# Since
#
# $$
# \frac{d\phi}{d\omega} =
# -3
# +
# 0.4\cos\omega,
# $$
#
# we obtain
#
# $$
# \boxed{
# \tau_g(\omega) =
# 3 -
# 0.4\cos\omega.
# }
# $$
#
# This is exactly the solution in the deck. 
#
# # Interpretation of the result
#
# The nominal delay is around
#
# $$
# 3
# $$
#
# samples.
#
# But group delay varies between
#
# $$
# 3-0.4 =
# 2.6
# $$
#
# and
#
# $$
# 3+0.4 =
# 3.4.
# $$
#
# So different frequency regions can experience delays differing by approximately
#
# $$
# 0.8
# $$
#
# samples.
#
# That is dispersion.
#
# # Homework 3: All-pass proof
#
# For real $a$,
#
# $$
# H_{ap}(z) =
# \frac{z^{-1}-a}
# {1-az^{-1}}.
# $$
#
# On the unit circle,
#
# $$
# z=e^{j\omega}.
# $$
#
# Then numerator magnitude squared is
#
# $$
# |e^{-j\omega}-a|^2 =
# 1+a^2-2a\cos\omega.
# $$
#
# Denominator magnitude squared is
#
# $$
# |1-ae^{-j\omega}|^2 =
# 1+a^2-2a\cos\omega.
# $$
#
# Therefore,
#
# $$
# \boxed{
# |H_{ap}(e^{j\omega})|=1.
# }
# $$
#
# A full-credit answer should explicitly show that numerator and denominator magnitudes match, not simply say “because it is all-pass.”
#
# # Homework 4: ECG mitigation
#
# The deck asks students to explain nonlinear phase effects and propose mitigations. 
#
# A good answer is:
#
# > Nonlinear phase can move the frequency components of a QRS complex by different delays, changing peak width, slope, or timing. For offline analysis, validated zero-phase filtering could preserve timing better. For streaming, use a linear-phase FIR or train and deploy with exactly the same causal preprocessing.
#
# A stronger answer adds:
#
# > If the classifier is trained on causally filtered ECGs, the distortion itself can become part of the learned data distribution. The real danger is inconsistent preprocessing.
#
# # Good concept-check answers
#
# ## Two filters have identical magnitude responses. Are their outputs always identical?
#
# $$
# \boxed{\text{No}}
# $$
#
# because
#
# $$
# Y_1(e^{j\omega}) =
# H_1(e^{j\omega})X(e^{j\omega})
# $$
#
# and
#
# $$
# Y_2(e^{j\omega}) =
# H_2(e^{j\omega})X(e^{j\omega}).
# $$
#
# Even if
#
# $$
# |H_1(e^{j\omega})| =
# |H_2(e^{j\omega})|,
# $$
#
# different phases imply
#
# $$
# \angle Y_1(e^{j\omega})
# \neq
# \angle Y_2(e^{j\omega}),
# $$
#
# so the time-domain waveforms can differ.
#
# The deck states precisely this. 
#
# # Good exit-ticket questions
#
# If I were selecting three from this lecture, I would use these.
#
# ## Exit ticket 1
#
# **Question**
#
# A pole moves closer to the unit circle at angle $\omega_0$. What happens near $\omega_0$ to magnitude and group delay?
#
# **Good answer**
#
# > The pole-to-unit-circle distance decreases, so the magnitude response increases near $\omega_0$. The phase also changes more rapidly there, producing a larger or more concentrated group-delay contribution.
#
# ---
#
# ## Exit ticket 2
#
# **Question**
#
# What is the conceptual difference between phase delay and group delay?
#
# **Good answer**
#
# > Phase delay interprets the phase of one sinusoidal component as an equivalent time shift,
#
# $$
# \tau_p(\omega) =
# -\frac{\phi(\omega)}{\omega},
# $$
#
# while group delay is the local phase slope,
#
# $$
# \tau_g(\omega) =
# -\frac{d\phi}{d\omega},
# $$
#
# and describes delay of narrowband envelopes or packets.
#
# ---
#
# ## Exit ticket 3
#
# **Question**
#
# Why can an all-pass filter alter an ML feature even though
#
# $$
# |H_{ap}(e^{j\omega})|=1?
# $$
#
# **Good answer**
#
# > Because an all-pass filter can strongly alter phase and group delay. If an ML feature depends on timing, onset shape, or waveform morphology, those features can change even though the magnitude spectrum is unchanged.
#
# That question ties together almost the entire lecture.
#
# # What I would expect a senior student to know
#
# A strong senior should be able to:
#
# derive magnitude qualitatively from pole-zero distances,
#
# explain the phase angle contribution of poles and zeros,
#
# distinguish
#
# $$
# \tau_p(\omega)
# $$
#
# from
#
# $$
# \tau_g(\omega),
# $$
#
# recognize that nonlinear phase creates dispersion,
#
# and explain why an all-pass system can have flat magnitude but nontrivial phase.
#
# # What I would expect a first-year graduate student to know
#
# In addition, I would expect them to:
#
# reason about how pole radius controls group-delay concentration,
#
# derive the all-pass unit-magnitude condition,
#
# explain reciprocal-conjugate pole-zero structure,
#
# understand why causal all-pass equalization generally adds delay rather than removing it,
#
# and evaluate whether a given ML preprocessing pipeline preserves the temporal statistics relevant to the task.
#
# # The most important takeaway
#
# The deepest statement in Lecture 9 is probably this:
#
# $$
# \boxed{
# |H(e^{j\omega})|
# \text{ tells you how much of each frequency remains;}
# \quad
# \angle H(e^{j\omega})
# \text{ tells you how those frequencies line up in time.}
# }
# $$
#
# A system can preserve the first and significantly alter the second.
#
# That is why magnitude-only reasoning is incomplete whenever waveform morphology or timing matters, and it is exactly why the lecture ends by connecting group delay to ECG peaks, audio onsets, vibration impulses, and ML preprocessing choices. 
#

# %% [markdown]
# ## Week 5, Lecture 10
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT:
#
# Yes. Lecture 10 is really the structural payoff of the phase discussion from Lecture 9.
#
# The main chain of ideas is
#
# $$
# \text{desired constant delay}
# \rightarrow
# \text{generalized linear phase}
# \rightarrow
# \text{symmetry or antisymmetry}
# \rightarrow
# \text{four FIR types}
# \rightarrow
# \text{forced zeros}
# \rightarrow
# \text{phase-aware Conv1d design}
# $$
#
# The deck is organized around exactly that progression: derive generalized linear phase, classify FIR Types I–IV, interpret the corresponding zero symmetries, then ask how those same constraints can be imposed on trainable 1-D convolution kernels. 
#
# # Slides 3–4: What generalized linear phase really means
#
# The lecture starts from
#
# $$\angle H(e^{j\omega}) =
# \beta-\alpha\omega.
# $$
#
# Equivalently,
#
# $$
# H(e^{j\omega}) =
# e^{-j\alpha\omega}
# e^{j\beta}
# A(\omega),
# $$
#
# where $A(\omega)$ is real for the standard real-coefficient FIR linear-phase cases. 
#
# The important physical quantity is the group delay:
#
# $$
# \tau_g(\omega) =
# -\frac{d}{d\omega}
# \angle H(e^{j\omega}).
# $$
#
# Substituting,
#
# $$
# \tau_g(\omega) =
# -\frac{d}{d\omega}
# (\beta-\alpha\omega) =
# \boxed{\alpha}.
# $$
#
# So generalized linear phase means:
#
# $$
# \boxed{
# \text{constant group delay}
# }
# $$
#
# even if the phase contains a constant offset $\beta$.
#
# ## Why is the word “generalized” useful?
#
# Because a real amplitude factor $A(\omega)$ can change sign.
#
# If
#
# $$
# A(\omega)<0,
# $$
#
# then the displayed phase acquires an additional
#
# $$
# \pi
# $$
#
# shift.
#
# So the plotted phase may look like it has jumps even though the underlying delay is still constant.
#
# A good answer to
#
# > “Why can a linear-phase FIR plot have apparent phase jumps?”
#
# is:
#
# > Because the real amplitude factor can cross zero and change sign. That sign change adds a $\pi$ phase jump without changing the underlying slope $-\alpha$.
#
# That is exactly the nuance emphasized in the deck. 
#
# # Slide 4: Why exact linear phase points to FIR structure
#
# The deck states that stable causal rational IIR filters generally have pole phase contributions that vary with frequency, making exact constant group delay difficult except in trivial scale/delay cases. 
#
# The intuitive reason is simple.
#
# A pole contributes phase like
#
# $$
# -\angle(e^{j\omega}-p_k),
# $$
#
# and that angle is nonlinear in $\omega$.
#
# So a generic pole produces curved phase.
#
# Exact linear phase would require these frequency-dependent pole effects to cancel exactly.
#
# In a stable causal rational setting, doing that nontrivially is not generally compatible with keeping a genuinely recursive filter structure.
#
# That is why practical exact linear-phase filters are FIR.
#
# A good student answer is:
#
# > “Exact linear phase requires constant group delay. Generic IIR poles create frequency-dependent phase slopes, whereas FIR symmetry gives a pure delay factor directly.”
#
# A stronger answer adds:
#
# > “The statement should be understood in the usual stable causal rational filter-design setting; trivial pure delays are FIR anyway.”
#
# # Slides 5–7: Why symmetry gives linear phase
#
# Suppose an FIR has length
#
# $$
# N=M+1
# $$
#
# and symmetric coefficients:
#
# $$
# h[n]
# ====
#
# h[M-n].
# $$
#
# Its frequency response is
#
# $$
# H(e^{j\omega})
# ==============
#
# \sum_{n=0}^{M}
# h[n]e^{-j\omega n}.
# $$
#
# Pair the term at $n$ with the term at $M-n$.
#
# For one pair,
#
# $$
# h[n]e^{-j\omega n}
# +
# h[M-n]e^{-j\omega(M-n)}.
# $$
#
# Using symmetry,
#
# $$
# h[M-n]=h[n],
# $$
#
# so
#
# $$
# h[n]
# \left(
# e^{-j\omega n}
# +
# e^{-j\omega(M-n)}
# \right).
# $$
#
# Factor out the midpoint delay:
#
# # $$
#
# h[n]
# e^{-j\omega M/2}
# \left(
# e^{j\omega(M/2-n)}
# +
# e^{-j\omega(M/2-n)}
# \right).
# $$
#
# Using
#
# $$
# e^{j\theta}+e^{-j\theta}
# ========================
#
# 2\cos\theta,
# $$
#
# we obtain
#
# # $$
#
# 2h[n]
# e^{-j\omega M/2}
# \cos
# \left(
# \omega\left(\frac{M}{2}-n\right)
# \right).
# $$
#
# The cosine term is real.
#
# Therefore the entire response can be written as
#
# $$
# \boxed{
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}
# A(\omega)
# }
# $$
#
# with $A(\omega)$ real.
#
# Hence,
#
# $$
# \boxed{
# \tau_g
# ======
#
# # \frac{M}{2}
#
# \frac{N-1}{2}.
# }
# $$
#
# That is the key derivation in the deck. 
#
# # What is the intuition?
#
# The symmetry centers the impulse response at
#
# $$
# n=\frac{M}{2}.
# $$
#
# So the entire filter behaves like:
#
# $$
# \text{zero-phase amplitude shaping}
# +
# \text{a pure delay of } \frac{M}{2}.
# $$
#
# That is the cleanest physical interpretation.
#
# # Antisymmetric case
#
# Now suppose
#
# $$
# h[n]
# ====
#
# -h[M-n].
# $$
#
# Pairing gives
#
# $$
# h[n]e^{-j\omega n}
# ------------------
#
# h[n]e^{-j\omega(M-n)}.
# $$
#
# Factor again:
#
# # $$
#
# h[n]
# e^{-j\omega M/2}
# \left(
# e^{j\theta}
# -----------
#
# e^{-j\theta}
# \right),
# $$
#
# where
#
# $$
# \theta
# ======
#
# \omega
# \left(
# \frac{M}{2}-n
# \right).
# $$
#
# Using
#
# $$
# e^{j\theta}-e^{-j\theta}
# ========================
#
# 2j\sin\theta,
# $$
#
# we get a sine-weighted real amplitude plus a fixed quadrature factor.
#
# So antisymmetric FIRs also have generalized linear phase. 
#
# # Slides 8–12: Why there are exactly four FIR types
#
# The four types come from two binary choices:
#
# 1. symmetric or antisymmetric,
# 2. odd length or even length.
#
# The deck summarizes them as: 
#
# $$
# \boxed{
# \text{Type I: odd }N,\text{ symmetric}
# }
# $$
#
# $$
# \boxed{
# \text{Type II: even }N,\text{ symmetric}
# }
# $$
#
# $$
# \boxed{
# \text{Type III: odd }N,\text{ antisymmetric}
# }
# $$
#
# $$
# \boxed{
# \text{Type IV: even }N,\text{ antisymmetric}
# }
# $$
#
# The parity matters because it determines whether the midpoint is an actual sample or lies between samples, and that in turn forces different endpoint zeros.
#
# # Type I: odd length, symmetric
#
# Conditions:
#
# $$
# N\text{ odd},
# $$
#
# $$
# h[n]=h[M-n].
# $$
#
# Then
#
# $$
# \tau_g
# ======
#
# \frac{N-1}{2}
# $$
#
# is an integer.
#
# No zero at
#
# $$
# z=1
# $$
#
# or
#
# $$
# z=-1
# $$
#
# is forced purely by symmetry.
#
# So Type I is the most general of the four.
#
# It can realize:
#
# $$
# \text{low-pass},
# $$
#
# $$
# \text{high-pass},
# $$
#
# $$
# \text{band-pass},
# $$
#
# $$
# \text{band-stop},
# $$
#
# and many other real linear-phase responses. 
#
# # Type II: even length, symmetric
#
# Conditions:
#
# $$
# N\text{ even},
# $$
#
# $$
# h[n]=h[M-n].
# $$
#
# This structure forces
#
# $$
# H(e^{j\pi})=0.
# $$
#
# Equivalently,
#
# $$
# z=-1
# $$
#
# is a zero.
#
# So a conventional high-pass filter that needs nonzero gain at Nyquist cannot be Type II. 
#
# ## Why is $H(e^{j\pi})=0$ forced?
#
# At
#
# $$
# \omega=\pi,
# $$
#
# the exponential is
#
# $$
# e^{-j\pi n}=(-1)^n.
# $$
#
# For an even-length symmetric sequence, the paired terms cancel at Nyquist.
#
# That is not a tuning issue.
#
# It is structural.
#
# A strong answer to
#
# > “Could I just increase the order and make a Type II high-pass?”
#
# is:
#
# > No. The zero at $\omega=\pi$ is forced by the symmetry and parity, so no order increase removes that constraint.
#
# The deck calls this out explicitly. 
#
# # Type III: odd length, antisymmetric
#
# Conditions:
#
# $$
# N\text{ odd},
# $$
#
# $$
# h[n]=-h[M-n].
# $$
#
# This forces
#
# $$
# H(e^{j0})=0
# $$
#
# and
#
# $$
# H(e^{j\pi})=0.
# $$
#
# So there are forced zeros at
#
# $$
# z=1
# $$
#
# and
#
# $$
# z=-1.
# $$
#
# This means Type III cannot have nonzero gain at DC or Nyquist. 
#
# That makes it useful for responses where zero DC is natural, such as differentiator-like structures.
#
# # Why does antisymmetry force zero DC?
#
# At DC,
#
# $$
# H(e^{j0})
# =========
#
# \sum_n h[n].
# $$
#
# For antisymmetric pairs,
#
# $$
# h[n]+h[M-n]=0.
# $$
#
# So the sum vanishes:
#
# $$
# \boxed{
# H(e^{j0})=0.
# }
# $$
#
# That is a nice quick derivation students should be able to give.
#
# # Type IV: even length, antisymmetric
#
# Conditions:
#
# $$
# N\text{ even},
# $$
#
# $$
# h[n]=-h[M-n].
# $$
#
# This forces
#
# $$
# H(e^{j0})=0,
# $$
#
# but not necessarily
#
# $$
# H(e^{j\pi})=0.
# $$
#
# Its group delay is half-integer. 
#
# This makes Type IV especially useful for differentiator and Hilbert-transform designs.
#
# # A compact table students should know
#
# | Type | Length | Symmetry      | Forced zero at $0$? | Forced zero at $\pi$? |
# | ---- | ------ | ------------- | ------------------: | --------------------: |
# | I    | odd    | symmetric     |                  no |                    no |
# | II   | even   | symmetric     |                  no |                   yes |
# | III  | odd    | antisymmetric |                 yes |                   yes |
# | IV   | even   | antisymmetric |                 yes |                    no |
#
# The most important skill is not memorizing the table blindly.
#
# Students should understand why the endpoint zeros arise.
#
# # Good classification examples
#
# ### Example 1
#
# $$
# h=[1,2,3,2,1].
# $$
#
# Length:
#
# $$
# N=5
# $$
#
# which is odd.
#
# Symmetric.
#
# Therefore:
#
# $$
# \boxed{\text{Type I}}.
# $$
#
# Group delay:
#
# $$
# \tau_g
# ======
#
# # \frac{5-1}{2}
#
# 2.
#
# $$
#
# ### Example 2
#
# $$
# h=[1,2,2,1].
# $$
#
# Length:
#
# $$
# N=4
# $$
#
# which is even.
#
# Symmetric.
#
# Therefore:
#
# $$
# \boxed{\text{Type II}}.
# $$
#
# And necessarily:
#
# $$
# H(e^{j\pi})=0.
# $$
#
# ### Example 3
#
# $$
# h=[1,2,0,-2,-1].
# $$
#
# Odd length and antisymmetric.
#
# Therefore:
#
# $$
# \boxed{\text{Type III}}.
# $$
#
# So:
#
# $$
# H(e^{j0})=0
# $$
#
# and
#
# $$
# H(e^{j\pi})=0.
# $$
#
# ### Example 4
#
# $$
# h=[1,2,-2,-1].
# $$
#
# Even length and antisymmetric.
#
# Therefore:
#
# $$
# \boxed{\text{Type IV}}.
# $$
#
# So:
#
# $$
# H(e^{j0})=0,
# $$
#
# but Nyquist need not be zero.
#
# # Slides 14–15: Why zeros appear in groups of four
#
# For a real linear-phase FIR, a generic complex zero off the unit circle appears in the constellation
#
# $$
# \boxed{
# z_0,\quad
# z_0^*,\quad
# \frac{1}{z_0},\quad
# \frac{1}{z_0^*}.
# }
# $$
#
# The deck explains the two symmetries behind this. 
#
# Real coefficients imply conjugate symmetry:
#
# $$
# z_0
# \Rightarrow
# z_0^*.
# $$
#
# Linear-phase palindromic or anti-palindromic structure implies reciprocal symmetry:
#
# $$
# z_0
# \Rightarrow
# \frac{1}{z_0}.
# $$
#
# Combining them gives all four.
#
# # Example zero constellation
#
# Suppose
#
# $$
# z_0
# ===
#
# 0.7e^{j0.6}.
# $$
#
# Then the conjugate is
#
# $$
# z_0^*
# =====
#
# 0.7e^{-j0.6}.
# $$
#
# The reciprocal is
#
# $$
# \frac{1}{z_0}
# =============
#
# \frac{1}{0.7}
# e^{-j0.6}
# \approx
# 1.4286e^{-j0.6}.
# $$
#
# And the reciprocal conjugate is
#
# $$
# \frac{1}{z_0^*}
# ===============
#
# 1.4286e^{j0.6}.
# $$
#
# So the four points are
#
# $$
# \boxed{
# 0.7e^{j0.6},
# \quad
# 0.7e^{-j0.6},
# \quad
# 1.4286e^{-j0.6},
# \quad
# 1.4286e^{j0.6}.
# }
# $$
#
# This is exactly the kind of thing students should be comfortable sketching in the $z$-plane.
#
# # Degenerate cases
#
# The deck notes that the four-point set can collapse. 
#
# For example:
#
# * if the zero is real, conjugation does nothing,
# * if the zero lies on the unit circle, reciprocal magnitude is the same,
# * if the zero is at $z=1$ or $z=-1$, several symmetry operations produce the same point.
#
# So not every zero literally appears as four distinct visible zeros.
#
# # Polynomial viewpoint
#
# The lecture also states
#
# $$
# H(z)
# ====
#
# \pm z^{-M}H(z^{-1}).
# $$
#
# For symmetric FIRs, the coefficient polynomial is palindromic.
#
# For antisymmetric FIRs, it is anti-palindromic. 
#
# This is the algebraic reason reciprocal zeros appear.
#
# A good graduate-level answer to
#
# > “Why does linear phase imply reciprocal zero symmetry?”
#
# is:
#
# > Because the symmetry or antisymmetry of the impulse response makes the transfer polynomial palindromic or anti-palindromic, so replacing $z$ by $1/z$ reproduces the same polynomial up to scale and delay.
#
# # Slide 16: Apparent phase jumps
#
# Suppose
#
# $$
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}A(\omega),
# $$
#
# with $A(\omega)$ real.
#
# If
#
# $$
# A(\omega)>0,
# $$
#
# then phase is approximately
#
# $$
# -\omega M/2.
# $$
#
# If
#
# $$
# A(\omega)<0,
# $$
#
# then the negative real sign adds
#
# $$
# \pi.
# $$
#
# So the plotted phase may jump by $\pi$ around zeros of $A(\omega)$.
#
# The deck warns not to interpret those sign-related jumps as actual timing dispersion. 
#
# The better diagnostic is group delay away from magnitude zeros.
#
# # Slides 17–20: `Conv1d` as a trainable FIR
#
# The deck writes a multi-channel 1-D convolution as
#
# $$
# y_c[n]
# ======
#
# b_c
# +
# \sum_m
# \sum_{k=0}^{K-1}
# w_{c,m}[k]x_m[n+k].
# $$
#
# For one input and one output channel, this is simply a finite collection of weights applied over a moving local window.
#
# That is FIR-like behavior, except deep-learning libraries use correlation ordering. 
#
# # Important ML insight
#
# A random `Conv1d` kernel is not symmetric by default.
#
# Therefore it generally does not have linear phase.
#
# The deck explicitly demonstrates that a random kernel has varying group delay, while symmetrizing the taps forces the response into a linear-phase FIR family. 
#
# That is a really important conceptual bridge.
#
# # Good answer to “Does a CNN convolution preserve timing?”
#
# Weak answer:
#
# > Yes, because convolution is shift invariant.
#
# That confuses shift equivariance with phase behavior.
#
# Better answer:
#
# > Not necessarily. A generic learned finite kernel can have nonlinear phase because its taps are unconstrained.
#
# Strong answer:
#
# > `Conv1d` is shift-equivariant in the usual sense, but its kernel does not automatically have symmetric coefficients, so its frequency-dependent group delay may vary.
#
# That is the level of distinction I would want from graduate students.
#
# # Slides 19–20: How to enforce symmetry
#
# Suppose an odd-length symmetric kernel should be
#
# $$
# h=
# [a_0,a_1,\ldots,a_R,\ldots,a_1,a_0].
# $$
#
# Instead of learning every tap independently, learn only half plus the center.
#
# For example:
#
# ```python
# half = nn.Parameter(torch.randn(R))
# center = nn.Parameter(torch.randn(1))
# ```
#
# Then construct
#
# ```python
# h = torch.cat([
#     half,
#     center,
#     torch.flip(half, dims=[0])
# ])
# ```
#
# The realized kernel is symmetric at every training step. 
#
# Autograd still computes gradients normally.
#
# The architecture simply reduces the number of independent parameters.
#
# # Why is this an inductive bias?
#
# Because before seeing data, you are telling the model:
#
# > Only consider kernels with linear-phase symmetry.
#
# That shrinks the hypothesis space.
#
# Benefits:
#
# $$
# \text{fewer independent parameters},
# $$
#
# $$
# \text{exact generalized linear phase},
# $$
#
# and
#
# $$
# \text{preserved relative timing}.
# $$
#
# Cost:
#
# The model cannot learn a nonlinear-phase kernel even if the task would benefit from one.
#
# The deck makes exactly this tradeoff. 
#
# # Good answer to “Is constraining symmetry always better?”
#
# No.
#
# A strong answer is:
#
# > It is beneficial only when physics, interpretability, or deployment requirements justify preserving linear phase. Otherwise it may unnecessarily reduce model flexibility.
#
# # Slide 22: Fixed, constrained, or fully learned
#
# The deck presents a useful spectrum. 
#
# ## Fixed DSP front end
#
# Design the taps manually.
#
# No learning.
#
# Best when specifications are known.
#
# ## Constrained learned front end
#
# Learn coefficients but enforce structure such as:
#
# $$
# \text{symmetry},
# $$
#
# $$
# \text{band structure},
# $$
#
# or
#
# $$
# \text{sparsity}.
# $$
#
# This combines physical prior knowledge with optimization.
#
# ## Fully learned front end
#
# Let the task loss choose arbitrary finite kernels.
#
# Most flexible, but least constrained.
#
# A very good summary is:
#
# > These approaches lie on a bias-flexibility continuum rather than being mutually exclusive philosophies.
#
# # Slide 23: Multirate transition
#
# The lecture correctly avoids saying that linear phase is mathematically necessary for anti-aliasing.
#
# Anti-aliasing requires sufficient stopband attenuation before downsampling. 
#
# So the actual requirement is magnitude-domain:
#
# $$
# \boxed{
# |H(e^{j\omega})|
# \text{ must adequately suppress frequencies that would alias.}
# }
# $$
#
# Linear phase is useful for a different reason:
#
# $$
# \boxed{
# \text{preserving relative timing and waveform shape.}
# }
# $$
#
# A strong answer to
#
# > “Why are FIR filters so common in multirate systems?”
#
# is:
#
# > They can provide exact linear phase, they are always stable, and they fit naturally into efficient polyphase structures.
#
# # Slide 24 concept check
#
# A $16$-tap symmetric FIR has:
#
# $$
# N=16.
# $$
#
# That is even length.
#
# Symmetric.
#
# Therefore:
#
# $$
# \boxed{\text{Type II}}.
# $$
#
# A Type II FIR has
#
# $$
# \boxed{
# H(e^{j\pi})=0.
# }
# $$
#
# So it cannot be a conventional high-pass with nonzero Nyquist gain.
#
# The deck gives exactly this answer. 
#
# # Homework 5: Derive the delay term
#
# For symmetric taps,
#
# $$
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}A(\omega).
# $$
#
# Therefore,
#
# $$
# \angle H(e^{j\omega})
# =====================
#
# -\omega\frac{M}{2}
# $$
#
# apart from sign-related phase offsets.
#
# Hence,
#
# $$
# \tau_g
# ======
#
# *
#
# \frac{d}{d\omega}
# \left(
# -\omega\frac{M}{2}
# \right)
# =======
#
# \boxed{
# \frac{M}{2}
# }.
# $$
#
# Since
#
# $$
# M=N-1,
# $$
#
# we get
#
# $$
# \boxed{
# \tau_g
# ======
#
# \frac{N-1}{2}.
# }
# $$
#
# The deck's solution sketch gives exactly this result. 
#
# # Homework 6: Classifying FIR types
#
# A full-credit answer should state not only the type but the forced endpoint behavior.
#
# $$
# \boxed{
# \text{Type I}
# =============
#
# \text{odd/symmetric}
# }
# $$
#
# No forced endpoint zeros.
#
# $$
# \boxed{
# \text{Type II}
# ==============
#
# \text{even/symmetric}
# }
# $$
#
# Forced:
#
# $$
# H(e^{j\pi})=0.
# $$
#
# $$
# \boxed{
# \text{Type III}
# ===============
#
# \text{odd/antisymmetric}
# }
# $$
#
# Forced:
#
# $$
# H(e^{j0})=0
# $$
#
# and
#
# $$
# H(e^{j\pi})=0.
# $$
#
# $$
# \boxed{
# \text{Type IV}
# ==============
#
# \text{even/antisymmetric}
# }
# $$
#
# Forced:
#
# $$
# H(e^{j0})=0.
# $$
#
# This matches the instructor solution. 
#
# # Homework 7: Four-zero constellation
#
# For
#
# $$
# z_0
# ===
#
# 0.7e^{j0.6},
# $$
#
# the full set is
#
# $$
# \boxed{
# 0.7e^{j0.6},
# \quad
# 0.7e^{-j0.6},
# \quad
# \frac{1}{0.7}e^{-j0.6},
# \quad
# \frac{1}{0.7}e^{j0.6}.
# }
# $$
#
# Numerically,
#
# $$
# \frac{1}{0.7}
# \approx
# 1.4286.
# $$
#
# So:
#
# $$
# \boxed{
# 0.7e^{j0.6},
# \quad
# 0.7e^{-j0.6},
# \quad
# 1.4286e^{-j0.6},
# \quad
# 1.4286e^{j0.6}.
# }
# $$
#
# # Homework 8: Symmetry-constrained `Conv1d`
#
# A good answer should propose learning only the independent half of the kernel and constructing the rest by reflection.
#
# For odd $N$,
#
# $$
# h=
# [a_0,a_1,\ldots,a_R,\ldots,a_1,a_0].
# $$
#
# If there are $2R+1$ taps, only
#
# $$
# R+1
# $$
#
# parameters are independently learned instead of
#
# $$
# 2R+1.
# $$
#
# The benefit is:
#
# $$
# \boxed{
# \text{exact symmetry at every optimizer step}.
# }
# $$
#
# The cost is:
#
# $$
# \boxed{
# \text{the model cannot represent arbitrary nonlinear-phase kernels}.
# }
# $$
#
# The deck's solution sketch makes this exact point. 
#
# # Weekly quiz: what good answers should look like
#
# ## Q1: What determines magnitude from a pole-zero plot?
#
# Good answer:
#
# > Multiply the distances from the unit-circle point to all zeros and divide by the product of the distances to all poles, including the gain factor.
#
# Mathematically,
#
# $$
# |H(e^{j\omega})|
# ================
#
# |K|
# \frac{\prod_i|e^{j\omega}-z_i|}
# {\prod_k|e^{j\omega}-p_k|}.
# $$
#
# ## Q2: State phase delay and group delay
#
# $$
# \boxed{
# \tau_p(\omega)
# ==============
#
# *
#
# \frac{\angle H(e^{j\omega})}{\omega}
# }
# $$
#
# and
#
# $$
# \boxed{
# \tau_g(\omega)
# ==============
#
# *
#
# \frac{d}{d\omega}
# \angle H(e^{j\omega}).
# }
# $$
#
# ## Q3: What defines all-pass?
#
# $$
# \boxed{
# |H_{ap}(e^{j\omega})|=1
# }
# $$
#
# for all $\omega$.
#
# ## Q4: Why can equal magnitudes give different transient outputs?
#
# Because phase and group delay can differ.
#
# The spectral components then arrive with different relative timing.
#
# These first four quiz answers are summarized directly in the deck. 
#
# ## Q5
#
# For a real symmetric FIR with
#
# $$
# N=21,
# $$
#
# the group delay is
#
# $$
# \tau_g
# ======
#
# # \frac{21-1}{2}
#
# \boxed{10}.
# $$
#
# ## Q6
#
# Even-length and symmetric means:
#
# $$
# \boxed{\text{Type II}}.
# $$
#
# Forced zero:
#
# $$
# \boxed{z=-1}
# $$
#
# or equivalently
#
# $$
# \boxed{\omega=\pi}.
# $$
#
# ## Q7
#
# For a generic complex zero $z_0$, the other three are
#
# $$
# \boxed{
# z_0^*,
# \quad
# \frac{1}{z_0},
# \quad
# \frac{1}{z_0^*}.
# }
# $$
#
# ## Q8
#
# Does a random `nn.Conv1d` kernel automatically have linear phase?
#
# $$
# \boxed{\text{No}.}
# $$
#
# Because no symmetry or antisymmetry constraint is imposed by default.
#
# These are exactly the instructor quiz answers. 
#
# # Good exit-ticket questions
#
# If I wanted three high-value exit tickets from Lecture 10, I would use these.
#
# ## Exit ticket 1
#
# **Question**
#
# Why does symmetry in an FIR impulse response produce constant group delay?
#
# **Good answer**
#
# > Pairing symmetric taps lets the response factor as
#
# $$
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}A(\omega),
# $$
#
# where $A(\omega)$ is real. The only frequency-dependent phase slope is therefore $-M/2$, giving
#
# $$
# \tau_g=M/2.
# $$
#
# ## Exit ticket 2
#
# **Question**
#
# Why can a Type II FIR not be a conventional high-pass filter?
#
# **Good answer**
#
# > Type II means even length and symmetric taps, which structurally forces
#
# $$
# H(e^{j\pi})=0.
# $$
#
# A conventional high-pass requires nonzero gain near Nyquist, so the requirements conflict.
#
# ## Exit ticket 3
#
# **Question**
#
# What inductive bias is introduced by forcing a `Conv1d` kernel to remain symmetric?
#
# **Good answer**
#
# > The model is restricted to a linear-phase FIR family. That reduces the number of independent parameters and preserves relative timing, but prevents the model from learning arbitrary nonlinear-phase kernels.
#
# # What I would expect a senior student to know
#
# A strong senior should be able to:
#
# derive
#
# $$
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}A(\omega),
# $$
#
# explain why
#
# $$
# \tau_g
# ======
#
# \frac{N-1}{2},
# $$
#
# identify Types I–IV from length and symmetry,
#
# state the forced endpoint zeros,
#
# and explain why a generic `Conv1d` kernel does not automatically have linear phase.
#
# # What I would expect a first-year graduate student to know
#
# In addition, I would expect them to:
#
# derive the cosine and sine pairing forms,
#
# explain the reciprocal-conjugate zero constellation from the polynomial symmetry,
#
# distinguish genuine group-delay behavior from apparent phase jumps at zeros,
#
# understand symmetry constraints as architectural inductive bias,
#
# and evaluate when restricting a learned convolution to a linear-phase family is physically justified versus unnecessarily restrictive.
#
# # The most important takeaway
#
# The deepest statement in Lecture 10 is:
#
# $$
# \boxed{
# \text{linear phase is not just a frequency-response property;
# it is a structural constraint on the impulse response.}
# }
# $$
#
# For real FIR filters, that structure is symmetry or antisymmetry.
#
# That same idea carries directly into machine learning: if relative timing is known to matter, you can encode that physics into the model by constraining a trainable `Conv1d` kernel instead of hoping the optimizer discovers the desired symmetry on its own. 
#
# ----------------------------
#

# %% [markdown]
# Week 5, Lectures 9 and 10, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# Gemini Pro:
#
# **Question 1: What geometric quantities from a pole-zero plot determine the magnitude response?**
#
# * Magnitude is determined by calculating a distance ratio.
#
#
# * This is evaluated as the lengths of the zero-vectors in the numerator divided by the lengths of the pole-vectors in the denominator.
#
#
#
# **Question 2: Why might two filters with identical magnitude responses produce differently shaped transient waveforms?**
#
# * Filters with the exact same magnitude response can still have different phase responses.
#
#
# * These differences in phase and group delay alter the relative arrival times of the signal's frequency components, which changes the shape of the resulting transient waveform.
#
#
#
# **Question 3: What is the defining property of an all-pass system, and what does it change?**
#
# * An all-pass system is defined by having a magnitude of exactly one across all frequencies.
#
#
# * While it leaves the total magnitude unchanged, it can strongly reshape the signal's phase and group delay.
#
#
#
# **Question 4: Why is it structurally impossible for a Type II FIR filter to act as a high-pass filter?**
#
# * A Type II filter is an even-length, symmetric FIR filter.
#
#
# * This specific structural symmetry forces a zero at the Nyquist frequency ($z=-1$ or $\omega=\pi$).
#
#
# * Because the gain is forced to vanish at the Nyquist frequency, the filter cannot achieve the nonzero high-frequency gain required for a high-pass response.
#
#
#
# **Question 5: If a complex zero $z_0$ is located off the unit circle, what other zeros must accompany it in a real linear-phase FIR filter?**
#
# * The original complex zero must be accompanied by three distinct companions to form a four-zero constellation.
#
#
# * The required accompanying zeros are $z_0^*$, $1/z_0$, and $1/z_0^*$.
#
#
#
# **Question 6: Does a generic, randomly initialized `nn.Conv1d` kernel automatically exhibit linear phase?**
#
# * No, a standard random convolutional kernel does not automatically possess linear phase.
#
#
# * Because generic kernels are completely unconstrained, they lack the symmetric structure guaranteed to produce linear phase behavior.
#
# --------------------------
#

# %% [markdown]
# Week 5, Lectures 9 and 10, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# ChatGPT:
#
# For these two lectures, I would make the exit tickets conceptual, short, and diagnostic. The best questions should tell you whether students can connect the mathematics to the physical interpretation and to the ML consequences.
#
# Lecture 9 emphasizes pole-zero geometry, phase versus group delay, all-pass behavior, and timing distortion.  Lecture 10 emphasizes generalized linear phase, FIR symmetry classes, zero constellations, and symmetry-constrained `Conv1d` kernels. 
#
# ## Lecture 9 — Time/Frequency Responses and Group Delay
#
# ### 1. What happens when a pole moves closer to the unit circle?
#
# **Question**
#
# A pole at angle $\omega_0$ moves radially from radius $0.5$ to $0.95$. What changes near $\omega_0$?
#
# **Good answer**
#
# The distance from the pole to the unit-circle point $e^{j\omega_0}$ becomes smaller, so the denominator in
#
# $$
# |H(e^{j\omega})|
# ================
#
# |K|
# \frac{\prod_i |e^{j\omega}-z_i|}
# {\prod_k |e^{j\omega}-p_k|}
# $$
#
# decreases. Therefore the magnitude increases sharply near $\omega_0$.
#
# A stronger answer adds that the phase changes more rapidly there, producing a more concentrated group-delay contribution. 
#
# ---
#
# ### 2. What is the difference between phase delay and group delay?
#
# **Question**
#
# Explain the difference between
#
# $$
# \tau_p(\omega)
# ==============
#
# -\frac{\angle H(e^{j\omega})}{\omega}
# $$
#
# and
#
# $$
# \tau_g(\omega)
# ==============
#
# -\frac{d}{d\omega}\angle H(e^{j\omega}).
# $$
#
# **Good answer**
#
# Phase delay interprets the phase shift of a single sinusoidal component as an equivalent time delay.
#
# Group delay measures the local slope of phase and describes how a narrowband packet or envelope is delayed.
#
# A strong answer adds:
#
# > They are equal for pure linear phase through the origin, but they are generally different. 
#
# ---
#
# ### 3. Why does nonlinear phase distort a transient?
#
# **Question**
#
# A filter has an acceptable magnitude response but strongly frequency-dependent group delay. What can happen to a short pulse?
#
# **Good answer**
#
# A pulse contains many frequency components. If different components experience different delays, they no longer recombine with the same relative timing. The pulse can become smeared, shifted, ring, or have a changed onset slope. 
#
# **Strong answer**
#
# > Magnitude tells us how much of each frequency remains; phase determines how those components line up in time.
#
# ---
#
# ### 4. Can two filters with identical magnitude responses produce different outputs?
#
# **Question**
#
# Suppose
#
# $$
# |H_1(e^{j\omega})|
# ==================
#
# |H_2(e^{j\omega})|
# $$
#
# for all $\omega$, but their phases differ. Must the time-domain outputs be identical?
#
# **Answer**
#
# $$
# \boxed{\text{No}}
# $$
#
# The output spectra may have identical magnitudes but different phases, so transient waveform shapes can differ. 
#
# ---
#
# ### 5. What does an all-pass filter actually do?
#
# **Question**
#
# If
#
# $$
# |H_{ap}(e^{j\omega})|=1
# $$
#
# for all $\omega$, does the filter do nothing?
#
# **Answer**
#
# No.
#
# It preserves magnitude but can change phase and group delay substantially.
#
# **Good explanation**
#
# > An all-pass filter is a phase-shaping element, not a magnitude-shaping element. 
#
# ---
#
# ### 6. What can an all-pass equalizer fix?
#
# **Question**
#
# A system has acceptable magnitude response but uneven group delay. What can an all-pass section improve, and what can it not improve?
#
# **Good answer**
#
# It can modify phase and group delay while preserving magnitude:
#
# $$
# \tau_{g,\text{tot}}(\omega)
# ===========================
#
# \tau_{g,H}(\omega)
# +
# \tau_{g,ap}(\omega).
# $$
#
# It cannot fix magnitude notches, inadequate stopband attenuation, or insufficient bandwidth because
#
# $$
# |H_{ap}(e^{j\omega})|=1.
# $$
#
#
#
# ---
#
# ### 7. ML context: when does phase distortion matter?
#
# **Question**
#
# When is nonlinear phase likely to matter to a machine-learning classifier?
#
# **Good answer**
#
# When the classifier relies on timing or waveform morphology, such as:
#
# * ECG peak timing,
# * audio onset shape,
# * inter-channel timing,
# * vibration impulse spacing.
#
# If the model depends mainly on long-term magnitude statistics, nonlinear phase may matter less. 
#
# ---
#
# ## Lecture 10 — Linear Phase, FIR Types, and ML Intersections
#
# ### 1. Why does FIR symmetry produce constant group delay?
#
# **Question**
#
# For a symmetric FIR,
#
# $$
# h[n]=h[M-n].
# $$
#
# Why does this imply constant group delay?
#
# **Good answer**
#
# Pairing symmetric taps allows the response to be written as
#
# $$
# H(e^{j\omega})
# ==============
#
# e^{-j\omega M/2}A(\omega),
# $$
#
# where $A(\omega)$ is real.
#
# Therefore the frequency-dependent phase slope is
#
# $$
# -\frac{M}{2},
# $$
#
# so
#
# $$
# \boxed{
# \tau_g=\frac{M}{2}=\frac{N-1}{2}.
# }
# $$
#
#
#
# ---
#
# ### 2. Why are there four linear-phase FIR types?
#
# **Question**
#
# What two choices produce the four standard FIR linear-phase types?
#
# **Good answer**
#
# The two choices are:
#
# * symmetric versus antisymmetric taps,
# * odd versus even filter length.
#
# These combinations determine whether the midpoint is an actual sample or a half-sample and which endpoint zeros are forced. 
#
# ---
#
# ### 3. Classify a filter quickly
#
# **Question**
#
# A real FIR has $N=16$ taps and satisfies
#
# $$
# h[n]=h[15-n].
# $$
#
# What type is it, and what frequency must be zero?
#
# **Answer**
#
# Since $N$ is even and the filter is symmetric,
#
# $$
# \boxed{\text{Type II}}
# $$
#
# and
#
# $$
# \boxed{
# H(e^{j\pi})=0.
# }
# $$
#
# Therefore it cannot be a conventional high-pass with nonzero Nyquist gain. 
#
# ---
#
# ### 4. Why can Type II not realize a conventional high-pass?
#
# **Question**
#
# Why is this impossible regardless of filter order?
#
# **Good answer**
#
# Type II symmetry structurally forces
#
# $$
# H(e^{j\pi})=0.
# $$
#
# A conventional high-pass requires appreciable gain at Nyquist, so the requirements conflict.
#
# A strong answer adds:
#
# > Increasing the order does not remove the forced zero; this is a structural constraint, not an approximation issue. 
#
# ---
#
# ### 5. What endpoint zeros are forced for the four FIR types?
#
# **Question**
#
# State the forced endpoint zeros for Types I–IV.
#
# **Answer**
#
# * Type I: no forced zero at $\omega=0$ or $\omega=\pi$
# * Type II:
#
# $$
# H(e^{j\pi})=0
# $$
#
# * Type III:
#
# $$
# H(e^{j0})=0
# $$
#
# and
#
# $$
# H(e^{j\pi})=0
# $$
#
# * Type IV:
#
# $$
# H(e^{j0})=0
# $$
#
# These constraints follow directly from parity and symmetry. 
#
# ---
#
# ### 6. Why do zeros appear in reciprocal-conjugate sets?
#
# **Question**
#
# If a real linear-phase FIR has a complex zero $z_0$ off the unit circle, what other zeros accompany it?
#
# **Answer**
#
# $$
# \boxed{
# z_0^*,
# \quad
# \frac{1}{z_0},
# \quad
# \frac{1}{z_0^*}
# }
# $$
#
# **Good explanation**
#
# Real coefficients imply conjugate symmetry, while linear-phase palindromic or anti-palindromic structure implies reciprocal symmetry. 
#
# ---
#
# ### 7. Does a random `Conv1d` kernel have linear phase?
#
# **Question**
#
# Does a generic randomly initialized `nn.Conv1d` kernel automatically preserve constant group delay?
#
# **Answer**
#
# $$
# \boxed{\text{No}}
# $$
#
# A generic kernel has no symmetry or antisymmetry constraint, so its phase and group delay can be nonlinear. 
#
# ---
#
# ### 8. What inductive bias does a symmetric trainable kernel add?
#
# **Question**
#
# Suppose we force a trainable kernel to have the form
#
# $$
# h=
# [a_0,a_1,\ldots,a_R,\ldots,a_1,a_0].
# $$
#
# What benefit and cost does that introduce?
#
# **Good answer**
#
# Benefit:
#
# * exact generalized linear phase,
# * fewer independent parameters,
# * preservation of relative timing.
#
# Cost:
#
# * the model cannot learn arbitrary nonlinear-phase kernels.
#
# This is an architectural inductive bias, not a post-processing correction. 
#
# ---
#
# ### 9. Is linear phase required for anti-alias filtering?
#
# **Question**
#
# Does an anti-aliasing filter have to be linear phase?
#
# **Answer**
#
# No.
#
# Anti-aliasing requires sufficient stopband attenuation before downsampling.
#
# Linear phase is often desirable because it preserves waveform timing and alignment, but it is not mathematically required for alias suppression. 
#
# ---
#
# ## My preferred three-question set for each lecture
#
# For **Lecture 9**, I would use:
#
# 1. A pole moves closer to the unit circle at angle $\omega_0$. What happens to magnitude and group delay near $\omega_0$?
# 2. What is the difference between phase delay and group delay?
# 3. Why can an all-pass filter alter an ML feature even though its magnitude response is exactly one?
#
# For **Lecture 10**, I would use:
#
# 1. Why does symmetry in an FIR impulse response produce constant group delay?
# 2. Why can a Type II FIR never be a conventional high-pass with nonzero Nyquist gain?
# 3. What is gained and lost by forcing a trainable `Conv1d` kernel to remain symmetric?
#
# Those six questions are short enough for about **4–6 minutes per lecture**, but they test whether students understand the structural reasoning behind the equations rather than simply recalling definitions.
#
# --------------------------
#
