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
# # FIR Type I-IV Problem Set
#
#

# %% [markdown]
# Can you create a problem set about FIR Type I - IV systems for senior electrical engineering students in an advanced DSP class using the Oppenheim/Schafer textbook?
#
# Gemini Thinking:
#
# # Linear Phase FIR Systems (Types I–IV) Problem Set
#
# **Course:** Advanced Discrete-Time Signal Processing (Senior EE / First-Year Graduate)
#
# **Textbook Reference:** Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd Edition), Chapter 5: *Differential Properties of Linear-Phase Systems*
#
# ---
#
# ## Summary of Reference Definitions
#
# For a causal FIR filter of order $M$ (length $N = M + 1$) with impulse response $h[n]$ defined on $0 \le n \le M$:
#
# $$\begin{array}{\|c\|c\|c\|c\|c\|} \hline \textbf{Type} & \textbf{Symmetry of } h[n] & \textbf{Order } M & \textbf{Frequency Response } H(e^{j\omega}) & \textbf{Mandatory Zeros} \\ \hline \text{Type I} & h[n] = h[M-n] & \text{Even} & A(\omega) e^{-j\omega M/2} & \text{None} \\ \hline \text{Type II} & h[n] = h[M-n] & \text{Odd} & A(\omega) e^{-j\omega M/2} & \omega = \pi \ (z = -1) \\ \hline \text{Type III} & h[n] = -h[M-n] & \text{Even} & j A(\omega) e^{-j\omega M/2} & \omega = 0, \pi \ (z = 1, -1) \\ \hline \text{Type IV} & h[n] = -h[M-n] & \text{Odd} & j A(\omega) e^{-j\omega M/2} & \omega = 0 \ (z = 1) \\ \hline \end{array}$$
#
# *Note: $A(\omega)$ represents the real-valued amplitude response function.*
#
# ---
#
# ## Problems
#
# ### Problem 1: Classification & Frequency Response Derivation
#
# Given four causal discrete-time FIR systems with impulse responses defined for $0 \le n \le M$:
#
# 1. $h_1[n] = [1, \, 3, \, 5, \, 3, \, 1]$
# 2. $h_2[n] = [2, \, -1, \, 4, \, 4, \, -1, \, 2]$
# 3. $h_3[n] = [1, \, 2, \, 0, \, -2, \, -1]$
# 4. $h_4[n] = [-1, \, 3, \, -3, \, 1]$
#
# **(a)** Classify each system into its corresponding FIR Type (I, II, III, or IV). Specify the filter order $M$ and sequence length $N$ for each.
#
# **(b)** For $h_2[n]$, derive the compact closed-form expression for its frequency response $H_2(e^{j\omega})$. Express $H_2(e^{j\omega})$ in the form $A(\omega) e^{-j\omega M/2}$ where $A(\omega)$ is a real-valued trigonometric sum.
#
# **(c)** For $h_3[n]$, derive $H_3(e^{j\omega})$ and show that it exhibits a constant $90^\circ$ ($\pi/2$ rad) phase shift relative to the linear phase delay term $e^{-j\omega M/2}$.
#
# ---
#
# ### Problem 2: Mandatory Zero Constraints & Filter Approximations
#
# Let $H(z) = \sum_{n=0}^{M} h[n] z^{-n}$ be the transfer function of a causal linear-phase FIR system with real impulse response $h[n]$.
#
# **(a)** Prove analytically using $H(z)$ evaluated at $z = -1$ that all Type II and Type III systems must possess a zero at $z = -1$ ($\omega = \pi$).
#
# **(b)** Prove analytically using $H(z)$ evaluated at $z = 1$ that all Type III and Type IV systems must possess a zero at $z = 1$ ($\omega = 0$).
#
# **(c)** Complete the feasibility table below for designing ideal frequency-selective filters and operators using FIR Types I–IV. Mark each combination as **Feasible** or **Infeasible**. For each **Infeasible** entry, state the exact zero constraint preventing its implementation.
#
# $$\begin{array}{\|c\|c\|c\|c\|c\|} \hline \textbf{Filter Application} & \textbf{Type I} & \textbf{Type II} & \textbf{Type III} & \textbf{Type IV} \\ \hline \text{Lowpass Filter (LPF)} & & & & \\ \hline \text{Highpass Filter (HPF)} & & & & \\ \hline \text{Bandpass Filter (BPF)} & & & & \\ \hline \text{Bandstop Filter (BSF)} & & & & \\ \hline \text{Wideband Differentiator } (H(\pi) \neq 0) & & & & \\ \hline \text{Wideband Hilbert Transformer} & & & & \\ \hline \end{array}$$
#
# ---
#
# ### Problem 3: Zero Constellation Symmetry & System Synthesis
#
# **(a)** Prove that if $z_0 = r e^{j\theta}$ (with $r \neq 1$ and $\theta \neq 0, \pi$) is a zero of a causal, real-coefficient linear-phase FIR filter $H(z)$, then $z_0^*$, $1/z_0$, and $1/z_0^*$ must also be zeros of $H(z)$ with equal multiplicity.
#
# **(b)** A Type I FIR filter of order $M = 4$ ($N = 5$) with real impulse response $h[n]$ has a complex zero at:
#
# $$z_1 = \frac{1}{2} e^{j \pi/3}$$
#
# i. List all four zeros of $H(z)$ in both polar and Cartesian forms.
#
# ii. Derive the normalized system transfer function $H(z)$ given that $H(e^{j0}) = 1$.
#
# iii. Determine the explicit impulse response sequence $h[n]$ for $n = 0, 1, 2, 3, 4$.
#
# ---
#
# ### Problem 4: System Interconnections & Composite Types
#
# Consider two causal linear-phase FIR filters: $H_A(z)$ of Type II with order $M_A$, and $H_B(z)$ of Type IV with order $M_B$.
#
# **(a)** Determine the FIR Type (I, II, III, or IV) of the cascaded system $H_C(z) = H_A(z) \cdot H_B(z)$. Express the overall order $M_C$ in terms of $M_A$ and $M_B$, and justify the symmetry of $h_C[n]$.
#
# **(b)** Assume $M_A = M_B = M$. Determine whether the parallel interconnection $H_D(z) = H_A(z) + H_B(z)$ maintains strict generalized linear phase. Support your answer mathematically by evaluating the phase of $H_D(e^{j\omega})$.
#
# **(c)** Can a parallel combination of a Type I filter $H_1(z)$ and a Type III filter $H_3(z)$ with identical delay $\tau = M/2$ yield a linear phase system? Explain.
#
# ---
#
# ## Solutions Manual
#
# ### Solution to Problem 1
#
# **(a) System Classification:**
#
# 1. $h_1[n] = [1, 3, 5, 3, 1]$
# * Order $M = 4$ (even), Length $N = 5$.
# * $h_1[n] = h_1[4-n]$ (Symmetric).
# * **Classification: Type I**
#
#
# 2. $h_2[n] = [2, -1, 4, 4, -1, 2]$
# * Order $M = 5$ (odd), Length $N = 6$.
# * $h_2[n] = h_2[5-n]$ (Symmetric).
# * **Classification: Type II**
#
#
# 3. $h_3[n] = [1, 2, 0, -2, -1]$
# * Order $M = 4$ (even), Length $N = 5$.
# * $h_3[n] = -h_3[4-n]$ (Antisymmetric).
# * **Classification: Type III**
#
#
# 4. $h_4[n] = [-1, 3, -3, 1]$
# * Order $M = 3$ (odd), Length $N = 4$.
# * $h_4[n] = -h_4[3-n]$ (Antisymmetric).
# * **Classification: Type IV**
#
#
#
# ---
#
# **(b) Derivation of $H_2(e^{j\omega})$:**
#
# $$H_2(e^{j\omega}) = \sum_{n=0}^{5} h_2[n] e^{-j\omega n} = 2 - e^{-j\omega} + 4 e^{-j 2\omega} + 4 e^{-j 3\omega} - e^{-j 4\omega} + 2 e^{-j 5\omega}$$
#
# Factor out the group delay term $e^{-j \frac{5}{2}\omega}$:
#
# $$H_2(e^{j\omega}) = e^{-j 2.5\omega} \left[ 2\left(e^{j 2.5\omega} + e^{-j 2.5\omega}\right) - \left(e^{j 1.5\omega} + e^{-j 1.5\omega}\right) + 4\left(e^{j 0.5\omega} + e^{-j 0.5\omega}\right) \right]$$
#
# Applying Euler's identity $\cos(\theta) = \frac{e^{j\theta} + e^{-j\theta}}{2}$:
#
# $$H_2(e^{j\omega}) = \left[ 4 \cos(2.5\omega) - 2 \cos(1.5\omega) + 8 \cos(0.5\omega) \right] e^{-j 2.5\omega}$$
#
# Thus:
#
# $$A(\omega) = 8 \cos(0.5\omega) - 2 \cos(1.5\omega) + 4 \cos(2.5\omega)$$
#
# $$H_2(e^{j\omega}) = A(\omega) e^{-j 2.5\omega}$$
#
# ---
#
# **(c) Derivation of $H_3(e^{j\omega})$:**
#
# $$H_3(e^{j\omega}) = 1 + 2e^{-j\omega} + 0 - 2e^{-j 3\omega} - e^{-j 4\omega}$$
#
# Factor out the group delay term $e^{-j 2\omega}$:
#
# $$H_3(e^{j\omega}) = e^{-j 2\omega} \left[ \left(e^{j 2\omega} - e^{-j 2\omega}\right) + 2\left(e^{j\omega} - e^{-j\omega}\right) \right]$$
#
# Using Euler's identity $\sin(\theta) = \frac{e^{j\theta} - e^{-j\theta}}{2j}$:
#
# $$H_3(e^{j\omega}) = e^{-j 2\omega} \left[ 2j \sin(2\omega) + 4j \sin(\omega) \right] = j \left[ 4 \sin(\omega) + 2 \sin(2\omega) \right] e^{-j 2\omega}$$
#
# Since $j = e^{j\pi/2}$:
#
# $$H_3(e^{j\omega}) = A(\omega) e^{j(\pi/2 - 2\omega)}, \quad \text{where } A(\omega) = 4 \sin(\omega) + 2 \sin(2\omega)$$
#
# This confirms the constant $90^\circ$ ($\pi/2$) phase shift added to the linear phase delay.
#
# ---
#
# ### Solution to Problem 2
#
# **(a) Proof of mandatory zero at $z = -1$ for Type II and III:**
#
# For $z = -1 = e^{j\pi}$:
#
# $$H(-1) = \sum_{n=0}^{M} h[n] (-1)^{-n} = \sum_{n=0}^{M} h[n] (-1)^n$$
#
# * **Type II ($h[n] = h[M-n]$, $M$ is odd):**
# Pairing terms $n$ and $M-n$:
# $$h[M-n] (-1)^{M-n} = h[n] (-1)^M (-1)^{-n} = -h[n] (-1)^n \quad (\text{since } M \text{ is odd})$$
#
#
# Grouping all terms:
# $$H(-1) = \sum_{n=0}^{(M-1)/2} \left[ h[n](-1)^n + h[M-n](-1)^{M-n} \right] = \sum_{n=0}^{(M-1)/2} \left[ h[n](-1)^n - h[n](-1)^n \right] = 0$$
#
#
# * **Type III ($h[n] = -h[M-n]$, $M$ is even):**
# For the center sample $n = M/2$: $h[M/2] = -h[M/2] \implies h[M/2] = 0$.
# Pairing remaining terms:
# $$h[M-n] (-1)^{M-n} = -h[n] (-1)^M (-1)^n = -h[n] (-1)^n \quad (\text{since } M \text{ is even})$$
#
#
# $$H(-1) = \sum_{n=0}^{(M/2)-1} \left[ h[n](-1)^n - h[n](-1)^n \right] + h[M/2](-1)^{M/2} = 0$$
#
#
#
# Thus, $H(-1) = 0$ for both Type II and Type III systems.
#
# ---
#
# **(b) Proof of mandatory zero at $z = 1$ for Type III and IV:**
#
# For $z = 1 = e^{j0}$:
#
# $$H(1) = \sum_{n=0}^{M} h[n]$$
#
# * **Type III ($h[n] = -h[M-n]$, $M$ is even):**
# $$H(1) = \sum_{n=0}^{(M/2)-1} (h[n] + h[M-n]) + h[M/2] = \sum_{n=0}^{(M/2)-1} (h[n] - h[n]) + 0 = 0$$
#
#
# * **Type IV ($h[n] = -h[M-n]$, $M$ is odd):**
# $$H(1) = \sum_{n=0}^{(M-1)/2} (h[n] + h[M-n]) = \sum_{n=0}^{(M-1)/2} (h[n] - h[n]) = 0$$
#
#
#
# Thus, $H(1) = 0$ for both Type III and Type IV systems.
#
# ---
#
# **(c) Completed Feasibility Table:**
#
# $$\begin{array}{\|c\|c\|c\|c\|c\|} \hline \textbf{Filter Application} & \textbf{Type I} & \textbf{Type II} & \textbf{Type III} & \textbf{Type IV} \\ \hline \text{Lowpass Filter (LPF)} & \text{Feasible} & \text{Feasible} & \text{Infeasible } (H(0)=0) & \text{Infeasible } (H(0)=0) \\ \hline \text{Highpass Filter (HPF)} & \text{Feasible} & \text{Infeasible } (H(\pi)=0) & \text{Infeasible } (H(\pi)=0) & \text{Feasible} \\ \hline \text{Bandpass Filter (BPF)} & \text{Feasible} & \text{Feasible} & \text{Feasible} & \text{Feasible} \\ \hline \text{Bandstop Filter (BSF)} & \text{Feasible} & \text{Infeasible } (H(\pi)=0) & \text{Infeasible } (H(0)=H(\pi)=0) & \text{Infeasible } (H(0)=0) \\ \hline \text{Wideband Differentiator} & \text{Infeasible } (\text{No } 90^\circ \text{ shift}) & \text{Infeasible } (H(\pi)=0) & \text{Infeasible } (H(\pi)=0) & \text{Feasible} \\ \hline \text{Wideband Hilbert Trans.} & \text{Infeasible } (\text{No } 90^\circ \text{ shift}) & \text{Infeasible } (H(\pi)=0) & \text{Feasible } (\text{Bandpass}) & \text{Feasible} \\ \hline \end{array}$$
#
# ---
#
# ### Solution to Problem 3
#
# **(a) Proof of Zero Quadruplet Symmetry:**
#
# 1. **Real impulse response property:**
# If $h[n] \in \mathbb{R}$, then $H(z^*) = [H(z)]^*$. Therefore, if $H(z_0) = 0$, then $H(z_0^*) = 0$.
# 2. **Linear phase symmetry property:**
# For linear phase FIR filters, $h[n] = \pm h[M-n]$. Taking $z$-transform:
# $$H(z) = \pm z^{-M} H(z^{-1})$$
#
#
# If $H(z_0) = 0$, then $\pm z_0^{-M} H(z_0^{-1}) = 0 \implies H(z_0^{-1}) = 0$ (since $z_0 \neq 0$).
# 3. **Combined Symmetry:**
# Combining conjugate symmetry and reciprocal symmetry gives the quadruplet of roots:
# $$\{ z_0, \, z_0^*, \, z_0^{-1}, \, (z_0^*)^{-1} \}$$
#
#
# In polar form with $z_0 = r e^{j\theta}$:
# $$\left\{ r e^{j\theta}, \ r e^{-j\theta}, \ \frac{1}{r} e^{-j\theta}, \ \frac{1}{r} e^{j\theta} \right\}$$
#
#
#
# ---
#
# **(b) System Synthesis:**
#
# **i. Zeros of $H(z)$:**
#
# Given $z_1 = \frac{1}{2} e^{j \pi/3} = \frac{1}{4} + j \frac{\sqrt{3}}{4}$:
#
# $$\begin{aligned} z_1 &= \frac{1}{2} e^{j \pi/3} = \frac{1}{4} + j \frac{\sqrt{3}}{4} \\ z_2 &= z_1^* = \frac{1}{2} e^{-j \pi/3} = \frac{1}{4} - j \frac{\sqrt{3}}{4} \\ z_3 &= \frac{1}{z_1} = 2 e^{-j \pi/3} = 1 - j \sqrt{3} \\ z_4 &= \frac{1}{z_1^*} = 2 e^{j \pi/3} = 1 + j \sqrt{3} \end{aligned}$$
#
# **ii. Transfer Function $H(z)$:**
#
# Group conjugate pairs into quadratic factors:
#
# $$\begin{aligned} Q_1(z) &= (1 - z_1 z^{-1})(1 - z_1^* z^{-1}) = 1 - 2\text{Re}(z_1)z^{-1} + \vert{}z_1\vert{}^2 z^{-2} \\ &= 1 - \frac{1}{2} z^{-1} + \frac{1}{4} z^{-2} \\ Q_2(z) &= (1 - z_3 z^{-1})(1 - z_4 z^{-1}) = 1 - 2\text{Re}(z_3)z^{-1} + \vert{}z_3\vert{}^2 z^{-2} \\ &= 1 - 2 z^{-1} + 4 z^{-2} \end{aligned}$$
#
# Multiply $Q_1(z)$ and $Q_2(z)$:
#
# $$\begin{aligned} H(z) &= C \cdot \left(1 - \frac{1}{2} z^{-1} + \frac{1}{4} z^{-2}\right) \left(1 - 2 z^{-1} + 4 z^{-2}\right) \\ &= C \cdot \left[ 1 - 2.5 z^{-1} + 5.25 z^{-2} - 2.5 z^{-3} + z^{-4} \right] \end{aligned}$$
#
# Apply normalization condition $H(1) = 1$:
#
# $$H(1) = C \cdot (1 - 2.5 + 5.25 - 2.5 + 1) = 2.25 C = 1 \implies C = \frac{1}{2.25} = \frac{4}{9}$$
#
# $$H(z) = \frac{4}{9} - \frac{10}{9} z^{-1} + \frac{21}{9} z^{-2} - \frac{10}{9} z^{-3} + \frac{4}{9} z^{-4}$$
#
# **iii. Impulse Response $h[n]$:**
#
# $$h[n] = \left[ \frac{4}{9}, \, -\frac{10}{9}, \, \frac{21}{9}, \, -\frac{10}{9}, \, \frac{4}{9} \right] \quad \text{for } n = 0, 1, 2, 3, 4$$
#
# ---
#
# ### Solution to Problem 4
#
# **(a) Cascade System $H_C(z) = H_A(z) H_B(z)$:**
#
# * **Order:** $M_C = M_A + M_B$. Since $M_A$ is odd and $M_B$ is odd, $M_C = \text{odd} + \text{odd} = \text{even}$.
# * **Frequency Response:**
# $$H_A(e^{j\omega}) = A_A(\omega) e^{-j\omega M_A/2}$$
#
#
# $$H_B(e^{j\omega}) = j A_B(\omega) e^{-j\omega M_B/2}$$
#
#
# $$H_C(e^{j\omega}) = H_A(e^{j\omega}) H_B(e^{j\omega}) = j \left[ A_A(\omega) A_B(\omega) \right] e^{-j\omega (M_A + M_B)/2} = j A_C(\omega) e^{-j\omega M_C/2}$$
#
#
#
# Since $M_C$ is even and the response contains the factor $j = e^{j\pi/2}$ (indicating anti-symmetric impulse response $h_C[n] = -h_C[M_C-n]$), $H_C(z)$ is a **Type III FIR filter**.
#
# ---
#
# **(b) Parallel System $H_D(z) = H_A(z) + H_B(z)$ with $M_A = M_B = M$:**
#
# $$H_D(e^{j\omega}) = A_A(\omega) e^{-j\omega M/2} + j A_B(\omega) e^{-j\omega M/2} = \left[ A_A(\omega) + j A_B(\omega) \right] e^{-j\omega M/2}$$
#
# Let $A_D(\omega) = A_A(\omega) + j A_B(\omega) = \vert{}A_D(\omega)\vert{} e^{j \theta(\omega)}$, where:
#
# $$\theta(\omega) = \arctan\left(\frac{A_B(\omega)}{A_A(\omega)}\right)$$
#
# The overall phase response is:
#
# $$\arg[H_D(e^{j\omega})] = \theta(\omega) - \frac{M}{2}\omega$$
#
# Because $\theta(\omega)$ is a non-linear function of $\omega$, the parallel combination **does NOT maintain strict generalized linear phase** (unless $A_B(\omega)/A_A(\omega)$ is independent of $\omega$, which cannot hold across all frequencies for non-trivial filters).
#
# ---
#
# **(c) Parallel Combination of Type I and Type III Systems:**
#
# **No.** A Type I system has a purely real amplitude response $A_1(\omega)$ relative to its linear phase delay $e^{-j\omega \tau}$, whereas a Type III system has a purely imaginary amplitude response $j A_3(\omega)$.
#
# Even if both filters share the exact same order $M$ (and group delay $\tau = M/2$), their sum yields:
#
# $$H(e^{j\omega}) = \left[ A_1(\omega) + j A_3(\omega) \right] e^{-j\omega M/2}$$
#
# The complex amplitude term introduces a frequency-dependent phase component $\theta(\omega) = \arctan(A_3(\omega)/A_1(\omega))$, which destroys strict linear phase.
#
# -----------------
#
# Please give me instructor solutions as well
#
# Gemini Thimking:
#
# # Instructor Solution Manual & Grading Guide
#
# **Course:** Advanced Discrete-Time Signal Processing (EE 4xxx / 5xxx)
#
# **Topic:** Linear Phase FIR Systems (Types I–IV)
#
# **Total Points:** 100 Points
#
# ---
#
# ## Problem 1: Classification & Frequency Response Derivation (25 Points)
#
# ### Grading Rubric
#
# * **Part (a): 8 Points** (2 pts per sub-part: 1 pt for Type, 1 pt for correct $M$ and $N$).
# * **Part (b): 9 Points** (3 pts for algebraic expansion, 3 pts for grouping symmetric pairs, 3 pts for final $A(\omega)$ expression).
# * **Part (c): 8 Points** (3 pts for grouping antisymmetric pairs, 3 pts for extracting $j$, 2 pts for identifying the constant $+90^\circ$ phase shift).
#
# ---
#
# ### Expanded Solution & Mathematical Proofs
#
# #### Part (a) Classification Breakdown
#
# * **$h_1[n] = [1, 3, 5, 3, 1]$**
# * Length $N = 5$, Order $M = N - 1 = 4$ (Even).
# * Symmetry: $h[0]=h[4]=1$, $h[1]=h[3]=3$, $h[2]=5 \implies h[n] = h[M-n]$.
# * **Result: Type I** ($M$ even, symmetric).
#
#
# * **$h_2[n] = [2, -1, 4, 4, -1, 2]$**
# * Length $N = 6$, Order $M = 5$ (Odd).
# * Symmetry: $h[0]=h[5]=2$, $h[1]=h[4]=-1$, $h[2]=h[3]=4 \implies h[n] = h[M-n]$.
# * **Result: Type II** ($M$ odd, symmetric).
#
#
# * **$h_3[n] = [1, 2, 0, -2, -1]$**
# * Length $N = 5$, Order $M = 4$ (Even).
# * Symmetry: $h[0]=-h[4]=1$, $h[1]=-h[3]=2$, $h[2]=0 \implies h[n] = -h[M-n]$.
# * **Result: Type III** ($M$ even, antisymmetric).
#
#
# * **$h_4[n] = [-1, 3, -3, 1]$**
# * Length $N = 4$, Order $M = 3$ (Odd).
# * Symmetry: $h[0]=-h[3]=-1$, $h[1]=-h[2]=3 \implies h[n] = -h[M-n]$.
# * **Result: Type IV** ($M$ odd, antisymmetric).
#
#
#
# ---
#
# #### Part (b) Derivation for $h_2[n]$
#
# Start from the DTFT definition:
#
# $$H_2(e^{j\omega}) = \sum_{n=0}^{5} h_2[n] e^{-j\omega n} = 2 - e^{-j\omega} + 4e^{-j2\omega} + 4e^{-j3\omega} - e^{-j4\omega} + 2e^{-j5\omega}$$
#
# Factor out the linear phase term corresponding to group delay $\tau = M/2 = 2.5$:
#
# $$H_2(e^{j\omega}) = e^{-j2.5\omega} \left[ 2 e^{j2.5\omega} - e^{j1.5\omega} + 4 e^{j0.5\omega} + 4 e^{-j0.5\omega} - e^{-j1.5\omega} + 2 e^{-j2.5\omega} \right]$$
#
# Group complex conjugate exponentials:
#
# $$H_2(e^{j\omega}) = e^{-j2.5\omega} \left[ 2\left(e^{j2.5\omega} + e^{-j2.5\omega}\right) - \left(e^{j1.5\omega} + e^{-j1.5\omega}\right) + 4\left(e^{j0.5\omega} + e^{-j0.5\omega}\right) \right]$$
#
# Apply Euler's identity $\cos(\theta) = \frac{e^{j\theta} + e^{-j\theta}}{2}$:
#
# $$H_2(e^{j\omega}) = \left[ 4\cos(2.5\omega) - 2\cos(1.5\omega) + 8\cos(0.5\omega) \right] e^{-j 2.5\omega}$$
#
# Thus, the real amplitude response is:
#
# $$A(\omega) = 8\cos(0.5\omega) - 2\cos(1.5\omega) + 4\cos(2.5\omega)$$
#
# ---
#
# #### Part (c) Derivation for $h_3[n]$
#
# Start from the DTFT definition with $M = 4$:
#
# $$H_3(e^{j\omega}) = 1 + 2e^{-j\omega} + 0e^{-j2\omega} - 2e^{-j3\omega} - e^{-j4\omega}$$
#
# Factor out the group delay term $e^{-j2\omega}$:
#
# $$H_3(e^{j\omega}) = e^{-j2\omega} \left[ \left(e^{j2\omega} - e^{-j2\omega}\right) + 2\left(e^{j\omega} - e^{-j\omega}\right) \right]$$
#
# Apply Euler's identity $\sin(\theta) = \frac{e^{j\theta} - e^{-j\theta}}{2j} \implies e^{j\theta} - e^{-j\theta} = 2j\sin(\theta)$:
#
# $$H_3(e^{j\omega}) = e^{-j2\omega} \left[ 2j\sin(2\omega) + 4j\sin(\omega) \right] = j \left[ 4\sin(\omega) + 2\sin(2\omega) \right] e^{-j2\omega}$$
#
# Substitute $j = e^{j\pi/2}$:
#
# $$H_3(e^{j\omega}) = \left[ 4\sin(\omega) + 2\sin(2\omega) \right] e^{-j(2\omega - \pi/2)} = A(\omega) e^{-j2\omega} e^{j\pi/2}$$
#
# > **Key Pedagogical Note:** Students must explicitly state that the $j$ factor represents a constant quadrature phase shift of $+\pi/2$ radians ($+90^\circ$) superposed on the linear phase delay line $-\omega M/2$.
#
# ---
#
# ### Common Student Misconceptions
#
# * Confusing sequence length $N$ with filter order $M$ ($M = N - 1$).
# * Forgetting that $h_3[M/2]$ must be 0 for Type III systems (since $h[2] = -h[2] \implies h[2] = 0$).
#
# ---
#
# ## Problem 2: Mandatory Zero Constraints & Filter Approximations (25 Points)
#
# ### Grading Rubric
#
# * **Part (a): 8 Points** (4 pts for Type II proof, 4 pts for Type III proof).
# * **Part (b): 8 Points** (4 pts for Type III proof, 4 pts for Type IV proof).
# * **Part (c): 9 Points** (0.375 pts per cell across the 24 matrix entries; full credit requires correct justification for infeasible entries).
#
# ---
#
# ### Expanded Solution & Mathematical Proofs
#
# #### Part (a) Proof that $H(-1) = 0$ for Types II and III
#
# The transfer function evaluated at $z = -1 = e^{j\pi}$ is:
#
# $$H(-1) = \sum_{n=0}^{M} h[n] (-1)^{-n} = \sum_{n=0}^{M} h[n] (-1)^n$$
#
# * **Type II Proof ($h[n] = h[M-n]$ with $M$ odd):**
# Split the summation into the lower half $n = 0, \dots, (M-1)/2$:
# $$H(-1) = \sum_{n=0}^{(M-1)/2} \left( h[n](-1)^n + h[M-n](-1)^{M-n} \right)$$
#
#
# Substitute $h[M-n] = h[n]$ and $(-1)^{M-n} = (-1)^M (-1)^{-n} = -(-1)^n$ (since $M$ is odd):
# $$H(-1) = \sum_{n=0}^{(M-1)/2} \left( h[n](-1)^n - h[n](-1)^n \right) = 0$$
#
#
# * **Type III Proof ($h[n] = -h[M-n]$ with $M$ even):**
# The center sample at $n = M/2$ must satisfy $h[M/2] = -h[M/2] \implies h[M/2] = 0$.
# Split the remaining sum into paired terms:
# $$H(-1) = \sum_{n=0}^{(M/2)-1} \left( h[n](-1)^n + h[M-n](-1)^{M-n} \right) + h[M/2](-1)^{M/2}$$
#
#
# Substitute $h[M-n] = -h[n]$ and $(-1)^{M-n} = (-1)^M (-1)^{-n} = (-1)^n$ (since $M$ is even):
# $$H(-1) = \sum_{n=0}^{(M/2)-1} \left( h[n](-1)^n - h[n](-1)^n \right) + 0 = 0$$
#
#
#
# ---
#
# #### Part (b) Proof that $H(1) = 0$ for Types III and IV
#
# The transfer function evaluated at $z = 1 = e^{j0}$ is:
#
# $$H(1) = \sum_{n=0}^{M} h[n]$$
#
# * **Type III Proof ($h[n] = -h[M-n]$ with $M$ even):**
# $$H(1) = \sum_{n=0}^{(M/2)-1} \left( h[n] + h[M-n] \right) + h[M/2] = \sum_{n=0}^{(M/2)-1} \left( h[n] - h[n] \right) + 0 = 0$$
#
#
# * **Type IV Proof ($h[n] = -h[M-n]$ with $M$ odd):**
# $$H(1) = \sum_{n=0}^{(M-1)/2} \left( h[n] + h[M-n] \right) = \sum_{n=0}^{(M-1)/2} \left( h[n] - h[n] \right) = 0$$
#
#
#
# ---
#
# #### Part (c) Detailed Table Explanations
#
# | Application | Type I | Type II | Type III | Type IV |
# | --- | --- | --- | --- | --- |
# | **Lowpass Filter** | **Feasible** | **Feasible** | **Infeasible** ($H(0)=0$) | **Infeasible** ($H(0)=0$) |
# | **Highpass Filter** | **Feasible** | **Infeasible** ($H(\pi)=0$) | **Infeasible** ($H(\pi)=0$) | **Feasible** |
# | **Bandpass Filter** | **Feasible** | **Feasible** | **Feasible** | **Feasible** |
# | **Bandstop Filter** | **Feasible** | **Infeasible** ($H(\pi)=0$) | **Infeasible** ($H(0)=H(\pi)=0$) | **Infeasible** ($H(0)=0$) |
# | **Differentiator** | **Infeasible** ($\text{No } 90^\circ \text{ shift}$) | **Infeasible** ($H(\pi)=0$) | **Infeasible** ($H(\pi)=0$) | **Feasible** |
# | **Hilbert Transformer** | **Infeasible** ($\text{No } 90^\circ \text{ shift}$) | **Infeasible** ($H(\pi)=0$) | **Feasible** ($\text{BPF behavior}$) | **Feasible** |
#
# > **Instructor Notes for Marking Part (c):**
# > * **Wideband Differentiator:** Ideal response $H_d(e^{j\omega}) = j\omega$ for $\omega \in (-\pi, \pi)$. Requires linear gain in $\omega$ and a constant $+\pi/2$ phase shift (antisymmetric Types III/IV). Since an ideal differentiator needs $H_d(e^{j\pi}) = j\pi \neq 0$, Type III is infeasible because $H(\pi)=0$. Type IV is the standard choice.
# > * **Hilbert Transformer:** Ideal response $H_{HT}(e^{j\omega}) = -j \text{sgn}(\omega)$. Requires constant magnitude across all frequencies and a $90^\circ$ phase shift. Type III is feasible for bandpass Hilbert transformers, while Type IV works across wideband up to $\omega = \pi$.
# > 
# > 
#
# ---
#
# ## Problem 3: Zero Constellation Symmetry & System Synthesis (25 Points)
#
# ### Grading Rubric
#
# * **Part (a): 8 Points** (4 pts for conjugate symmetry proof, 4 pts for reciprocal symmetry proof).
# * **Part (b-i): 4 Points** (1 pt per correct root in polar/Cartesian form).
# * **Part (b-ii): 8 Points** (3 pts for quadratic factor formation, 3 pts for expansion, 2 pts for finding $C = 4/9$).
# * **Part (b-iii): 5 Points** (Full credit for correct 5-point sequence $h[n]$).
#
# ---
#
# ### Expanded Solution & Mathematical Proofs
#
# #### Part (a) Proof of Quadruplet Symmetry
#
# Let $H(z) = \sum_{n=0}^{M} h[n] z^{-n}$ with real $h[n] \in \mathbb{R}$.
#
# 1. **Conjugate Symmetry:**
# $$[H(z)]^* = \left( \sum_{n=0}^{M} h[n] z^{-n} \right)^* = \sum_{n=0}^{M} h[n]^* (z^*)^{-n} = \sum_{n=0}^{M} h[n] (z^*)^{-n} = H(z^*)$$
#
#
# Therefore, if $H(z_0) = 0$, then $H(z_0^*)=0$.
# 2. **Reciprocal Symmetry:**
# For a linear phase filter with symmetry $h[n] = \pm h[M-n]$, write $H(z)$:
# $$H(z) = \sum_{n=0}^{M} \pm h[M-n] z^{-n}$$
#
#
# Let $k = M - n$:
# $$H(z) = \pm \sum_{k=0}^{M} h[k] z^{-(M-k)} = \pm z^{-M} \sum_{k=0}^{M} h[k] z^{k} = \pm z^{-M} H(z^{-1})$$
#
#
# If $z_0$ is a zero ($H(z_0) = 0$) and $z_0 \neq 0$:
# $$0 = H(z_0) = \pm z_0^{-M} H(z_0^{-1}) \implies H(z_0^{-1}) = 0$$
#
#
# 3. **Combined Quadruplet:**
# Applying both properties to $z_0 = r e^{j\theta}$:
# * $z_1 = r e^{j\theta}$
# * $z_2 = z_1^* = r e^{-j\theta}$
# * $z_3 = 1/z_1 = \frac{1}{r} e^{-j\theta}$
# * $z_4 = 1/z_1^* = \frac{1}{r} e^{j\theta}$
#
#
#
# ---
#
# #### Part (b) Step-by-Step Synthesis
#
# **Sub-part i: Root Evaluation**
#
# Given $z_1 = \frac{1}{2} e^{j\pi/3}$:
#
# * $z_1 = \frac{1}{2}\left(\cos\frac{\pi}{3} + j\sin\frac{\pi}{3}\right) = \frac{1}{4} + j\frac{\sqrt{3}}{4}$
# * $z_2 = z_1^* = \frac{1}{2} e^{-j\pi/3} = \frac{1}{4} - j\frac{\sqrt{3}}{4}$
# * $z_3 = \frac{1}{z_1} = 2 e^{-j\pi/3} = 1 - j\sqrt{3}$
# * $z_4 = \frac{1}{z_1^*} = 2 e^{j\pi/3} = 1 + j\sqrt{3}$
#
# **Sub-part ii: Polynomial Factorization & Normalization**
#
# Form quadratic factors with real coefficients:
#
# $$Q_1(z) = (1 - z_1 z^{-1})(1 - z_1^* z^{-1}) = 1 - 2\text{Re}(z_1) z^{-1} + \vert{}z_1\vert{}^2 z^{-2} = 1 - \frac{1}{2} z^{-1} + \frac{1}{4} z^{-2}$$
#
# $$Q_2(z) = (1 - z_3 z^{-1})(1 - z_4 z^{-1}) = 1 - 2\text{Re}(z_3) z^{-1} + \vert{}z_3\vert{}^2 z^{-2} = 1 - 2 z^{-1} + 4 z^{-2}$$
#
# Multiply $Q_1(z) \cdot Q_2(z)$:
#
# $$\begin{aligned} P(z) &= \left(1 - \frac{1}{2} z^{-1} + \frac{1}{4} z^{-2}\right) \left(1 - 2 z^{-1} + 4 z^{-2}\right) \\ &= 1 \cdot \left(1 - 2 z^{-1} + 4 z^{-2}\right) - \frac{1}{2} z^{-1} \left(1 - 2 z^{-1} + 4 z^{-2}\right) + \frac{1}{4} z^{-2} \left(1 - 2 z^{-1} + 4 z^{-2}\right) \\ &= 1 - 2 z^{-1} + 4 z^{-2} - \frac{1}{2} z^{-1} + z^{-2} - 2 z^{-3} + \frac{1}{4} z^{-2} - \frac{1}{2} z^{-3} + z^{-4} \\ &= 1 - \frac{5}{2} z^{-1} + \frac{21}{4} z^{-2} - \frac{5}{2} z^{-3} + z^{-4} \end{aligned}$$
#
# Set gain constant $C$ such that $H(1) = 1$:
#
# $$H(1) = C \cdot P(1) = C \cdot \left( 1 - \frac{5}{2} + \frac{21}{4} - \frac{5}{2} + 1 \right) = C \cdot \left( 2 - 5 + \frac{21}{4} \right) = C \cdot \frac{9}{4} = 1 \implies C = \frac{4}{9}$$
#
# Distribute $C = 4/9$:
#
# $$H(z) = \frac{4}{9} - \frac{10}{9} z^{-1} + \frac{21}{9} z^{-2} - \frac{10}{9} z^{-3} + \frac{4}{9} z^{-4}$$
#
# **Sub-part iii: Sequence Coefficients**
#
# $$h[0] = \frac{4}{9}, \quad h[1] = -\frac{10}{9}, \quad h[2] = \frac{21}{9}, \quad h[3] = -\frac{10}{9}, \quad h[4] = \frac{4}{9}$$
#
# ---
#
# ## Problem 4: System Interconnections & Composite Types (25 Points)
#
# ### Grading Rubric
#
# * **Part (a): 9 Points** (3 pts for order addition, 3 pts for overall phase shift algebra, 3 pts for final Type III classification).
# * **Part (b): 8 Points** (3 pts for parallel combination expression, 3 pts for non-linear phase term $\theta(\omega)$, 2 pts for concluding loss of linear phase).
# * **Part (c): 8 Points** (4 pts for complex combination $A_1(\omega) + j A_3(\omega)$, 4 pts for showing phase non-linearity).
#
# ---
#
# ### Expanded Solution & Mathematical Proofs
#
# #### Part (a) Cascade Configuration
#
# Given:
#
# * $H_A(z)$ is Type II: $M_A$ is odd, $h_A[n] = h_A[M_A - n] \implies H_A(e^{j\omega}) = A_A(\omega) e^{-j\omega M_A/2}$.
# * $H_B(z)$ is Type IV: $M_B$ is odd, $h_B[n] = -h_B[M_B - n] \implies H_B(e^{j\omega}) = j A_B(\omega) e^{-j\omega M_B/2}$.
#
# **Order Analysis:**
#
# $$M_C = M_A + M_B = \text{odd} + \text{odd} = \text{even}$$
#
# **Frequency Response:**
#
# $$H_C(e^{j\omega}) = H_A(e^{j\omega}) \cdot H_B(e^{j\omega}) = \left( A_A(\omega) e^{-j\omega M_A/2} \right) \cdot \left( j A_B(\omega) e^{-j\omega M_B/2} \right) = j \left[ A_A(\omega) A_B(\omega) \right] e^{-j\omega (M_A + M_B)/2}$$
#
# Let $A_C(\omega) = A_A(\omega) A_B(\omega)$ (real-valued function). Then:
#
# $$H_C(e^{j\omega}) = j A_C(\omega) e^{-j\omega M_C/2}$$
#
# An overall order $M_C$ that is **even** combined with a $j = e^{j\pi/2}$ factor (antisymmetric impulse response) uniquely defines a **Type III FIR System**.
#
# ---
#
# #### Part (b) Parallel Configuration
#
# With $M_A = M_B = M$ (both odd):
#
# $$H_D(e^{j\omega}) = H_A(e^{j\omega}) + H_B(e^{j\omega}) = A_A(\omega) e^{-j\omega M/2} + j A_B(\omega) e^{-j\omega M/2} = \left[ A_A(\omega) + j A_B(\omega) \right] e^{-j\omega M/2}$$
#
# Define $C(\omega) = A_A(\omega) + j A_B(\omega) = \vert{}C(\omega)\vert{} e^{j \theta(\omega)}$, where:
#
# $$\theta(\omega) = \arctan\left( \frac{A_B(\omega)}{A_A(\omega)} \right)$$
#
# The overall phase response is:
#
# $$\angle H_D(e^{j\omega}) = \theta(\omega) - \frac{M}{2}\omega$$
#
# Since $A_A(\omega)$ and $A_B(\omega)$ are distinct trigonometric polynomials, $\theta(\omega)$ is a non-linear function of $\omega$. Therefore, $H_D(e^{j\omega})$ does **not** possess generalized linear phase.
#
# ---
#
# #### Part (c) Parallel Type I + Type III Systems
#
# Let $H_1(z)$ be Type I with order $M_1$ and $H_3(z)$ be Type III with order $M_3$. Assume identical group delay $\tau = M_1/2 = M_3/2 \implies M_1 = M_3 = M$ (even).
#
# $$H_{13}(e^{j\omega}) = A_1(\omega) e^{-j\omega M/2} + j A_3(\omega) e^{-j\omega M/2} = \left[ A_1(\omega) + j A_3(\omega) \right] e^{-j\omega M/2}$$
#
# Similar to Part (b), the sum introduces a phase contribution:
#
# $$\phi(\omega) = \arctan\left( \frac{A_3(\omega)}{A_1(\omega)} \right)$$
#
# Because $A_3(\omega)$ contains sine terms and $A_1(\omega)$ contains cosine terms, their ratio varies continuously with frequency. The overall phase $\angle H_{13}(e^{j\omega}) = \phi(\omega) - \omega M/2$ is non-linear. Thus, **no linear phase system can result**.
#
# ---
#
# ## Python Verification Script
#
# Instructors and TAs can use this standalone Python script to verify all numeric outputs and system classifications.
#
# ```python
# import numpy as np
# import scipy.signal as signal
#
# def verify_problem_1():
#     h1 = np.array([1, 3, 5, 3, 1])
#     h2 = np.array([2, -1, 4, 4, -1, 2])
#     h3 = np.array([1, 2, 0, -2, -1])
#     h4 = np.array([-1, 3, -3, 1])
#     
#     print("--- Problem 1 Verification ---")
#     for idx, h in enumerate([h1, h2, h3, h4], 1):
#         M = len(h) - 1
#         is_sym = np.allclose(h, h[::-1])
#         is_antisym = np.allclose(h, -h[::-1])
#         
#         if is_sym and M % 2 == 0:
#             sys_type = "Type I"
#         elif is_sym and M % 2 != 0:
#             sys_type = "Type II"
#         elif is_antisym and M % 2 == 0:
#             sys_type = "Type III"
#         elif is_antisym and M % 2 != 0:
#             sys_type = "Type IV"
#         else:
#             sys_type = "Non-linear phase"
#             
#         print(f"h{idx}: Order M={M}, Length N={len(h)} -> {sys_type}")
#
# def verify_problem_3():
#     print("\n--- Problem 3 Verification ---")
#     z1 = 0.5 * np.exp(1j * np.pi / 3)
#     z2 = np.conj(z1)
#     z3 = 1.0 / z1
#     z4 = 1.0 / z2
#     
#     zeros = [z1, z2, z3, z4]
#     poly_coeffs = np.poly(zeros)
#     
#     # Normalize such that H(z=1) = 1
#     gain_at_z1 = np.sum(poly_coeffs)
#     h_seq = poly_coeffs / gain_at_z1
#     
#     print("Zeros (Cartesian):", [np.round(z, 4) for z in zeros])
#     print("Computed h[n]:", np.round(h_seq, 4))
#     print("Fractional h[n]: [4/9, -10/9, 21/9, -10/9, 4/9] = ", 
#           np.round([4/9, -10/9, 21/9, -10/9, 4/9], 4))
#
# if __name__ == "__main__":
#     verify_problem_1()
#     verify_problem_3()
#
# ```
#
# -------------------
#
#
