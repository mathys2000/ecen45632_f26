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
# # Week 2 easier
#
# **Week 2: Sampling Theory.** Continuous-Time to Discrete-Time (C/D) conversion, D/C conversion, Nyquist theorem, aliasing, and quantization noise.
#

# %% [markdown]
# Week 2 easier
#
# Can you map out the two 75-minute lectures for Week 2 covering Sampling, Aliasing, and C/D conversion? 
#
# Gemini Pro:
#
# Here is the detailed breakdown for the two 75-minute lectures in Week 2.
#
# This week bridges the physical world (continuous-time acoustic waves) and the digital domain (discrete-time arrays). It is vital to emphasize that any mistakes made during sampling (like aliasing) permanently destroy information, which will later ruin the performance of their machine learning models.
#

# %% [markdown]
# ## Lecture 3: Continuous-to-Discrete (C/D) Conversion & Aliasing
#
# **Objective:** Mathematically model the sampling process using impulse trains, derive the frequency-domain spectrum of a sampled signal, and prove the Nyquist-Shannon Sampling Theorem.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 15 min** | **The C/D Interface** | Motivation: Microphones capture continuous-time voltage $x_c(t)$, but PyTorch only accepts discrete arrays $x[n]$. Define the relationship $x[n] = x_c(nT)$. |
# | **15 - 35 min** | **Ideal Impulse Train Sampling** | Introduce the mathematical model. Multiply $x_c(t)$ by an impulse train $s(t) = \sum \delta(t - nT)$ to create the modulated signal $x_s(t)$. |
# | **35 - 55 min** | **Sampling in the Frequency Domain** | Derive the spectrum of $x_s(t)$. Show that multiplication in time is convolution in frequency, resulting in infinite, scaled copies of the baseband spectrum separated by $\Omega_s$. |
# | **55 - 75 min** | **Nyquist Theorem & Aliasing** | Define the Nyquist rate mathematically: $\Omega_s > 2\Omega_N$. Draw what happens on the board when $\Omega_s < 2\Omega_N$ (the spectral copies overlap). *Audio ML connection:* Show how aliased high-frequency noise creates false patterns in a spectrogram. |
#
# ### Key Equations to Board
#
# * **Impulse Train Sampling:**
#
# $$x_s(t) = x_c(t) \sum_{n=-\infty}^{\infty} \delta(t - nT) = \sum_{n=-\infty}^{\infty} x_c(nT)\delta(t - nT)$$
#
#
# * **Spectrum of Sampled Signal:**
#
# $$X_s(j\Omega) = \frac{1}{T} \sum_{k=-\infty}^{\infty} X_c(j(\Omega - k\Omega_s))$$
#
#
# * **Continuous vs. Discrete Frequency:**
#
# $$\omega = \Omega T$$
#
#
#
# ---
#

# %% [markdown]
# ## Lecture 4: Reconstruction, D/A, and Quantization Noise
#
# **Objective:** Explain the mathematics of ideal reconstruction (D/C conversion), the realities of physical Digital-to-Analog hardware, and model the effects of bit-depth (quantization) on audio quality.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 20 min** | **Ideal D/C Conversion** | Explain how to isolate the baseband spectrum using an ideal continuous-time low-pass filter $H_r(j\Omega)$. Derive the time-domain interpolation formula using the $\text{sinc}$ function. |
# | **20 - 40 min** | **Practical D/A: Zero-Order Hold** | The math of ideal $\text{sinc}$ interpolation is non-causal and infinitely long. Introduce the Zero-Order Hold (ZOH) circuit. Show how ZOH introduces a $\text{sinc}$-shaped envelope drop-off in the frequency domain. |
# | **40 - 55 min** | **Quantization & Bit-Depth** | Time is discrete, but amplitude must also be discretized. Model quantization as an additive, wide-sense stationary noise source $e[n]$. Derive the $6.02 \text{ dB}$ per bit rule. |
# | **55 - 75 min** | **Live Audio Demo** | Take a clean vocal recording (24-bit, 48 kHz). Decimate it without an anti-aliasing filter to hear aliasing (sounds like robotic ringing). Then reduce it to 8-bit and 4-bit to hear quantization noise (sounds like loud white noise/fuzz). |
#
# ### Key Equations to Board
#
# * **Ideal Bandlimited Interpolation:**
#
# $$x_r(t) = \sum_{n=-\infty}^{\infty} x[n] \frac{\sin(\pi(t-nT)/T)}{\pi(t-nT)/T}$$
#
#
# * **Quantization Noise Model:**
#
# $$x_q[n] = x[n] + e[n]$$
#
#
# * **Signal-to-Quantization-Noise Ratio (SQNR):**
#
# $$\text{SQNR} \approx 6.02B + 1.76 \text{ dB}$$
#
#
#
# (where $B$ is the number of bits).
#
# > **Instructor Tip:** When discussing quantization, explicitly connect $B$ (bit depth) to data types they will use in PyTorch. For instance, explain why raw audio is usually 16-bit integer (giving $\sim 96 \text{ dB}$ of dynamic range), but neural networks require casting these to 32-bit floats.
#
# ---
#

# %% [markdown]
#
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# Chat GPT:
#
# Done. I created the complete **Week 2 DSP + ML teaching package** for the two 75-minute classes.
#
# ### Complete package
#
# [Download the complete Week 2 package](sandbox:/mnt/data/week2_dsp_ml/Week2_DSP_ML_Complete_Package.zip)
#
# ### Individual files
#
# * [Lecture 3 — Continuous-to-Discrete Conversion & Aliasing](sandbox:/mnt/data/week2_dsp_ml/Week2_Lecture3_Sampling_Aliasing.pptx) — 22 slides with speaker notes and timing guidance.
# * [Lecture 4 — Reconstruction, D/A, and Quantization Noise](sandbox:/mnt/data/week2_dsp_ml/Week2_Lecture4_Reconstruction_Quantization.pptx) — 24 slides with speaker notes and the live-demo sequence.
# * [Week 2 Jupyter Notebook](sandbox:/mnt/data/week2_dsp_ml/Week2_DSP_ML_Code.ipynb) — executed notebook with NumPy, SciPy, Matplotlib, PyTorch, spectrograms, and playable audio.
# * [Standalone Python Code](sandbox:/mnt/data/week2_dsp_ml/Week2_DSP_ML_Code.py)
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week2_dsp_ml/Week2_Homework_Quiz_Solutions.docx)
#
# ### Audio files for the Lecture 4 demo
#
# * [Clean 48 kHz reference](sandbox:/mnt/data/week2_dsp_ml/audio/01_clean_48k.wav)
# * [Naive 48 → 12 kHz decimation with aliasing](sandbox:/mnt/data/week2_dsp_ml/audio/02_decimated_naive_12k_alias.wav)
# * [Anti-aliased 48 → 12 kHz decimation](sandbox:/mnt/data/week2_dsp_ml/audio/03_decimated_antialias_12k.wav)
# * [8-bit quantized audio](sandbox:/mnt/data/week2_dsp_ml/audio/04_quantized_8bit.wav)
# * [4-bit quantized audio](sandbox:/mnt/data/week2_dsp_ml/audio/05_quantized_4bit.wav)
#
# A few teaching choices I built in deliberately:
#
# **Lecture 3 treats aliasing as a non-uniqueness problem**, not merely as vague “distortion.” Students see mathematically that a 2 kHz and 10 kHz cosine sampled at 12 kHz generate identical samples. That makes the argument that ML cannot magically recover lost information particularly strong.
#
# The spectral-replica derivation goes all the way through
#
#
# $$S(j\Omega)=\frac{2\pi}{T}\sum_k\delta(\Omega-k\Omega_s)$$
#
#
# to
#
#
# $$X_s(j\Omega) = 
# \frac{1}{T}
# \sum_k X_c\left(j(\Omega-k\Omega_s)\right),$$
#
#
# so the Nyquist condition follows geometrically rather than appearing as a rule to memorize.
#
# **Lecture 4 derives both sinc reconstruction and the ZOH response.** The ZOH section shows why
#
#
# $$H_{\mathrm{ZOH}}(j\Omega) =
# T e^{-j\Omega T/2}
# \frac{\sin(\Omega T/2)}{\Omega T/2}$$
#
#
# produces the familiar (-3.92) dB droop at Nyquist.
#
# I also included a distinction that is often confusing in audio courses: the roughly **96 dB “6 dB/bit” figure for 16 bits versus the 98.08 dB full-scale-sine SQNR** obtained from (6.02B+1.76).
#
# The PyTorch material explicitly emphasizes that converting `int16` PCM to `float32` **changes the computational representation but does not recover quantization precision or undo aliasing**.
#
# The homework contains **8 core problems plus a graduate stretch problem**, and the weekly quiz is a **20-point, approximately 15-minute assessment** with a grading guide. The notebook was executed successfully, both slide decks were rendered and checked for overflow, and the homework/quiz document was rendered page-by-page for layout verification.
#
# -------------------------------------
#

# %% [markdown]
# # Week 2, Lecture 3, LaTeX math
#
# Here is the lecture content converted with properly formatted LaTeX for all the mathematical expressions, organized for clear readability:
#
# This lecture addresses a fundamental question in signal acquisition:
#
# $$\boxed{\text{When do samples contain enough information to uniquely represent the original analog signal?}}$$
#
# The core mathematical progression flows as follows:
#
# $$\boxed{x_c(t) \rightarrow \text{impulse-train sampling} \rightarrow X_s(j\Omega) \rightarrow \text{spectral replicas} \rightarrow \text{Nyquist condition} \rightarrow \text{aliasing if replicas overlap}}$$
#
# Framing sampling as an **information-preservation problem** rather than merely taking measurements every $T$ seconds highlights a critical machine learning consequence: if aliasing introduces false spectral structure before the data reaches a model, the model cannot distinguish that artificial structure from authentic acoustic features.
#
# ---
#
# ## 1. The C/D Interface: What Actually Changes When We Sample?
#
# We begin with a continuous-time signal $x_c(t)$. For audio applications, this represents continuous physical phenomena such as microphone voltage.
#
# Sampling every $T$ seconds yields:
#
# $$\boxed{x[n]=x_c(nT)}$$
#
# where
#
# $$f_s = \frac{1}{T}$$
#
# Key variables:
#
# * $t$: Continuous time variable (seconds)
# * $n$: Discrete integer sample index
# * $T$: Sampling period (seconds/sample)
# * $f_s$: Sampling rate (samples/second)
#
# Sampling discretizes **time**, whereas quantization discretizes **amplitude**.
#
# ### Example 1
#
# At $f_s = 48\text{ kHz}$, what physical time corresponds to sample $n = 2400$?
#
# $$t = \frac{2400}{48000} = 0.05\text{ s} \implies \boxed{50\text{ ms}}$$
#
# ### Example 2
#
# Does ideal sampling change amplitude values?
#
# * **Answer:** No. Ideal sampling merely evaluates the continuous-time function at discrete points. Amplitude restriction occurs later during quantization.
#
# ---
#
# ## 2. Continuous Frequency vs. Discrete Frequency
#
# A continuous-time sinusoid is represented as:
#
# $$x_c(t) = \cos(\Omega_0 t)$$
#
# where $\Omega_0$ is continuous angular frequency in rad/s. Sampling yields:
#
# $$x[n] = \cos(\Omega_0 n T)$$
#
# Defining the normalized discrete-time frequency $\omega_0$:
#
# $$\boxed{\omega_0 = \Omega_0 T}$$
#
# gives:
#
# $$x[n] = \cos(\omega_0 n)$$
#
# Here, $\omega$ carries units of rad/sample and is unique only modulo $2\pi$. Because $\Omega = 2\pi f$, the relationship simplifies to:
#
# $$\boxed{\omega = 2\pi\frac{f}{f_s}}$$
#
# ---
#
# ## 3. Normalized Frequency Mapping
#
# At $f_s = 12\text{ kHz}$, the normalized discrete-time frequency corresponding to a 3 kHz sinusoid is:
#
# $$\omega = 2\pi\frac{3}{12} = \frac{\pi}{2}$$
#
# For a 15 kHz sinusoid at the same sample rate:
#
# $$\omega = 2\pi\frac{15}{12} = \frac{5\pi}{2}$$
#
# Evaluating modulo $2\pi$:
#
# $$\frac{5\pi}{2} \equiv \frac{\pi}{2}$$
#
# Thus, 3 kHz and 15 kHz map to the exact same discrete-time complex exponential when sampled at 12 kHz.
#
# ---
#
# ## 4. The Impulse Train as an Analytical Tool
#
# Ideal impulse-train sampling models physical sampling mathematically. Defining the periodic impulse train:
#
# $$s(t) = \sum_{n=-\infty}^{\infty} \delta(t-nT)$$
#
# The sampled continuous-time signal $x_s(t)$ is:
#
# $$x_s(t) = x_c(t) s(t) = x_c(t) \sum_{n=-\infty}^{\infty} \delta(t-nT)$$
#
# Applying the sifting property $x_c(t)\delta(t-nT) = x_c(nT)\delta(t-nT)$:
#
# $$\boxed{x_s(t) = \sum_{n=-\infty}^{\infty} x_c(nT)\delta(t-nT)}$$
#
# Each impulse location marks a sample instant, with the discrete sequence $x[n]$ embedded as the weights of the impulses.
#
# ---
#
# ## 5. Sample Weight Extraction
#
# Consider $x_c(t) = \cos(2\pi \cdot 1000 t)$ sampled at $f_s = 8000\text{ Hz}$ ($T = \frac{1}{8000}\text{ s}$).
#
# The impulse weights evaluate to:
#
# $$x_c(nT) = \cos\left(2\pi \cdot 1000 \frac{n}{8000}\right) = \cos\left(\frac{\pi n}{4}\right)$$
#
# Yielding the impulse-train representation:
#
# $$x_s(t) = \sum_{n=-\infty}^{\infty} \cos\left(\frac{\pi n}{4}\right) \delta(t-nT)$$
#
# The underlying discrete sequence is extracted directly from these weights: $x[n] = \cos\left(\frac{\pi n}{4}\right)$.
#
# ---
#
# ## 6. Frequency Representation of the Impulse Train
#
# A periodic impulse train in the time domain transforms into an impulse train in the frequency domain. For $s(t) = \sum_{n=-\infty}^{\infty} \delta(t-nT)$, its Continuous-Time Fourier Transform (CTFT) is:
#
# $$\boxed{S(j\Omega) = \frac{2\pi}{T} \sum_{k=-\infty}^{\infty} \delta(\Omega - k\Omega_s)}$$
#
# where
#
# $$\boxed{\Omega_s = \frac{2\pi}{T} = 2\pi f_s}$$
#
# This relationship ($T \longleftrightarrow \Omega_s = \frac{2\pi}{T}$) causes spectral replication in the frequency domain.
#
# ---
#
# ## 7. Derivation of Spectral Replicas
#
# Because multiplication in time corresponds to convolution in frequency:
#
# $$X_s(j\Omega) = \frac{1}{2\pi} X_c(j\Omega) * S(j\Omega)$$
#
# Substituting $S(j\Omega)$:
#
# $$X_s(j\Omega) = \frac{1}{2\pi} X_c(j\Omega) * \left[ \frac{2\pi}{T} \sum_{k=-\infty}^{\infty} \delta(\Omega - k\Omega_s) \right]$$
#
# Simplifying:
#
# $$X_s(j\Omega) = \frac{1}{T} X_c(j\Omega) * \sum_{k=-\infty}^{\infty} \delta(\Omega - k\Omega_s)$$
#
# Since convolution with a shifted Dirac impulse shifts the spectrum $X_c(j\Omega) * \delta(\Omega - k\Omega_s) = X_c(j(\Omega - k\Omega_s))$:
#
# $$\boxed{X_s(j\Omega) = \frac{1}{T} \sum_{k=-\infty}^{\infty} X_c(j(\Omega - k\Omega_s))}$$
#
# Sampling replicates the original continuous spectrum $X_c(j\Omega)$ at integer multiples of the sampling frequency $\Omega_s$, scaled by $\frac{1}{T}$.
#
# ---
#
# ## 8. Spectral Overlay Visualization
#
# Assuming $X_c(j\Omega)$ is bandlimited to $\Omega_M$ ($X_c(j\Omega) = 0$ for $\vert{}\Omega\vert{} > \Omega_M$):
#
# ```text
#              ______
#             /      \
# -----------/        \-----------
#         -ΩM     0      ΩM
#
# ```
#
# Sampling produces infinitely many shifted copies centered at $0, \pm\Omega_s, \pm2\Omega_s, \ldots$:
#
# ```text
#       copy             baseband              copy
#        /\                 /\                  /\
# ------/  \---------------/  \----------------/  \------
#    -Ωs                 0                    Ωs
#
# ```
#
# As long as neighboring spectral replicas do not overlap, the baseband signal remains intact.
#
# ---
#
# ## 9. Derivation of the Nyquist Condition
#
# The baseband spectrum occupies the range $-\Omega_M \le \Omega \le \Omega_M$ (total width $2\Omega_M$).
#
# The adjacent positive replica is centered at $\Omega_s$, extending down to its left edge at $\Omega_s - \Omega_M$. To prevent spectral overlap:
#
# $$\Omega_s - \Omega_M > \Omega_M \implies \Omega_s > 2\Omega_M$$
#
# Expressed in Hz:
#
# $$\boxed{f_s > 2f_M}$$
#
# When this condition holds, an ideal low-pass filter with cutoff frequency $\Omega_c = \frac{\Omega_s}{2}$ isolates the original spectrum $X_c(j\Omega)$ perfectly.
#
# ---
#
# ## 10. Nyquist Frequency vs. Nyquist Rate
#
# * **Nyquist Frequency ($f_N$):** A property of the **system**. Given sampling rate $f_s$:
#
# $$\boxed{f_N = \frac{f_s}{2}}$$
#
#
#
# *(e.g., $f_s = 48\text{ kHz} \implies f_N = 24\text{ kHz}$)*
# * **Nyquist Rate:** A property of the **signal**. Given maximum signal frequency $f_M$:
#
# $$\boxed{\text{Nyquist Rate} = 2f_M}$$
#
#
#
# *(e.g., $f_M = 18\text{ kHz} \implies \text{Nyquist Rate} = 36\text{ kHz}$)*
#
# ---
#
# ## 11. Practical Guard Bands
#
# While theoretical derivations state $f_s \ge 2f_M$, physical systems cannot construct ideal rectangular brick-wall filters. Practical anti-aliasing filters require a finite **transition band**:
#
# $$f_M < \frac{f_s}{2}$$
#
# Physical analog front-ends need operating margin between the maximum signal frequency and $f_N$.
#
# ---
#
# ## 12. Defining Aliasing
#
# $$\boxed{\text{Aliasing is non-uniqueness.}}$$
#
# When spectral replicas overlap, higher-frequency components fold into lower frequencies. Aliasing is non-invertible distortion; it creates deterministic, false sinusoidal components in the passband that are identical to real lower-frequency signals.
#
# ---
#
# ## 13. Indistinguishable Discrete-Time Sinusoids
#
# Consider $f_s = 12\text{ kHz}$ comparing $x_1(t) = \cos(2\pi \cdot 2000 t)$ and $x_2(t) = \cos(2\pi \cdot 10000 t)$.
#
# Sampling $x_1(t)$:
#
# $$x_1[n] = \cos\left(2\pi\frac{2000}{12000}n\right) = \cos\left(\frac{\pi n}{3}\right)$$
#
# Sampling $x_2(t)$:
#
# $$x_2[n] = \cos\left(2\pi\frac{10000}{12000}n\right) = \cos\left(\frac{5\pi n}{3}\right)$$
#
# Using periodicity $\frac{5\pi}{3} = 2\pi - \frac{\pi}{3}$ and cosine symmetry:
#
# $$\cos\left(\frac{5\pi n}{3}\right) = \cos\left(2\pi n - \frac{\pi n}{3}\right) = \cos\left(-\frac{\pi n}{3}\right) = \cos\left(\frac{\pi n}{3}\right)$$
#
# $$\boxed{x_1[n] = x_2[n]}$$
#
# Both continuous-time signals produce identical digital samples.
#
# ---
#
# ## 14. Irreversibility
#
# Once sampled into $x[n] = \cos\left(\frac{\pi n}{3}\right)$, all historic context regarding whether the source frequency was 2 kHz or 10 kHz is lost. The sample sequence contains no metadata identifying its original continuous-time frequency.
#
# ---
#
# ## 15. The Folded Frequency Formula
#
# For a continuous sinusoid of frequency $f$ sampled at $f_s$, the observed alias frequency $f_{\text{alias}}$ in the baseband $0 \le f_{\text{alias}} \le \frac{f_s}{2}$ satisfies:
#
# $$f_{\text{alias}} = \vert{}f - k f_s\vert{} \quad \text{for } k \in \mathbb{Z}$$
#
# For $f_s = 12\text{ kHz}$ ($f_N = 6\text{ kHz}$):
#
# * **7 kHz:** $\vert{}7 - 12\vert{} = 5\text{ kHz} \implies \boxed{7\text{ kHz} \rightarrow 5\text{ kHz}}$
# * **9 kHz:** $\vert{}9 - 12\vert{} = 3\text{ kHz} \implies \boxed{9\text{ kHz} \rightarrow 3\text{ kHz}}$
# * **10 kHz:** $\vert{}10 - 12\vert{} = 2\text{ kHz} \implies \boxed{10\text{ kHz} \rightarrow 2\text{ kHz}}$
# * **13 kHz:** $\vert{}13 - 12\vert{} = 1\text{ kHz} \implies \boxed{13\text{ kHz} \rightarrow 1\text{ kHz}}$
#
# ---
#
# ## 16. Visualizing Spectral Folding
#
# At $f_s = 12\text{ kHz}$, frequencies reflect symmetrically around the Nyquist boundary $f_N = 6\text{ kHz}$:
#
# | Input $f$ (kHz) | Folded $f_{\text{alias}}$ (kHz) |
# | --- | --- |
# | 0 | 0 |
# | 2 | 2 |
# | 6 | 6 |
# | 7 | 5 |
# | 10 | 2 |
# | 12 | 0 |
#
# ---
#
# ## 17. Normalized Frequency Perspective
#
# Re-evaluating the 10 kHz input at $f_s = 12\text{ kHz}$ in normalized frequency:
#
# $$\omega = 2\pi\frac{10}{12} = \frac{5\pi}{3}$$
#
# Modular reduction yields:
#
# $$\frac{5\pi}{3} \equiv -\frac{\pi}{3} \pmod{2\pi}$$
#
# Taking the real part for a cosine:
#
# $$\cos\left(-\frac{\pi}{3}n\right) = \cos\left(\frac{\pi}{3}n\right)$$
#
# Converting back to linear frequency in Hz:
#
# $$f = \frac{\pi/3}{2\pi} \cdot 12000 = 2000\text{ Hz}$$
#
# ---
#
# ## 18. Pre-ADC Filtering Requirements
#
# If an analog input contains a desired 2 kHz component alongside an unwanted 10 kHz component, sampling at $f_s = 12\text{ kHz}$ maps both to 2 kHz:
#
# $$x[n] = A_1\cos\left(2\pi\frac{2000}{f_s}n\right) + A_2\cos\left(2\pi\frac{2000}{f_s}n\right) = (A_1 + A_2)\cos\left(2\pi\frac{2000}{f_s}n\right)$$
#
# Because the terms combine linearly into a single 2 kHz discrete component, no digital filter can separate them post-sampling. The anti-aliasing filter must exist in the **analog domain prior to sampling**:
#
# $$\boxed{\text{Sensor} \rightarrow \text{Analog Anti-Alias Filter} \rightarrow \text{ADC} \rightarrow \text{Digital Processing}}$$
#
# ---
#
# ## 19. Digital Inability to Fix Aliasing
#
# Digital post-processing cannot fix aliasing because violation of the Nyquist criterion represents a many-to-one mapping. Overlapping spectral content destroys linear independence between baseband and alias frequencies.
#
# ---
#
# ## 20. Downsampling / Decimation
#
# Downsampling an already digitized signal requires pre-filtering. Converting an audio stream from 48 kHz to 12 kHz reduces the Nyquist frequency from 24 kHz to 6 kHz.
#
# $$\boxed{\text{Low-pass filter first, decimate second.}}$$
#
# * Direct slicing (`x[::4]`) without filtering causes high frequencies (6–24 kHz) to alias into the 0–6 kHz band.
# * Polyphase filtering (`scipy.signal.resample_poly`) suppresses content above 6 kHz prior to decimation.
#
# ---
#
# ## 21. Mechanism of Naive Decimation Aliasing
#
# Consider a 10 kHz component in a 48 kHz signal ($f_N = 24\text{ kHz}$). Selecting every 4th sample drops $f_s$ to 12 kHz ($f_N' = 6\text{ kHz}$). The 10 kHz component now exceeds $f_N'$ and folds:
#
# $$\vert{}10 - 12\vert{} = 2\text{ kHz} \implies \boxed{10\text{ kHz} \rightarrow 2\text{ kHz}}$$
#
# ---
#
# ## 22. Proper Decimation Trade-offs
#
# Proper decimation strips energy above the target Nyquist rate (6 kHz) using a digital low-pass filter before sample removal. Information above 6 kHz is lost by design, avoiding false low-frequency artifacts.
#
# ---
#
# ## 23. Downstream Machine Learning Vulnerabilities
#
# If audio containing an un-filtered 10 kHz tone undergoes naive decimation, a 2 kHz line appears in the resulting spectrogram.
#
# $$\boxed{\text{Strong energy at 2 kHz}}$$
#
# A convolutional neural network (CNN) trained on these features will treat this line as valid 2 kHz acoustic data, potentially causing the model to learn hardware-dependent sampling artifacts rather than true physical class features.
#
# ---
#
# ## 24. Dataset Shortcuts
#
# In machine learning, if Class A recordings originate from an ADC with poor anti-alias filtering while Class B recordings do not, a CNN may achieve 100% training accuracy strictly by classifying hardware artifacts. When deployed on standard hardware, performance collapses.
#
# ---
#
# ## 25. Application Problem 1
#
# > **Scenario:** A sensor signal is bandlimited to 18 kHz. Is $f_s = 32\text{ kHz}$ sufficient?
#
# * **Calculation:** $\text{Nyquist Rate} = 2 f_M = 2(18) = 36\text{ kHz}$.
# * **Evaluation:** $32\text{ kHz} < 36\text{ kHz}$.
# * **Verdict:** $\boxed{\text{No.}}$ The replicas will overlap because $f_N = 16\text{ kHz} < 18\text{ kHz}$.
#
# ---
#
# ## 26. Application Problem 2
#
# > **Scenario:** A 17 kHz tone is sampled at $f_s = 20\text{ kHz}$. What frequency appears in the output?
#
# * **Calculation:** $f_N = 10\text{ kHz}$.
# * **Alias Calculation:** $\vert{}17 - 20\vert{} = 3\text{ kHz}$.
# * **Normalized Frequency Verification:**
#
# $$\omega = 2\pi\frac{17}{20} = 1.7\pi \equiv -0.3\pi \pmod{2\pi}$$
#
#
# $$f = \frac{0.3\pi}{2\pi}(20\text{ kHz}) = \boxed{3\text{ kHz}}$$
#
#
#
# ---
#
# ## 27. Application Problem 3
#
# > **Scenario:** Why can't a digital low-pass filter placed after the ADC remove aliasing?
#
# Once sampled, the aliased component and legitimate in-band signal at the same frequency merge into a single sequence value. Information distinguishing their origins is destroyed.
#
# ---
#
# ## 28. Ultrasonic Interference
#
# Even if useful acoustic signals top out at 20 kHz and $f_s = 48\text{ kHz}$ ($f_N = 24\text{ kHz}$), out-of-band ultrasonic noise, radio frequency demodulation, or circuit harmonics above 24 kHz can still alias into the audible spectrum without an analog anti-aliasing filter.
#
# ---
#
# ## 29. Practical Bandlimiting
#
# Theoretical signals satisfy $X_c(j\Omega) = 0$ for $\vert{}\Omega\vert{} > \Omega_M$. Real physical signals are rarely strictly bandlimited, requiring analog filtering to attenuate out-of-band energy below the dynamic range floor of the ADC.
#
# ---
#
# ## 30. Band Support Boundaries
#
# Consider a continuous signal with spectral support in the range $[-4\text{ kHz}, +4\text{ kHz}]$.
#
# * **Case A ($f_s = 12\text{ kHz}$):** Replicas center at $0, \pm 12\text{ kHz}$.
# * Baseband: $[-4, +4]\text{ kHz}$
# * First replica: $[8, 16]\text{ kHz}$
# * *Result:* No spectral overlap.
#
#
# * **Case B ($f_s = 6\text{ kHz}$):** Replicas center at $0, \pm 6\text{ kHz}$.
# * Baseband: $[-4, +4]\text{ kHz}$
# * First replica: $[2, 10]\text{ kHz}$
# * *Result:* Overlap occurs in the $[2, 4]\text{ kHz}$ region.
#
#
#
# ---
#
# ## 31. Single Sinusoid Ambiguity
#
# $$\boxed{\text{Aliasing is irreversible from the samples alone.}}$$
#
# Without external prior assumptions regarding signal bandlimits, an infinite set of continuous-time frequencies $\{\vert{}f_{\text{alias}} \pm k f_s\vert{}\}$ maps to the exact same sample sequence.
#
# ---
#
# ## 32. Physical ADC Architecture
#
# A complete continuous-to-digital hardware chain contains distinct functional blocks:
#
# $$\text{Analog Input} \rightarrow \text{Anti-Alias Filter} \rightarrow \text{Sample-and-Hold} \rightarrow \text{Quantizer} \rightarrow \text{Digital Code}$$
#
# ---
#
# ## 33. Aliasing vs. Quantization Noise
#
# * **Aliasing:** Caused by under-sampling in time ($f_s < 2f_M$). Rotates high-frequency energy into false lower frequencies.
#
# $$\boxed{\text{Aliasing changes frequency identity.}}$$
#
#
# * **Quantization:** Caused by finite bit-depth resolution in amplitude. Modeled as additive noise $e[n] = x_q[n] - x[n]$.
#
# $$\boxed{\text{Quantization introduces amplitude error.}}$$
#
#
#
# ---
#
# ## 34. Checkpoint Questions
#
# 1. **A 5 kHz sinusoid is sampled at 8 kHz. What frequency appears digitally?**
# * Calculation: $\vert{}5 - 8\vert{} = 3\text{ kHz}$.
# * **Answer:** 3 kHz.
#
#
# 2. **A signal is bandlimited to 6 kHz. Which rate satisfies Nyquist with margin?**
# * Calculation: $2f_M = 12\text{ kHz}$. Rate must strictly exceed 12 kHz.
# * **Answer:** 16 kHz.
#
#
# 3. **Which component must appear first in an ADC front-end?**
# * **Answer:** Analog anti-aliasing filter.
#
#
# 4. **Why is aliasing problematic for ML models?**
# * **Answer:** It introduces false, structured spectral features into the input data.
#
#
# 5. **At $f_s = 16\text{ kHz}$, which two frequencies produce identical discrete cosines?**
# * Calculation: $\vert{}16 - 13\vert{} = 3\text{ kHz}$.
# * **Answer:** 3 kHz and 13 kHz.
#
#
#
# ---
#
# ## 35. Core Concepts
#
# * **Spectral Replication Origin:** Time-domain multiplication with an impulse train corresponds to frequency-domain convolution with an impulse train.
# * **Factor of 2 in Nyquist:** A real bandlimited spectrum extending from $-f_M$ to $+f_M$ occupies a bandwidth of $2f_M$.
# * **Anti-Alias Filtering Location:** Must occur in the analog domain to remove energy above $f_N$ before sampling makes it indistinguishable from lower-frequency content.
#
# ---
#
# ## 36. Information-Theoretic Summary
#
# $$\boxed{\text{No overlap} \Rightarrow \text{unique reconstruction}; \qquad \text{overlap} \Rightarrow \text{aliasing and non-uniqueness}}$$
#
# If $f_s > 2f_M$, spectral replicas remain isolated, guaranteeing unique signal recovery. If $f_s < 2f_M$, spectral overlap destroys uniqueness.
#
# ---
#
# ## 37. Final Conceptual Mapping
#
# $$\boxed{x_c(t) \xrightarrow{\text{sample every } T} x[n]}$$
#
# $$\boxed{x_c(t) \times \sum_{n} \delta(t-nT)}$$
#
# $$\boxed{X_s(j\Omega) = \frac{1}{T} \sum_{k=-\infty}^{\infty} X_c(j(\Omega-k\Omega_s))}$$
#
# $$\boxed{\text{Spectral replicas}}$$
#
# $$\boxed{f_s > 2f_M \Rightarrow \text{Replicas separate}}$$
#
# $$\boxed{f_s < 2f_M \Rightarrow \text{Replicas overlap} \Rightarrow \text{Aliasing}}$$
#
# $$\boxed{\text{Once false structure enters the samples, downstream models treat it as valid data.}}$$
#
# That is the larger reason this lecture belongs early in a DSP + ML course: students learn that **the data acquisition chain is part of the machine-learning system**, not merely something that happened before the “interesting” ML began.
#
# -----------------------------------------------
#

# %% [markdown]
# # Week 2, Lecture 4, LaTeX math
#
# Here is the lecture content converted with properly formatted LaTeX for all the mathematical expressions, organized for clear readability:
#
# This lecture covers signal reconstruction, continuous-to-discrete amplitude mapping, and the theoretical and practical foundations of quantization.
#
# $$\boxed{\text{When do samples faithfully represent the original analog signal?}}$$
#
# Lecture 4 addresses two key follow-up questions:
#
# $$\boxed{\text{If the samples are valid, how do we turn them back into a continuous waveform?}}$$
#
# $$\boxed{\text{What happens when amplitudes can only be represented with a finite number of bits?}}$$
#
# This presents two distinct analytical paths:
#
# $$\text{samples} \rightarrow \text{ideal reconstruction} \rightarrow \text{practical DAC/ZOH}$$
#
# $$\text{continuous amplitude} \rightarrow \text{quantization} \rightarrow \text{error/noise} \rightarrow \text{SQNR}$$
#
# Sampling error (aliasing) shifts energy to incorrect frequencies and creates non-uniqueness, whereas quantization maps continuous amplitudes onto a finite set of discrete levels.
#
# ---
#
# ### 1. Distinguishing the Two Kinds of Discretization
#
# * **Sampling Discretizes Time:**
# Starting with continuous-time signal $x_c(t)$, sampling yields discrete-time sequence:
#
# $$x[n] = x_c(nT)$$
#
#
#
# In an ideal mathematical model, amplitudes $x[n]$ remain unconstrained continuous real values.
# * **Quantization Discretizes Amplitude:**
# Quantization maps continuous amplitude values to discrete levels.
#
# $$\boxed{\text{sampling: continuous time} \rightarrow \text{discrete time}}$$
#
# $$\boxed{\text{quantization: continuous amplitude} \rightarrow \text{discrete amplitude levels}}$$
#
# ---
#
# ### 2. Ideal Reconstruction: Signal Recovery
#
# Ideal sampling creates spectral replicas:
#
# $$X_s(j\Omega) = \frac{1}{T} \sum_{k=-\infty}^{\infty} X_c(j(\Omega - k\Omega_s))$$
#
# When the Nyquist criterion is satisfied, adjacent copies do not overlap. Reconstructing the original signal requires isolating the baseband spectrum centered at $\Omega = 0$ while rejecting all other copies using an ideal reconstruction low-pass filter:
#
# $$H_r(j\Omega) = \begin{cases} T, & \vert{}\Omega\vert{} < \Omega_c \\ 0, & \text{otherwise} \end{cases}$$
#
# ---
#
# ### 3. Reconstruction Filter Gain $T$
#
# Inside the baseband region, the sampled spectrum is:
#
# $$X_s(j\Omega) = \frac{1}{T} X_c(j\Omega)$$
#
# Applying a unit-gain filter leaves an amplitude scaling of $\frac{1}{T} X_c(j\Omega)$. Setting filter gain $H_r(j\Omega) = T$ in the passband balances the scale factor:
#
# $$T \left(\frac{1}{T} X_c(j\Omega)\right) = X_c(j\Omega)$$
#
# The filter does not construct missing signal information; under bandlimited conditions, the samples contain sufficient information to uniquely specify the continuous waveform.
#
# ---
#
# ### 4. Mathematical Origin of Sinc Interpolation
#
# An ideal rectangular brick-wall filter in the frequency domain corresponds to a $\operatorname{sinc}$ function in the time domain:
#
# $$h_r(t) = \operatorname{sinc}\left(\frac{t}{T}\right)$$
#
# Convolving the impulse train with $h_r(t)$ yields the ideal continuous-time reconstruction formula:
#
# $$\boxed{x_r(t) = \sum_{n=-\infty}^{\infty} x[n] \operatorname{sinc}\left(\frac{t - nT}{T}\right)}$$
#
# where normalized $\operatorname{sinc}$ is defined as:
#
# $$\operatorname{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$$
#
# Each sample weight scales a centered, shifted $\operatorname{sinc}$ kernel.
#
# ---
#
# ### 5. Physical Meaning of Sinc Interpolation
#
# Every sample $x[n]$ scales a $\operatorname{sinc}$ function centered at $t = nT$. The reconstructed signal is the infinite sum of all shifted kernels across time:
#
# $$x_r(t) = \cdots + x[-1]\operatorname{sinc}\left(\frac{t+T}{T}\right) + x[0]\operatorname{sinc}\left(\frac{t}{T}\right) + x[1]\operatorname{sinc}\left(\frac{t-T}{T}\right) + \cdots$$
#
# Because the $\operatorname{sinc}$ function has infinite support, every sample influences intermediate signal values across all time.
#
# ---
#
# ### 6. Sample Point Exact Intersections
#
# The normalized $\operatorname{sinc}$ function satisfies:
#
# $$\operatorname{sinc}(0) = 1 \quad \text{and} \quad \operatorname{sinc}(k) = 0 \quad \forall k \in \mathbb{Z}, k \neq 0$$
#
# Evaluating reconstruction at sample instant $t = mT$:
#
# $$x_r(mT) = \sum_{n=-\infty}^{\infty} x[n] \operatorname{sinc}(m - n)$$
#
# All terms evaluate to zero except where $n = m$, yielding:
#
# $$x_r(mT) = x[m]$$
#
# At $t = 3T$, the contribution from the sample at $n = 1$ uses argument $3 - 1 = 2$:
#
# $$\operatorname{sinc}(2) = 0 \implies \boxed{0}$$
#
# ---
#
# ### 7. Uniqueness Between Sample Points
#
# For arbitrary unconstrained functions, infinitely many curves pass through a set of discrete points. Under the assumption that the original signal is strictly bandlimited below $\frac{f_s}{2}$, only **one unique continuous signal** can generate those exact sample values.
#
# ---
#
# ### 8. Physical DAC Constraints
#
# Ideal $\operatorname{sinc}$ interpolation cannot be implemented directly in physical hardware due to:
#
# * Infinite time duration (infinite support)
# * Non-causality ($h_r(t) \neq 0$ for $t < 0$)
# * Unachievable ideal rectangular frequency response
# * Incomplete bandlimiting in physical inputs
#
# Hardware uses causal continuous-time approximations, such as the Zero-Order Hold (ZOH).
#
# ---
#
# ### 9. Zero-Order Hold (ZOH) Functionality
#
# A ZOH converter maintains a constant voltage output equal to the most recent sample value until the subsequent clock edge:
#
# ```text
# 1.0  ──────────┐
#                │
# 0.5            └──────────┐
#                           │
# 0                    -----│
#                           └────────
# -0.2
#        0          T          2T
#
# ```
#
# The resulting piecewise-constant staircase is a continuous-time analog signal containing high-frequency spectral images and low-pass magnitude droop.
#
# ---
#
# ### 10. ZOH as Pulse Convolution
#
# Impulse-train representation:
#
# $$x_s(t) = \sum_{n=-\infty}^{\infty} x[n] \delta(t - nT)$$
#
# Defining the rectangular pulse $p(t)$:
#
# $$p(t) = \begin{cases} 1, & 0 \le t < T \\ 0, & \text{otherwise} \end{cases}$$
#
# The ZOH output equals the convolution of impulses with the pulse shape:
#
# $$x_{\text{ZOH}}(t) = x_s(t) * p(t)$$
#
# $$\boxed{\text{impulse} * \text{rectangle} = \text{shifted rectangle}}$$
#
# ---
#
# ### 11. ZOH Frequency Response Derivation
#
# Taking the Fourier transform of pulse $p(t) = u(t) - u(t - T)$:
#
# $$H_{\text{ZOH}}(j\Omega) = \int_0^T e^{-j\Omega t} \, dt = \frac{1 - e^{-j\Omega T}}{j\Omega}$$
#
# Factoring symmetrically:
#
# $$1 - e^{-j\Omega T} = e^{-j\Omega T / 2} \left(e^{j\Omega T / 2} - e^{-j\Omega T / 2}\right) = 2j e^{-j\Omega T / 2} \sin\left(\frac{\Omega T}{2}\right)$$
#
# Substituting into the transfer function:
#
# $$H_{\text{ZOH}}(j\Omega) = T e^{-j\Omega T / 2} \frac{\sin(\Omega T / 2)}{\Omega T / 2}$$
#
# $$\boxed{H_{\text{ZOH}}(j\Omega) = T e^{-j\Omega T / 2} \operatorname{sinc}\left(\frac{\Omega T}{2\pi}\right)}$$
#
# This response consists of a linear-phase delay term combined with a $\operatorname{sinc}$ magnitude envelope.
#
# ---
#
# ### 12. Linear-Phase Delay Factor
#
# The exponential factor $e^{-j\Omega T / 2}$ represents a fixed time delay of half the sample period:
#
# $$t_0 = \frac{T}{2}$$
#
# This delay arises because the holding pulse extends forward in time from $t = 0$ to $t = T$ rather than being centered at $t = 0$.
#
# ---
#
# ### 13. ZOH Magnitude Response & Attenuation
#
# Ignoring constant scale factor $T$:
#
# $$\left\vert{} \frac{H_{\text{ZOH}}(j\Omega)}{T} \right\vert{} = \left\vert{} \frac{\sin(\Omega T / 2)}{\Omega T / 2} \right\vert{}$$
#
# * Near DC ($\Omega \approx 0$): $\frac{\sin x}{x} \approx 1$
# * Higher frequencies suffer increasing attenuation up to the first spectral zero at $\Omega = \frac{2\pi}{T}$
#
# $$\boxed{\text{ZOH attenuates high frequencies more than low frequencies.}}$$
#
# ---
#
# ### 14. Quantifying Nyquist Droop
#
# At the digital Nyquist limit ($\Omega T = \pi$):
#
# $$\frac{\vert{}H\vert{}}{T} = \frac{\sin(\pi / 2)}{\pi / 2} = \frac{2}{\pi} \approx 0.6366$$
#
# Expressed in decibels:
#
# $$20 \log_{10}(0.6366) \approx -3.92\text{ dB}$$
#
# This $-3.92\text{ dB}$ attenuation near Nyquist is deterministic frequency shaping inherent to holding continuous samples, not random noise.
#
# ---
#
# ### 15. Role of the Analog Reconstruction Filter
#
# The ZOH envelope attenuates high-frequency spectral replicas centered at $f_s, 2f_s, \dots$ but does not eliminate them entirely:
#
# $$\boxed{x[n] \rightarrow \text{DAC/ZOH} \rightarrow \text{Analog Reconstruction LPF} \rightarrow x_r(t)}$$
#
# The post-DAC analog low-pass filter suppresses remaining high-frequency images.
#
# ---
#
# ### 16. Oversampling Advantage
#
# * **Base Rate Sampling ($f_s = 44.1\text{ kHz}$):**
# Passband ends at $20\text{ kHz}$ and Nyquist is $22.05\text{ kHz}$, requiring steep analog filter roll-off in a $2.05\text{ kHz}$ transition band.
# * **Oversampled Conversion ($f_s = 176.4\text{ kHz}$):**
# First spectral images are pushed up to $176.4\text{ kHz}$, allowing a gradual analog filter transition band.
#
# ---
#
# ### 17. Quantization Dynamics
#
# Quantization converts continuous amplitude values into $B$-bit numerical codes:
#
# $$\boxed{L = 2^B}$$
#
# | Resolution ($B$) | Code Levels ($L = 2^B$) |
# | --- | --- |
# | 4 bits | 16 |
# | 8 bits | 256 |
# | 16 bits | 65,536 |
# | 24 bits | 16,777,216 |
#
# Inputs exceeding the input voltage range $[-A, A)$ cause saturation clipping.
#
# ---
#
# ### 18. Quantizer Step Size Formula
#
# For dynamic range $[-A, A)$ with total width $2A$ split across $2^B$ levels:
#
# $$\boxed{\Delta \approx \frac{2A}{2^B}}$$
#
# * For $A = 1, B = 8$: $\Delta \approx \frac{2}{256} = 0.0078125$
# * For $A = 1, B = 4$: $\Delta \approx \frac{2}{16} = 0.125$
#
# ---
#
# ### 19. Additive Quantization Error Model
#
# Quantization error $e[n]$ is defined as:
#
# $$\boxed{e[n] = x_q[n] - x[n]} \implies x_q[n] = x[n] + e[n]$$
#
# For round-to-nearest quantizers, error bounds are:
#
# $$-\frac{\Delta}{2} \le e[n] < \frac{\Delta}{2}$$
#
# ---
#
# ### 20. Statistical Noise Model Assumptions
#
# Quantization is deterministic, but error sequence $e[n]$ can be modeled as zero-mean uniform white noise:
#
# $$e[n] \sim \text{Uniform}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$$
#
# This model applies when signal amplitude spans many quantization steps, possesses complex phase structure, and quantization resolution is sufficiently high.
#
# ---
#
# ### 21. Derivation of Error Variance $\sigma_e^2$
#
# For uniformly distributed random variable $e \sim U\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$ with $p(e) = \frac{1}{\Delta}$:
#
# $$\sigma_e^2 = \mathbb{E}[e^2] = \int_{-\Delta/2}^{\Delta/2} e^2 \left(\frac{1}{\Delta}\right) \, de$$
#
# $$\sigma_e^2 = \frac{1}{\Delta} \left[ \frac{e^3}{3} \right]_{-\Delta/2}^{\Delta/2} = \frac{1}{\Delta} \left( \frac{\Delta^3}{24} - \left(-\frac{\Delta^3}{24}\right) \right)$$
#
# $$\boxed{\sigma_e^2 = \frac{\Delta^2}{12}}$$
#
# ---
#
# ### 22. Derivation of 6 dB/Bit Rule
#
# Incrementing bit depth by 1 bit doubles available levels:
#
# $$2^B \rightarrow 2^{B+1} \implies \Delta \rightarrow \frac{\Delta}{2}$$
#
# Error power scales quadratically with step size:
#
# $$\sigma_e^2 \propto \Delta^2 \implies \sigma_{e, \text{new}}^2 = \frac{(\Delta/2)^2}{12} = \frac{1}{4} \sigma_e^2$$
#
# In decibels:
#
# $$10 \log_{10}(4) \approx 6.02\text{ dB}$$
#
# $$\boxed{\text{one additional bit} \approx 6.02\text{ dB improvement}}$$
#
# ---
#
# ### 23. Full-Scale Sinusoid SQNR Derivation
#
# For full-scale sinusoid $x[n] = A \sin(\omega n)$:
#
# * RMS Amplitude: $\frac{A}{\sqrt{2}}$
# * Signal Power: $P_x = \frac{A^2}{2}$
# * Step Size: $\Delta = \frac{2A}{2^B}$
# * Noise Power: $P_e = \frac{\Delta^2}{12} = \frac{1}{12} \frac{4A^2}{2^{2B}} = \frac{A^2}{3 \cdot 2^{2B}}$
#
# Calculating Signal-to-Quantization-Noise Ratio:
#
# $$\text{SQNR} = \frac{P_x}{P_e} = \frac{A^2 / 2}{A^2 / (3 \cdot 2^{2B})} = \frac{3}{2} \cdot 2^{2B}$$
#
# Converting to decibels:
#
# $$\text{SQNR}_{\text{dB}} = 10 \log_{10}\left(\frac{3}{2}\right) + 20B \log_{10}(2)$$
#
# $$\boxed{\text{SQNR} \approx 6.02B + 1.76\text{ dB}}$$
#
# ---
#
# ### 24. Standard SQNR Reference Values
#
# * **8-bit:** $6.02(8) + 1.76 = 49.92\text{ dB} \approx \boxed{50\text{ dB}}$
# * **12-bit:** $6.02(12) + 1.76 = 74.00\text{ dB} \approx \boxed{74\text{ dB}}$
# * **16-bit:** $6.02(16) + 1.76 = 98.08\text{ dB} \approx \boxed{98.1\text{ dB}}$
#
# ---
#
# ### 25. 96 dB vs. 98.1 dB Dynamic Range Distinction
#
# * $6.02 \times 16 = 96.32\text{ dB}$ estimates raw 16-bit dynamic range.
# * $6.02(16) + 1.76 = 98.08\text{ dB}$ calculates ideal SQNR specifically for a full-scale sinusoid.
#
# ---
#
# ### 26. Signal Power Dependencies
#
# The $1.76\text{ dB}$ constant assumes a full-scale sinusoid. If signal amplitude drops to $0.1A$ ($-20\text{ dB}$ input level) while step size $\Delta$ remains fixed, actual SQNR drops by $20\text{ dB}$.
#
# ---
#
# ### 27. Quantization vs. Overload Clipping
#
# * **Quantization Error:** Bounded round-off within $[-\frac{\Delta}{2}, \frac{\Delta}{2})$.
# * **Clipping:** Non-linear waveform truncation occurring when $\vert{}x(t)\vert{} > A$. Creates severe harmonic distortion.
#
# ---
#
# ### 28. Coarse Quantization Artifacts (4-bit Audio)
#
# At low resolutions ($B = 4$, 16 levels), error terms correlate strongly with signal trajectories, generating coarse, harmonic distortion rather than white additive noise.
#
# ---
#
# ### 29. Low-Amplitude Periodic Error
#
# Low-amplitude sinusoids crossing only 2–3 quantization levels yield periodic error signals. Periodic errors introduce distinct harmonic spectral lines instead of uniform noise floors.
#
# ---
#
# ### 30. Function of Dither
#
# Dither adds small-amplitude random noise (e.g., triangular probability density function with amplitude $\pm \Delta$) prior to quantization. Dither decorrelates quantization error from the input signal, converting structured harmonic distortion into stationary, non-objectionable white noise.
#
# $$\boxed{\text{structured distortion}} \longrightarrow \boxed{\text{randomized noise floor}}$$
#
# ---
#
# ### 31. Machine Learning Sensitivity to Artifacts
#
# Models trained on coarsely quantized inputs without dither may learn false, structured amplitude artifacts as valid class features.
#
# ---
#
# ### 32. Numerical Type Casting (`int16` to `float32`)
#
# Converting 16-bit PCM integer samples to floating-point tensors changes computational precision, not underlying acquisition measurement resolution.
#
# ```python
# x = pcm.astype(np.float32) / 32768.0  # Scales numeric range to [-1.0, 1.0)
#
# ```
#
# ---
#
# ### 33. Information Invariance Under Type Conversion
#
# Casting an `int16` value to `float32` does not recover amplitude detail removed during quantization.
#
# $$\boxed{\text{more bits in container} \neq \text{more measurement detail}}$$
#
# ---
#
# ### 34. Non-Reversibility of Hardware Distortion
#
# Type casting to `float64` cannot restore information lost to aliasing or quantization.
#
# ---
#
# ### 35. Precision Conceptual Check
#
# Loading 16-bit audio into PyTorch as a `torch.float32` tensor updates the arithmetic container format used for gradient updates and operations, leaving physical measurement resolution fixed at 16 bits.
#
# ---
#
# ### 36. Audio Demonstration Progression
#
# 1. **Clean 48 kHz:** Baseline reference.
# 2. **Naive 48 kHz $\rightarrow$ 12 kHz:** Severe aliasing fold-over ($\boxed{\text{false frequencies}}$).
# 3. **Filtered 48 kHz $\rightarrow$ 12 kHz:** Bandwidth reduced above $6\text{ kHz}$ without aliasing distortion.
# 4. **8-bit Quantization:** Higher noise floor, speech remains intelligible.
# 5. **4-bit Quantization:** Heavy signal-correlated granular distortion.
#
# ---
#
# ### 37. Qualitative Distinction: Aliasing vs. Quantization
#
# * **Aliasing:** Maps spectral energy to incorrect baseband locations, causing irreversible frequency overlap.
# * **Quantization:** Perturbs sample amplitude values, adding noise or harmonic distortion.
#
# ---
#
# ### 38. Structural Comparison: ZOH Droop vs. Quantization Noise
#
# * **ZOH Droop:** Deterministic continuous-time linear magnitude attenuation ($\operatorname{sinc}$ shaped).
# * **Quantization Noise:** Amplitude error statistical approximation ($\frac{\Delta^2}{12}$ variance).
#
# ---
#
# ### 39. Core Conceptual Answers
#
# * **Ideal Reconstruction Sinc Basis:** The ideal low-pass reconstruction filter has a rectangular spectrum in frequency, corresponding to a $\operatorname{sinc}(t)$ impulse response in time.
# * **ZOH Frequency Impact:** Introduces a $\operatorname{sinc}$-shaped attenuation envelope ($-3.92\text{ dB}$ at Nyquist) and a $T/2$ linear-phase delay.
# * **12-bit Full-Scale Sine SQNR:**
#
# $$\text{SQNR} = 6.02(12) + 1.76 = \boxed{74.0\text{ dB}}$$
#
#
#
# ---
#
# ### 40. Verification Questions
#
# 1. **Sinc Reconstruction Contribution:** Each sample weight launches an infinitely long shifted $\operatorname{sinc}$ function.
# 2. **Zero-Order Hold Core Function:** Holds sample amplitudes constant across sampling interval $T$.
# 3. **Adding One Bit:** Step size $\Delta$ halves, noise power drops by a factor of 4, improving SQNR by $\approx 6\text{ dB}$.
# 4. **`int16` to `float32` Conversion:** Changes internal compute precision without increasing acquisition information.
# 5. **Unmodeled Behavior:** Overload clipping falls outside the standard $\frac{\Delta^2}{12}$ uniform noise model.
#
# ---
#
# ### 41. Level-Based Analysis Summary
#
# * **Undergraduate/Senior Formulations:**
#
# $$\Delta \approx \frac{2A}{2^B}, \quad \sigma_e^2 = \frac{\Delta^2}{12}, \quad \text{SQNR} \approx 6.02B + 1.76\text{ dB}$$
#
#
# * **Graduate Formulations:** Integration of $e^2 p(e)$, bounds of uniform noise models, signal-error correlation dynamics, decorrelation via dither, and ZOH Fourier transform derivations.
#
# ---
#
# ### 42. End-to-End Acquisition & Reconstruction Chain
#
# 1. **Anti-Alias Filter:** $x_c(t) \rightarrow \text{Analog LPF}$ (suppresses frequencies above $f_N$).
# 2. **Sampling:** $x[n] = x_c(nT)$ (discretizes time).
# 3. **Quantization:** $x_q[n] = x[n] + e[n]$ (discretizes amplitude).
# 4. **Type Casting:** `int16` $\rightarrow$ `float32` (prepares storage container for computation).
# 5. **D/A Conversion:** ZOH produces staircase waveform $x_{\text{ZOH}}(t) = x_s(t) * p(t)$.
# 6. **Reconstruction Filter:** Suppresses spectral images around $k f_s$.
#
# ---
#
# ### 43. Hardware-Software Pipeline Principles
#
# $$\boxed{\text{Sampling determines whether frequency information survives.}}$$
#
# $$\boxed{\text{Quantization determines how accurately amplitude survives.}}$$
#
# $$\boxed{\text{Reconstruction determines how digital values map back to physical waveforms.}}$$
#
# That gives students a much stronger conceptual basis than treating “ADC,” “DAC,” “bit depth,” and “sample rate” as isolated engineering vocabulary.
#
# ---------------------------------------
#

# %% [markdown]
# ## Week 2, Lecture 3, Exit Tickets
#
# What are good exit ticket questions with answers for this lecture?
#
# Chat GPT (Medium):
#
# For this lecture, I’d make the exit ticket short enough to finish in about **3–5 minutes**, but design it to test whether students understand the *reasoning* behind sampling rather than just remembering (f_s>2f_M). Your lecture emphasizes the sampling model, spectral replication, aliasing as non-uniqueness, and why anti-alias filtering must occur before sampling or decimation. 
#
# Here are some strong choices.
#
# 1. **A continuous-time signal contains frequencies only up to 7 kHz. What is the minimum theoretical sampling rate required to avoid aliasing?**
#
#    **Answer:**
#    
#    $$f_s > 2f_M = 14\text{ kHz}$$
#    
#    So theoretically the sampling rate must be **greater than 14 kHz**. In practice, you would choose something higher to allow room for the anti-alias filter transition band.
#
#    **What this tests:** Whether students distinguish the signal bandwidth from the Nyquist frequency and can apply the no-overlap condition. Your slides explicitly distinguish the Nyquist frequency (f_s/2) from the Nyquist rate (2f_M). 
#
# 2. **A 9 kHz sinusoid is sampled at (f_s=12) kHz. What frequency will appear in the sampled data?**
#
#    **Answer:**
#    The Nyquist frequency is 6 kHz. The 9 kHz sinusoid folds around 6 kHz:
#
#    
#    $$f_{\text{alias}}=|9-12|=3\text{ kHz}.$$
#    
#
#    So the sampled data looks like a **3 kHz sinusoid**.
#
#    **What this tests:** Whether students can actually predict an alias rather than only say “aliasing occurs.”
#
# 3. **True or false: If aliasing occurs during sampling, we can usually remove it afterward with a digital low-pass filter. Explain in one sentence.**
#
#    **Answer:** **False.** Once aliasing occurs, an aliased component and a genuine in-band component can produce the same samples, so the digital system cannot determine which one was originally present.
#
#    This is one of the most important conceptual points in the lecture: aliasing is a loss of uniqueness, not simply extra noise. 
#
# 4. **Why does sampling create repeated copies of the analog spectrum?**
#
#    **Good answer:** Because ideal sampling multiplies the continuous-time signal by a periodic impulse train. Multiplication in time corresponds to convolution in frequency, and the Fourier transform of the impulse train is itself an impulse train. Convolving (X_c(j\Omega)) with those impulses creates shifted copies of the spectrum spaced by the sampling frequency.
#
#    **Excellent student answer:**
#    “Sampling multiplies (x_c(t)) by an impulse train; in frequency this becomes convolution with another impulse train, producing replicas of (X_c(j\Omega)) every (\Omega_s).”
#
#    That is almost exactly the central derivation developed in the lecture.  
#
# 5. **Suppose two different continuous-time sinusoids produce exactly the same discrete-time samples. What phenomenon does this illustrate?**
#
#    **Answer:** **Aliasing.**
#
#    **Better answer:** Aliasing demonstrates that the mapping from continuous-time signals to samples is no longer one-to-one; multiple analog signals are consistent with the same digital sequence.
#
#    **What this tests:** Whether they understand the deeper idea that aliasing is **non-uniqueness** rather than merely “frequencies getting messed up.” Your lecture makes that distinction explicitly. 
#
# 6. **You have audio sampled at 48 kHz and want to reduce the rate to 12 kHz. Why is `x[::4]` potentially wrong? What should you do instead?**
#
#    **Answer:** Keeping every fourth sample changes the Nyquist frequency from 24 kHz to 6 kHz. Any energy above 6 kHz can therefore alias into the new baseband.
#
#    Correct procedure:
#
#    
#    $$\boxed{\text{Low-pass filter first} \rightarrow \text{downsample by 4}}$$
#    
#
#    This connects sampling theory directly to the practical decimation example in the lecture. 
#
# 7. **A 10 kHz component is present in a 48 kHz recording. The recording is naively downsampled to 12 kHz. Where will the 10 kHz component appear?**
#
#    **Answer:**
#
#    
#    $$f_{\text{alias}}=|10-12|=2\text{ kHz}.$$
#    
#
#    It appears falsely as a **2 kHz component**.
#
#    **Follow-up:** Why could this be particularly dangerous for a machine-learning classifier?
#
#    **Answer:** Because the classifier may learn the repeatable 2 kHz alias as though it were a real physical feature of the signal. Your lecture specifically emphasizes that a CNN cannot know that this structure originated as an alias. 
#
# 8. **Complete the sentence: An anti-alias filter must be placed ______ the ADC because ______.**
#
#    **Answer:**
#    “An anti-alias filter must be placed **before** the ADC because once out-of-band energy folds into the sampled baseband, its original frequency cannot generally be recovered.”
#
#    **What this tests:** Whether students understand the physical ordering, not merely the vocabulary. 
#
# For this particular lecture, my favorite **three-question exit ticket** would be:
#
# > **1.** A 9 kHz tone is sampled at 12 kHz. What frequency appears in the samples?
# > **Answer:** 3 kHz.
# >
# > **2.** Why can't a digital filter generally remove aliasing after sampling?
# > **Answer:** Because after frequencies fold together, genuine in-band and aliased components are indistinguishable.
# >
# > **3.** In one or two sentences, explain why sampling creates spectral replicas.
# > **Answer:** Sampling multiplies the signal by a periodic impulse train. Multiplication in time becomes convolution in frequency, creating shifted copies of the original spectrum at multiples of the sampling frequency.
#
# That combination is particularly useful because **Question 1 tests calculation, Question 2 tests physical/conceptual understanding, and Question 3 tests whether they understood the mathematical mechanism**. Together they give you a better diagnosis than three Nyquist-rate calculations.
#
# One question I would **not** use as the primary exit-ticket question is simply, *“State the Nyquist sampling theorem.”* A student can memorize (f_s>2f_M) and still misunderstand essentially everything important in this lecture. Your stated learning objective is specifically that students should be able to **derive—not just quote—the sampling theorem**, so the exit ticket should reflect that goal. 
#
# -----------------
#

# %% [markdown]
# ## Week 2, Lecture 4, Exit Tickets
#
# What are good exit ticket questions with answers for this lecture?
#
# Chat GPT (Medium):
#
# For this lecture, I’d make the exit ticket test three different things: **ideal reconstruction**, **practical DAC behavior**, and **quantization/SQNR**. That matches the lecture’s stated outcomes very well. 
#
# Here are several good choices with answers.
#
# 1. **Why does ideal reconstruction use sinc functions?**
#
#    **Answer:** Because the ideal reconstruction filter is a brick-wall low-pass filter, and its impulse response is a sinc function. Therefore, reconstruction is performed by summing shifted sinc functions weighted by the sample values.
#
#    A strong student answer would be:
#    “Each sample scales a shifted sinc, and the sinc functions interpolate exactly through the original sample values.”
#
#    This directly follows the lecture’s derivation from ideal low-pass filtering to sinc interpolation. 
#
# 2. **Why does sinc interpolation pass exactly through the sample values?**
#
#    **Answer:** Because the normalized sinc function is zero at every nonzero integer offset and equals 1 at zero. At (t=mT), all sinc terms vanish except the one centered at (mT).
#
#    So:
#
#    
#    $$x_r(mT)=x[m].$$
#    
#
#    This is a very good conceptual check because it shows whether students understand why sinc interpolation actually works rather than simply recognizing the formula. 
#
# 3. **Why don't practical DACs usually implement ideal sinc interpolation directly?**
#
#    **Answer:** Because an ideal sinc impulse response has infinite duration and is noncausal. A real DAC instead usually produces a simpler waveform such as a zero-order hold and then uses an analog reconstruction filter.
#
#    The lecture explicitly contrasts ideal interpolation with the practical DAC approach. 
#
# 4. **What does a zero-order hold do in the time domain?**
#
#    **Answer:** It holds each sample value constant for one sampling period.
#
#    **Follow-up:** What does that do in frequency?
#
#    **Answer:** It introduces a sinc-shaped magnitude response, causing progressively greater attenuation at higher frequencies.
#
#    This makes a nice two-part exit-ticket question because students must connect the time-domain staircase to the frequency-domain droop.  
#
# 5. **True or false: The frequency droop produced by a zero-order hold is random noise.**
#
#    **Answer:** **False.** It is a deterministic frequency-response effect.
#
#    **Better answer:** The ZOH has a sinc-shaped transfer-function magnitude, so higher-frequency components are attenuated in a predictable way.
#
#    That distinction is explicitly emphasized in the lecture. 
#
# 6. **A DAC uses a zero-order hold. What is the purpose of the analog reconstruction filter that follows it?**
#
#    **Answer:** Primarily to remove spectral images around multiples of the sampling frequency. It can also help compensate some of the ZOH passband droop.
#
#    This tests whether students see the DAC and analog filter as one reconstruction chain rather than treating the DAC output as the final waveform. 
#
# 7. **What is the difference between sampling and quantization?**
#
#    **Answer:**
#
#    * Sampling discretizes **time**.
#    * Quantization discretizes **amplitude**.
#
#    This is simple, but it is worth asking because students often mix the two concepts. The lecture makes this distinction explicitly. 
#
# 8. **How many nominal quantization levels does an (8)-bit quantizer have?**
#
#    **Answer:**
#
#    
#    $$2^8=256.$$
#    
#
#    **Follow-up:** What happens to the quantization step $\Delta$ if you add one more bit?
#
#    **Answer:** It is approximately cut in half.
#
#    Since quantization-noise power is proportional to $\Delta^2$, halving $\Delta$ reduces the noise power by approximately a factor of 4.  
#
# 9. **Approximately how much does ideal SQNR improve when one bit is added?**
#
#    **Answer:** About
#
#    
#    $$6\text{ dB per bit}.$$
#    
#
#    More precisely, for a full-scale sine,
#
#    
#    $$\mathrm{SQNR}\approx6.02B+1.76\text{ dB}.$$
#    
#
#    The “6 dB per bit” result comes from the fact that one extra bit halves $\Delta$, reducing quantization-noise power by a factor of 4. 
#
# 10. **Estimate the ideal SQNR of a full-scale 10-bit sinusoid.**
#
# **Answer:**
#
#
# $$\mathrm{SQNR}\approx 6.02(10)+1.76$$
#
#
#
# $$\approx61.96\text{ dB}$$
#
#
# or about **62 dB**.
#
# 11. **Estimate the ideal SQNR of a full-scale 12-bit sinusoid.**
#
# **Answer:**
#
#
# $$6.02(12)+1.76
# =74.0\text{ dB}.$$
#
#
# This is also the calculation already included in the lecture's existing exit check. 
#
# 12. **Why is the common statement “16-bit audio has about 96 dB of dynamic range” slightly different from the (98.1) dB SQNR result?**
#
# **Answer:** They use slightly different reference calculations. Roughly (6) dB/bit gives about (96) dB, while
#
#
# $$6.02B+1.76$$
#
#
# gives approximately (98.1) dB for a full-scale sine because it incorporates the RMS power of the sine and the uniform quantization-error model.
#
# That is a stronger exit question for an advanced class because it tests whether students understand where the numbers come from rather than treating them as conflicting facts. 
#
# 13. **A recording is stored as 16-bit PCM. You convert it to a PyTorch `float32` tensor. Has the signal now gained additional amplitude resolution?**
#
# **Answer:** **No.**
#
# The representation used for subsequent computation has changed, but the information captured during quantization has not. Values between the original 16-bit quantization levels cannot be recovered just by casting the data to `float32`. 
#
# This would be one of my favorite exit-ticket questions because it connects the DSP material directly to the later ML part of the course.
#
# 14. **True or false: If an audio recording is stored as `float32`, it cannot contain significant quantization error.**
#
# **Answer:** **False.**
#
# A `float32` tensor may simply contain values that originated from a much lower-resolution ADC or PCM file. More computational precision does not restore information that was lost during acquisition. 
#
# 15. **The additive quantization-noise model assumes quantization error is approximately uniform and uncorrelated with the signal. Is this always true?**
#
# **Answer:** **No.**
#
# It is a useful approximation when the signal spans many quantization levels and the error is sufficiently decorrelated from the signal. Low-level signals can produce structured or signal-correlated quantization error.
#
# This is a very good higher-level conceptual question because it tests whether students understand the distinction between a mathematical model and physical reality. 
#
# For a **3-question exit ticket**, I would probably use these:
#
# **1. Explain in one sentence why ideal reconstruction produces sinc interpolation.**
#
# **Answer:** Because the ideal reconstruction system is a brick-wall low-pass filter whose impulse response is a sinc.
#
# **2. A DAC uses a zero-order hold. What happens to its frequency response as frequency increases?**
#
# **Answer:** The response follows a sinc-shaped envelope, so higher frequencies experience progressively greater magnitude droop.
#
# **3. An audio file recorded with a 12-bit ADC is converted to `float32` for PyTorch. Does it now have 32-bit measurement accuracy? Explain.**
#
# **Answer:** No. `float32` gives more numerical precision for computation, but it cannot recreate amplitude information lost during the original 12-bit quantization.
#
# I particularly like that set because the questions move from **ideal theory → practical hardware → DSP/ML interpretation**, which is exactly the progression of the lecture. The lecture summary itself follows essentially that same structure: sinc interpolation, ZOH behavior, quantization/SQNR, and the distinction between PCM bit depth and computational dtype. 
#
# If you want a slightly more quantitative exit ticket, I would substitute Question 3 with:
#
# **A full-scale 8-bit sinusoid has approximately what ideal SQNR?**
#
#
# $$6.02(8)+1.76\approx49.9\text{ dB}.$$
#
#
# That one is useful because a student who answers **48 dB** is showing that they remember the (6\text{ dB/bit}) approximation, while a student who gets **about 50 dB** understands the full-scale-sine formula. Both responses tell you something useful about where they are conceptually.
#
# ------------------------
#

# %%
