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
# # Week 9 easier
#
# **Week 9: Spectral Estimation.** Power Spectral Density (PSD), periodograms, and Welch’s method.
#

# %% [markdown]
# ## Week 9: Spectral Estimation
#
# **Objective:** Map random time-domain signals into the frequency domain using statistical estimators, addressing the limitations of naive frequency transformations.
#
# ### Lecture 17: Power Spectral Density and the Periodogram (75 min)
#
# * **The Wiener-Khinchin Theorem (20 min):** Defining Power Spectral Density (PSD) as the Discrete-Time Fourier Transform (DTFT) of the autocorrelation sequence: $S_{xx}(e^{j\omega}) = \sum_{m=-\infty}^{\infty} r_{xx}[m] e^{-j\omega m}$.
# * **Filtering in the Frequency Domain (15 min):** The PSD of an LTI system output: $S_{yy}(e^{j\omega}) = S_{xx}(e^{j\omega}) \vert{}H(e^{j\omega})\vert{}^2$.
# * **The Periodogram (25 min):** Formulating the sample estimate of the PSD directly from finite data: $I(\omega) = \frac{1}{N} \left\vert{} \sum_{n=0}^{N-1} x[n]e^{-j\omega n} \right\vert{}^2$.
# * **Statistical Flaws of the Periodogram (15 min):** Proving that the periodogram is an inconsistent estimator (its variance does not approach zero as $N \to \infty$).
#
# ### Lecture 18: Non-Parametric Spectral Estimation (75 min)
#
# * **Bartlett’s Method (20 min):** Reducing variance by splitting the signal into $K$ non-overlapping segments and averaging their periodograms. Analyzing the trade-off between variance reduction and frequency resolution.
# * **Welch’s Method (30 min):** Improving upon Bartlett by introducing overlapping segments and applying tapering windows (e.g., Hamming, Hann) to mitigate spectral leakage.
# * **Resolution vs. Variance Trade-offs (15 min):** Quantitative analysis of window choice, overlap percentage, and segment length on the quality of the PSD estimate.
# * **ML Bridge (10 min):** Extracting stable PSD estimates (via Welch's method) as robust frequency-domain features for classical ML classifiers like Random Forests or SVMs.
#
# ------------------------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# ChatGPT: (Medium)
#
# Done. I created the complete Week 9 Spectral Estimation package.
#
# ### Lecture slides
#
# * [Lecture 17 — Power Spectral Density and the Periodogram](sandbox:/mnt/data/week9_spectral_estimation/Week9_Lecture17_PSD_Periodogram.pptx)
# * [Lecture 18 — Non-Parametric Spectral Estimation](sandbox:/mnt/data/week9_spectral_estimation/Week9_Lecture18_Nonparametric_Spectral_Estimation.pptx)
#
# The decks are paced for the two 75-minute classes and include worked examples, statistical interpretations, concept checks, visual PSD comparisons, instructor prompts, ML connections, and exit-ticket questions.
#
# Lecture 17 emphasizes an important distinction that students often miss: increasing record length improves frequency detail, but the variance of a raw periodogram at a fixed frequency does not generally vanish. A Monte Carlo example shows this directly for white Gaussian noise.
#
# Lecture 18 then builds naturally from that problem into Bartlett and Welch estimation, with explicit treatment of:
#
# * segment length,
# * number of averages,
# * window leakage,
# * overlap,
# * true resolution versus zero-padding,
# * and the resolution–variance trade-off.
#
# ### Python material
#
# * [Jupyter notebook — Week 9 Spectral Estimation DSP + ML Demos](sandbox:/mnt/data/week9_spectral_estimation/Week9_Spectral_Estimation_DSP_ML_Demos.ipynb)
# * [Plain Python version](sandbox:/mnt/data/week9_spectral_estimation/Week9_Spectral_Estimation_DSP_ML_Demos.py)
#
# The code includes:
#
# * numerical Wiener–Khinchin demonstrations,
# * PSD shaping through an LTI filter,
# * Monte Carlo demonstration of raw-periodogram inconsistency,
# * Bartlett estimation implemented explicitly,
# * comparison of periodogram, Bartlett, and Welch,
# * segment-length experiments,
# * zero-padding versus actual frequency resolution,
# * Welch bandpower feature extraction,
# * SVM classification,
# * and Random Forest cross-validation.
#
# I executed both the Python script and notebook successfully.
#
# ### Homework and weekly quiz
#
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week9_spectral_estimation/Week9_Spectral_Estimation_Homework_Quiz_Solutions.docx)
#
# The packet contains 8 homework problems with solutions and grading guidance, plus an 8-question weekly quiz with solutions. It includes both first-year-accessible interpretation questions and more quantitative senior-level estimator-design questions.
#
# ### Complete package
#
# * [Download the entire Week 9 package](sandbox:/mnt/data/week9_spectral_estimation/Week9_Spectral_Estimation_DSP_ML_Package.zip)
#
# One theme I reinforced throughout both lectures is that **frequency resolution and statistical variance are different properties**. Longer data records and zero-padding affect the frequency representation differently from averaging, segmentation, and Welch processing; that distinction becomes especially important when PSD estimates are used as ML features.
#
# -------------------------
#

# %% [markdown]
# ## Week 9, Lecture 17
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT (Medium):
#
# Below is a more detailed instructor-oriented explanation of Lecture 17, including what is happening mathematically, what students should understand conceptually, and examples of strong answers to the concept questions. The lecture’s main arc is: **theoretical PSD $\rightarrow$ finite-data estimation $\rightarrow$ periodogram $\rightarrow$ why the raw periodogram is noisy**. 
#
# ## 1. What is the central idea of this lecture?
#
# The lecture is making an important transition.
#
# Up to this point, students may be comfortable treating spectra as deterministic objects:
#
# * Take a signal.
# * Compute a Fourier transform.
# * Examine its frequency content.
#
# With a **random process**, there is an additional layer. We are usually not interested in the exact Fourier transform of one random realization. We want to characterize how the **average power of the underlying stochastic process** is distributed over frequency.
#
# That theoretical object is the **power spectral density**, or PSD.
#
# The practical problem is that the PSD is defined using statistical quantities such as expectations, but in a real experiment we often have only one finite record:
#
# $$
# x[0],x[1],\ldots,x[N-1].
# $$
#
# So the lecture is fundamentally asking:
#
# > How much can we learn about the theoretical PSD of a random process from one finite realization?
#
# That distinction between the **process** and the **realization** is probably the single most important conceptual point in the lecture.
#
# ---
#
# # 2. PSD: “Where does the average power live?”
#
# For a wide-sense stationary process, the autocorrelation depends only on the lag $m$:
#
# $$
# r_{xx}[m]=E\{x[n]x^*[n+m]\}.
# $$
#
# The PSD is the DTFT of the autocorrelation:
#
# $$
# S_{xx}(e^{j\omega}) =
# \sum_{m=-\infty}^{\infty}
# r_{xx}[m]e^{-j\omega m}.
# $$
#
# The deck describes correlation as organizing second-order information by time lag, while PSD organizes the same information by frequency. 
#
# ### What students should understand
#
# The PSD is **not simply the magnitude squared of one Fourier transform**.
#
# It is fundamentally a statistical quantity.
#
# Autocorrelation answers questions such as:
#
# > If I know the value of the process now, how strongly related is it to the value $m$ samples away?
#
# The PSD answers:
#
# > At what frequencies is the average power of this random process concentrated?
#
# These are two ways of expressing exactly the same second-order information.
#
# ### Useful classroom example
#
# Suppose a process changes very slowly from one sample to the next.
#
# Then $x[n]$ and $x[n+1]$ are highly correlated.
#
# Therefore $r_{xx}[1]$ is large.
#
# Slow variation corresponds to low frequencies, so we would expect the PSD to contain substantial power near $\omega=0$.
#
# Conversely, a rapidly alternating process tends to have more high-frequency energy.
#
# This gives students a physical connection between autocorrelation and frequency content before they focus on the transform formulas.
#
# ---
#
# # 3. Wiener–Khinchin in both directions
#
# The relationship is
#
# $$
# S_{xx}(e^{j\omega}) =
# \operatorname{DTFT}\{r_{xx}[m]\}
# $$
#
# and
#
# $$
# r_{xx}[m] =
# \frac{1}{2\pi}
# \int_{-\pi}^{\pi}
# S_{xx}(e^{j\omega})e^{j\omega m}\,d\omega.
# $$
#
# At $m=0$,
#
# $$
# r_{xx}[0] =
# E\{|x[n]|^2\}.
# $$
#
# Therefore
#
# $$
# E\{|x[n]|^2\} =
# \frac{1}{2\pi}
# \int_{-\pi}^{\pi}
# S_{xx}(e^{j\omega})\,d\omega.
# $$
#
# So the **area under the PSD is directly related to total average power**. 
#
# ### A useful way to say this in class
#
# You might tell students:
#
# > A Fourier transform tells us where a deterministic signal lives in frequency. A PSD tells us where the average power of a random process lives in frequency.
#
# And then:
#
# > If you integrate all of that power density over all frequencies, you must recover the total average power.
#
# That gives the formula an intuitive conservation interpretation.
#
# ---
#
# # 4. Why “density” matters
#
# Students often overlook the word **density**.
#
# $S_{xx}(e^{j\omega})$ is not simply “power at frequency $\omega$.” It is power **per unit frequency**.
#
# For continuous frequency,
#
# $$
# \text{power in a small band}
# \approx
# S_{xx}(e^{j\omega})\,d\omega.
# $$
#
# Therefore changing the frequency variable changes the numerical value of the density.
#
# For example, if PSD is expressed in terms of Hz instead of rad/sample, the horizontal-axis scaling changes, so the vertical-axis density must change correspondingly if total power is to remain the same.
#
# This is why students should not casually compare:
#
# * PSD in power/Hz,
# * PSD in power/(rad/sample),
# * spectrum values,
# * or FFT magnitude squared,
#
# without checking normalization.
#
# ---
#
# # 5. White noise example
#
# For discrete-time white noise with variance $\sigma^2$,
#
# $$
# r_{xx}[m] =
# \sigma^2\delta[m].
# $$
#
# Taking the DTFT,
#
# $$
# S_{xx}(e^{j\omega}) =
# \sigma^2.
# $$
#
# So the theoretical PSD is perfectly flat. 
#
# ### Why?
#
# White noise has zero correlation at every nonzero lag:
#
# $$
# r_{xx}[m]=0,\qquad m\neq0.
# $$
#
# There is no preferred time scale.
#
# Correspondingly, there is no preferred frequency.
#
# Hence the power is distributed uniformly across frequency.
#
# ### Important conceptual trap
#
# Students may generate 1024 white-noise samples, calculate an FFT, and expect a flat line.
#
# They will not get one.
#
# They will get something that looks extremely jagged.
#
# That does **not** contradict the statement
#
# $$
# S_{xx}(e^{j\omega})=\sigma^2.
# $$
#
# The theoretical PSD is an ensemble property.
#
# The periodogram is one random estimate from one realization.
#
# That distinction is central to the second half of the lecture.
#
# ---
#
# # 6. PSD through an LTI system
#
# For
#
# $$
# y[n]=h[n]*x[n],
# $$
#
# the output PSD is
#
# $$
# S_{yy}(e^{j\omega}) =
# |H(e^{j\omega})|^2S_{xx}(e^{j\omega}).
# $$
#
# This appears in the lecture as the fundamental filtering relationship for WSS random processes. 
#
# Students already know that for deterministic signals,
#
# $$
# Y(e^{j\omega}) =
# H(e^{j\omega})X(e^{j\omega}).
# $$
#
# The random-process result is closely related, except we are describing **power**.
#
# Because power is related to magnitude squared,
#
# $$
# |Y|^2 =
# |H|^2|X|^2.
# $$
#
# That is why the PSD gets multiplied by $|H|^2$, rather than $H$.
#
# ---
#
# # 7. Why does filter phase disappear?
#
# Students may ask why
#
# $$
# S_{yy} =
# |H|^2S_{xx}
# $$
#
# contains no phase.
#
# Suppose
#
# $$
# H(e^{j\omega}) =
# |H(e^{j\omega})|e^{j\phi(\omega)}.
# $$
#
# Then
#
# $$
# H(e^{j\omega})H^*(e^{j\omega}) =
# |H(e^{j\omega})|^2.
# $$
#
# The phase factors cancel:
#
# $$
# e^{j\phi(\omega)}e^{-j\phi(\omega)} =
# 1.
# $$
#
# So the PSD contains information about the amount of power at each frequency, but not the output phase relationships.
#
# That does **not** mean filter phase is unimportant generally. It means the output **power spectral density alone** does not preserve that phase information.
#
# ---
#
# # 8. Filtering white noise
#
# If
#
# $$
# S_{xx}(e^{j\omega})=\sigma^2,
# $$
#
# then
#
# $$
# S_{yy}(e^{j\omega}) =
# \sigma^2|H(e^{j\omega})|^2.
# $$
#
# This is an especially useful result.
#
# It means white noise can be used as a broadband probing signal.
#
# If you excite a filter with white noise, the output PSD takes the shape of $|H|^2$.
#
# This is what the deck means when it says the filter “colors” white noise. 
#
# ### Concrete example
#
# For a lowpass filter,
#
# $$
# |H(e^{j\omega})|^2
# $$
#
# is large at low frequencies and small at high frequencies.
#
# Therefore filtered white noise will sound smoother and contain primarily low-frequency content.
#
# The input is “white.”
#
# The output may be called **colored noise** because its PSD is no longer flat.
#
# ---
#
# # 9. Concept check: filter notch
#
# The lecture asks:
#
# > If $H$ has a deep notch at $\omega_0$, what should happen to $S_{yy}$ at $\omega_0$?
#
# The best answer is:
#
# $$
# S_{yy}(e^{j\omega_0}) =
# S_{xx}(e^{j\omega_0})
# |H(e^{j\omega_0})|^2.
# $$
#
# If
#
# $$
# |H(e^{j\omega_0})|\approx0,
# $$
#
# then
#
# $$
# S_{yy}(e^{j\omega_0})\approx0.
# $$
#
# So the output PSD should exhibit a corresponding deep notch. 
#
# ### Excellent student answer
#
# > Because the output PSD equals the input PSD multiplied by $|H|^2$, a deep notch in the filter strongly suppresses the output power near that frequency. If the filter has an ideal zero at $\omega_0$, the output PSD is zero there regardless of the nonzero input PSD.
#
# That is stronger than simply saying “the PSD goes down” because it connects the result directly to the equation.
#
# ---
#
# # 10. The transition from theory to data
#
# The lecture then makes its crucial transition.
#
# The theoretical PSD contains an expectation:
#
# $$
# r_{xx}[m] =
# E\{x[n]x^*[n+m]\}.
# $$
#
# But we generally do not have an ensemble of infinitely many realizations.
#
# We may have only one measured record:
#
# $$
# x[0],x[1],\ldots,x[N-1].
# $$
#
# So we need an **estimator**. 
#
# This distinction is worth making explicit:
#
# **PSD:**
#
# $$
# S_{xx}(e^{j\omega})
# $$
#
# is a property of the random process.
#
# **Estimate:**
#
# $$
# \hat S_{xx}(e^{j\omega})
# $$
#
# is a random quantity calculated from finite observed data.
#
# Even if the underlying process is exactly the same, two experiments produce different estimates.
#
# ---
#
# # 11. The periodogram
#
# The basic periodogram is
#
# $$
# I(\omega) =
# \frac{1}{N}
# \left|
# \sum_{n=0}^{N-1}
# x[n]e^{-j\omega n}
# \right|^2.
# $$
#
# If
#
# $$
# X_N(e^{j\omega}) =
# \sum_{n=0}^{N-1}
# x[n]e^{-j\omega n},
# $$
#
# then
#
# $$
# I(\omega) =
# \frac{1}{N}|X_N(e^{j\omega})|^2.
# $$
#
# This is the finite-record PSD estimate introduced in the lecture. 
#
# ### Simple mental model
#
# Students can think of it as:
#
# 1. take the finite data record;
# 2. calculate its Fourier transform;
# 3. square the magnitude;
# 4. normalize by record length.
#
# In software, the FFT makes Step 2 efficient.
#
# ---
#
# # 12. Periodogram versus FFT
#
# It is useful to make the distinction very explicit:
#
# The FFT produces complex numbers:
#
# $$
# X[k].
# $$
#
# The periodogram uses their squared magnitude:
#
# $$
# I[k] =
# \frac{1}{N}|X[k]|^2.
# $$
#
# Therefore the FFT is a computational algorithm.
#
# The periodogram is a statistical spectral estimator.
#
# Students often casually use “FFT” and “spectrum” interchangeably, so this distinction helps.
#
# ---
#
# # 13. Why normalization conventions cause confusion
#
# The deck correctly warns students that normalization conventions differ. 
#
# For example, software may return:
#
# * two-sided PSD,
# * one-sided PSD,
# * spectrum rather than spectral density,
# * values per Hz,
# * values per rad/sample,
# * linear power,
# * or decibels.
#
# For real signals, a one-sided PSD often doubles most positive-frequency bins so that integrating only from
#
# $$
# 0\leq f\leq \frac{F_s}{2}
# $$
#
# still produces the full signal power.
#
# A good practical rule for students is:
#
# > Before interpreting a spectral plot quantitatively, check the horizontal units, vertical units, sidedness, and normalization.
#
# ---
#
# # 14. Why tones leak into nearby bins
#
# Observing only $N$ samples is mathematically equivalent to multiplying an infinite signal by a finite window.
#
# For a rectangular observation window,
#
# $$
# x_{\text{obs}}[n] =
# x[n]w[n].
# $$
#
# Multiplication in time becomes convolution in frequency:
#
# $$
# X_{\text{obs}}(e^{j\omega}) =
# \frac{1}{2\pi}
# X(e^{j\omega})*W(e^{j\omega}).
# $$
#
# Therefore an ideal spectral line does not remain infinitely narrow.
#
# Instead, it gets smeared according to the spectrum of the observation window.
#
# That is **spectral leakage**. 
#
# ---
#
# # 15. A very useful leakage example
#
# Suppose
#
# $$
# x[n] =
# \cos\left(2\pi\frac{k_0}{N}n\right).
# $$
#
# If $k_0$ is an integer, the sinusoid completes an integer number of cycles in the record.
#
# Its frequency aligns exactly with a DFT bin.
#
# Most of the spectral energy appears in the expected bins.
#
# Now change the frequency slightly:
#
# $$
# x[n] =
# \cos\left(
# 2\pi\frac{k_0+0.3}{N}n
# \right).
# $$
#
# Now the record does not contain an integer number of cycles.
#
# At the artificial boundary between the end and beginning of the record, the periodic extension assumed by the DFT has a discontinuity.
#
# The resulting DFT spreads energy across many bins.
#
# This is one of the most visually effective demonstrations of spectral leakage.
#
# ---
#
# # 16. Window trade-off
#
# A tapered window such as Hann or Hamming reduces sidelobe levels.
#
# That means less distant leakage.
#
# But the main lobe becomes wider.
#
# So there is a tradeoff:
#
# > Lower sidelobes $\leftrightarrow$ wider main lobe.
#
# Or, more intuitively:
#
# > Better suppression of leakage comes at the cost of reduced ability to distinguish closely spaced frequencies.
#
# That is what the slide means when it says tapered windows reduce sidelobes but widen the main lobe. 
#
# ---
#
# # 17. Frequency resolution and record length
#
# For $N$ samples collected at sampling rate $F_s$, the DFT frequency spacing is
#
# $$
# \Delta f =
# \frac{F_s}{N}.
# $$
#
# So increasing $N$ makes the frequency grid finer. 
#
# ### Example
#
# Suppose
#
# $$
# F_s=1000\text{ Hz}.
# $$
#
# For
#
# $$
# N=1000,
# $$
#
# we have
#
# $$
# \Delta f=1\text{ Hz}.
# $$
#
# For
#
# $$
# N=5000,
# $$
#
# we have
#
# $$
# \Delta f=0.2\text{ Hz}.
# $$
#
# So longer observation gives finer spectral detail.
#
# ---
#
# # 18. But resolution means more than bin spacing
#
# Students often equate
#
# $$
# \Delta f=\frac{F_s}{N}
# $$
#
# with “frequency resolution.”
#
# That is useful shorthand, but there are really at least two issues:
#
# 1. **frequency-grid spacing**, and
# 2. **ability to distinguish two nearby spectral components**.
#
# The second depends on window main-lobe width as well.
#
# A finer grid does not automatically mean two nearby tones can actually be resolved.
#
# This distinction becomes especially important when discussing zero-padding.
#
# ---
#
# # 19. What zero-padding really does
#
# Suppose the original data contains $N$ samples.
#
# If we calculate an $N$-point FFT, we sample the underlying finite-record DTFT at $N$ frequencies.
#
# If instead we append zeros and calculate a $4N$-point FFT, we obtain more closely spaced samples of the **same underlying finite-record spectrum**.
#
# We have not obtained any additional measured data.
#
# Therefore zero-padding produces a smoother-looking curve but does not fundamentally improve spectral resolution.
#
# That is the misconception addressed directly by the lecture. 
#
# ### Excellent analogy
#
# Imagine drawing a smooth curve through 20 measurements.
#
# Computing more interpolation points makes the displayed curve look smoother.
#
# But it does not mean you took more measurements.
#
# Zero-padding is similar.
#
# ---
#
# # 20. What actually improves resolution?
#
# Collecting data for a longer physical time.
#
# If
#
# $$
# T_{\text{obs}} =
# \frac{N}{F_s},
# $$
#
# then roughly,
#
# $$
# \Delta f
# \sim
# \frac{1}{T_{\text{obs}}}.
# $$
#
# So observing the signal longer provides genuinely more information about fine frequency structure.
#
# This is different from simply asking the computer to evaluate the FFT at additional frequencies.
#
# ---
#
# # 21. The surprising statistical flaw of the periodogram
#
# This is perhaps the most important part of the lecture.
#
# Students may naturally expect:
#
# > If I collect more samples, eventually the periodogram should become almost exactly equal to the true PSD.
#
# That intuition is not generally correct for the **raw periodogram at a fixed frequency**.
#
# A statistically consistent estimator would satisfy approximately
#
# $$
# \operatorname{Var}\{\hat S_{xx}\}
# \rightarrow 0
# $$
#
# as
#
# $$
# N\rightarrow\infty.
# $$
#
# But the raw periodogram does not generally have this property. 
#
# ---
#
# # 22. Why the periodogram remains noisy
#
# For white Gaussian noise, a DFT coefficient at a non-DC frequency is still a random variable.
#
# Its real and imaginary parts are approximately Gaussian.
#
# Therefore its squared magnitude behaves roughly like an exponential random variable.
#
# The lecture summarizes this as
#
# $$
# E\{I\}\approx\sigma^2
# $$
#
# and
#
# $$
# \operatorname{Var}\{I\}\approx\sigma^4.
# $$
#
# Hence the coefficient of variation is roughly
#
# $$
# \frac{\sqrt{\operatorname{Var}\{I\}}}
# {E\{I\}}
# \approx
# \frac{\sigma^2}{\sigma^2} =
# 1.
# $$
#
# That means the size of the fluctuations is of the same order as the mean itself. 
#
# That is a **huge amount of relative scatter**.
#
# ---
#
# # 23. A numerical intuition
#
# Suppose the true PSD level is
#
# $$
# S_{xx}=1.
# $$
#
# You might reasonably hope that a long-record estimate looks like
#
# $$
# 0.99,\;1.01,\;1.00,\;1.02,\ldots
# $$
#
# across neighboring bins.
#
# A raw periodogram can instead look something more like
#
# $$
# 0.2,\;2.1,\;0.5,\;1.8,\;0.1,\;0.9,\;3.0,\ldots
# $$
#
# even though its expected value is around $1$.
#
# The individual spectral estimates remain highly variable.
#
# That jagged appearance is not necessarily evidence of real spectral structure.
#
# ---
#
# # 24. Why does larger $N$ still help if variance remains large?
#
# This is an important subtlety.
#
# Increasing $N$ does several useful things:
#
# * increases observation time,
# * makes the frequency grid finer,
# * narrows the rectangular-window main lobe,
# * gives better ability to locate tones and spectral features.
#
# But it does **not**, by itself, eliminate raw-periodogram variance.
#
# So there are two fundamentally different goals:
#
# ### Goal 1: Better spectral detail
#
# Increase record length.
#
# ### Goal 2: Lower statistical variance
#
# Average independent or approximately independent spectral estimates.
#
# This distinction sets up Bartlett and Welch methods in Lecture 18.
#
# ---
#
# # 25. What “inconsistent” does not mean
#
# The lecture correctly emphasizes that calling the periodogram inconsistent does not mean it is useless. 
#
# It remains extremely useful for:
#
# * detecting strong sinusoidal components,
# * identifying approximate spectral bands,
# * visual exploration,
# * seeing filter characteristics,
# * and serving as the basis for better PSD estimators.
#
# “Inconsistent” has a precise statistical meaning:
#
# > Its variance at a fixed frequency does not generally approach zero as the record length becomes infinite.
#
# It is not a qualitative insult about the estimator.
#
# ---
#
# # 26. Concept check: two records from the same process
#
# The slide asks:
#
# > Two long records come from the same zero-mean WSS process. Their raw periodograms look noticeably different. Is that evidence that the process PSD changed?
#
# The intended answer is **no, not necessarily**. 
#
# ### Good student answer
#
# > No. The periodogram itself is a random estimator with high variance. Two realizations of exactly the same WSS process can therefore have substantially different raw periodograms. We would need averaging, confidence intervals, or some statistical test before concluding that the underlying PSD changed.
#
# ### Excellent answer
#
# > The underlying PSD is an ensemble property, whereas each periodogram is computed from one finite realization. The difference between the plots could simply reflect estimator variance rather than a change in the process.
#
# The second answer demonstrates that the student understands the deeper process-versus-estimate distinction.
#
# ---
#
# # 27. A helpful analogy for students
#
# Suppose a six-sided die is fair.
#
# Its true mean is
#
# $$
# E\{X\}=3.5.
# $$
#
# Roll it four times and calculate the sample mean.
#
# You may get
#
# $$
# 2.75.
# $$
#
# Repeat the experiment and you may get
#
# $$
# 4.5.
# $$
#
# The true probability distribution did not change.
#
# The **estimate** changed because the observations were random.
#
# The same idea applies to PSD estimation, except a periodogram estimates an entire function of frequency rather than one scalar parameter.
#
# ---
#
# # 28. The four coding demonstrations
#
# The lecture identifies four things students should verify computationally. 
#
# ## Demo 1: White-noise autocorrelation and PSD
#
# Generate
#
# $$
# x[n]\sim\mathcal{N}(0,\sigma^2).
# $$
#
# Estimate its autocorrelation.
#
# Students should see approximately
#
# $$
# \hat r_{xx}[0]\approx\sigma^2
# $$
#
# and
#
# $$
# \hat r_{xx}[m]\approx0,
# \qquad m\neq0.
# $$
#
# The periodogram should fluctuate around a roughly flat average level.
#
# ### What students should say
#
# > Individual frequencies vary considerably, but there is no persistent preferred spectral region. The theoretical PSD is flat even though the finite-record estimate is not.
#
# ---
#
# # 29. Demo 2: Filter white noise
#
# Generate white noise and pass it through, for example, a lowpass filter.
#
# Then compare
#
# $$
# \hat S_{yy}(e^{j\omega})
# $$
#
# to
#
# $$
# |H(e^{j\omega})|^2.
# $$
#
# Students should notice that the periodogram is noisy but its broad shape follows the filter response.
#
# ### Good answer
#
# > The individual periodogram samples fluctuate randomly, but the filter determines the systematic frequency-dependent envelope.
#
# That word **systematic** is useful: $|H|^2$ gives the deterministic trend, while the random data produce fluctuations around it.
#
# ---
#
# # 30. Demo 3: Increase $N$
#
# Generate many independent white-noise realizations.
#
# For each, compute a raw periodogram.
#
# Compare results for, say,
#
# $$
# N=256,
# $$
#
# $$
# N=2048,
# $$
#
# and
#
# $$
# N=16384.
# $$
#
# Students will see increasingly fine frequency detail, but individual periodograms remain visibly noisy.
#
# ### Strong conclusion
#
# > Increasing $N$ makes the estimate more finely localized in frequency but does not make each periodogram ordinate concentrate tightly around the true PSD.
#
# ---
#
# # 31. Demo 4: Zero-padding
#
# Take exactly the same record.
#
# Calculate:
#
# * a 256-point FFT,
# * a 1024-point zero-padded FFT,
# * a 4096-point zero-padded FFT.
#
# The plots become progressively smoother.
#
# But all three contain exactly the same measured information.
#
# ### Good answer
#
# > Zero-padding increases the density of frequency samples used to display the finite-record DTFT. It improves interpolation and makes spectral peaks easier to visualize, but it does not improve the underlying information-limited resolving capability.
#
# ---
#
# # 32. A particularly good two-tone example
#
# Suppose
#
# $$
# F_s=1000\text{ Hz}
# $$
#
# and the record length is
#
# $$
# N=100.
# $$
#
# Then
#
# $$
# T_{\text{obs}}=0.1\text{ s}
# $$
#
# and
#
# $$
# \Delta f=10\text{ Hz}.
# $$
#
# Now create
#
# $$
# x[n] =
# \cos\left(2\pi100\frac{n}{1000}\right)
# +
# \cos\left(2\pi103\frac{n}{1000}\right).
# $$
#
# A very large zero-padded FFT may make the spectrum look beautifully smooth.
#
# But because the original record lasts only $0.1$ s, distinguishing two tones only $3$ Hz apart remains difficult.
#
# Now increase the actual observation length to
#
# $$
# N=2000.
# $$
#
# Then
#
# $$
# T_{\text{obs}}=2\text{ s}
# $$
#
# and
#
# $$
# \Delta f=0.5\text{ Hz}.
# $$
#
# Now the two tones can be distinguished much more readily.
#
# This cleanly shows the difference between **zero-padding** and **collecting more data**.
#
# ---
#
# # 33. A deeper interpretation of the whole lecture
#
# There are really three different objects students need to keep straight:
#
# ### 1. True process PSD
#
# $$
# S_{xx}(e^{j\omega}).
# $$
#
# This is a theoretical statistical property of the random process.
#
# ### 2. Finite-record Fourier transform
#
# $$
# X_N(e^{j\omega}) =
# \sum_{n=0}^{N-1}
# x[n]e^{-j\omega n}.
# $$
#
# This depends on the particular realization that was observed.
#
# ### 3. Periodogram
#
# $$
# I(\omega) =
# \frac{1}{N}|X_N(e^{j\omega})|^2.
# $$
#
# This is an estimator of the PSD.
#
# A strong student should be able to explain why these are related without saying they are the same thing.
#
# ---
#
# # 34. What would be a good answer to the lecture's core question?
#
# The core question is essentially:
#
# > What can one finite realization tell us about the underlying distribution of signal power over frequency?
#
# A strong answer would be:
#
# > A finite realization allows us to form a spectral estimate such as the periodogram. It can reveal broad spectral structure, tones, and the effects of filtering, but because the estimate is random and has high variance, one finite record does not uniquely determine the underlying PSD. Longer records improve frequency detail, while averaging is needed to reduce statistical variance.
#
# That answer captures almost the entire lecture.
#
# ---
#
# # 35. Good answer to the final exit ticket
#
# The deck closes with:
#
# > Why does a longer record not automatically make a raw periodogram a low-variance PSD estimator? 
#
# ### Minimum acceptable answer
#
# > A longer record improves frequency resolution, but the raw periodogram still has large random fluctuations.
#
# ### Good answer
#
# > Increasing $N$ makes the frequency grid finer and generally improves spectral resolution, but the variance of the raw periodogram at a fixed frequency does not generally approach zero. Therefore the estimator remains noisy even for long records.
#
# ### Excellent answer
#
# > For a raw periodogram, increasing $N$ gives us more frequency detail but does not average away the randomness of each spectral ordinate. For white Gaussian noise, for example, the periodogram value at a frequency has relative standard deviation of roughly the same order as its mean. Thus variance reduction requires averaging, as in Bartlett or Welch estimation, rather than simply increasing the FFT length.
#
# ---
#
# # 36. The most important misconceptions to watch for
#
# For this lecture, I would listen specifically for these statements from students:
#
# **“The PSD is just $|\text{FFT}|^2$.”**
#
# Better:
#
# > The periodogram is based on the squared FFT magnitude; the PSD is a property of the underlying stochastic process.
#
# **“White noise should have a perfectly flat FFT.”**
#
# Better:
#
# > Its theoretical PSD is flat. Any finite realization produces a randomly fluctuating spectral estimate.
#
# **“More FFT points give better frequency resolution.”**
#
# Better:
#
# > More measured samples can improve resolution. More zero-padded FFT points merely sample the existing finite-record spectrum more densely.
#
# **“Longer data makes the periodogram smooth.”**
#
# Better:
#
# > Longer data improves frequency detail, but the raw periodogram remains a high-variance estimator.
#
# **“Windowing removes leakage.”**
#
# Better:
#
# > Windowing changes the leakage pattern. Tapered windows reduce sidelobes but broaden the main lobe.
#
# **“If two periodograms look different, the source must have changed.”**
#
# Better:
#
# > Differences can arise simply from estimator variance, even when both records come from the same stationary process.
#
# ---
#
# ## The conceptual thread I would emphasize throughout the 75 minutes
#
# A useful repeated phrase for students is:
#
# > **Resolution and variance are different problems.**
#
# Longer records primarily help with **frequency detail**.
#
# Averaging primarily helps with **statistical reliability**.
#
# And zero-padding primarily helps with **display/interpolation**.
#
# If students leave Lecture 17 able to clearly distinguish those three effects, they will be very well prepared for Lecture 18 and for understanding why Welch's method is so widely used in practical DSP.
#
# -------------------------
#

# %% [markdown]
# ## Week 9, Lecture 18
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT (Medium):
#
# Below is a more detailed instructor-oriented explanation of Lecture 18. The central theme is the practical response to the problem established in Lecture 17: **the raw periodogram has high variance, so we reduce variance by averaging spectral estimates—but that averaging costs frequency resolution.** The lecture then connects Bartlett and Welch estimators to practical ML feature construction. 
#
# ## 1. What is the big idea of this lecture?
#
# Lecture 17 established that a raw periodogram can have good frequency detail but remain statistically noisy.
#
# Lecture 18 asks:
#
# > Can we deliberately give up some frequency resolution in exchange for a more stable estimate of the PSD?
#
# That is the central design trade-off.
#
# If we have one long record of $N$ samples, we could compute one periodogram using all $N$ samples.
#
# That gives us:
#
# * long observation time,
# * fine frequency grid,
# * relatively narrow spectral peaks,
# * but high variance.
#
# Instead, we can split the record into several shorter pieces.
#
# Each short piece gives us its own periodogram.
#
# Then we average those periodograms.
#
# This gives:
#
# * several estimates to average,
# * lower variance,
# * but shorter effective observation time per estimate,
# * therefore poorer frequency resolution.
#
# That trade-off is the foundation of both Bartlett's method and Welch's method. 
#
# ---
#
# # 2. Why does averaging help?
#
# Suppose we have $K$ approximately independent estimates of the PSD at some frequency:
#
# $$
# I_1(f), I_2(f), \ldots, I_K(f).
# $$
#
# We form the average
#
# $$
# \hat S(f) =
# \frac{1}{K}
# \sum_{k=1}^{K}
# I_k(f).
# $$
#
# If the estimates are independent and each has variance
#
# $$
# \operatorname{Var}\{I_k(f)\} =
# \sigma_I^2,
# $$
#
# then
#
# $$
# \operatorname{Var}\{\hat S(f)\} =
# \frac{\sigma_I^2}{K}.
# $$
#
# So the standard deviation decreases approximately as
#
# $$
# \frac{1}{\sqrt K}.
# $$
#
# This is simply the familiar statistical principle that averaging independent noisy measurements reduces random variation.
#
# ### Good analogy
#
# Suppose you want to estimate the average weight of apples.
#
# Measuring one apple gives a noisy estimate of the population mean.
#
# Measuring 20 apples and averaging gives a more reliable estimate.
#
# The same logic applies here.
#
# Each periodogram is a noisy measurement of the underlying PSD.
#
# Averaging several of them suppresses some of that randomness.
#
# ---
#
# # 3. But where do the multiple estimates come from?
#
# This is the clever part.
#
# Usually we do not have $K$ completely separate experiments.
#
# We may have only one long signal record:
#
# $$
# x[0],x[1],\ldots,x[N-1].
# $$
#
# Bartlett's idea is:
#
# > Break the long record into shorter sections and treat the sections as multiple approximately independent observations.
#
# If the record is divided into $K$ non-overlapping segments, each segment has length
#
# $$
# L=\frac{N}{K}.
# $$
#
# So we trade one long estimate for $K$ shorter estimates. 
#
# This is where the resolution-versus-variance trade-off comes from.
#
# More segments means more averaging.
#
# But more segments also means each segment is shorter.
#
# ---
#
# # 4. Bartlett's method step by step
#
# Suppose
#
# $$
# N=KL.
# $$
#
# Split the data into $K$ non-overlapping segments:
#
# $$
# x_0[n], x_1[n], \ldots, x_{K-1}[n].
# $$
#
# Each segment contains $L$ samples.
#
# For segment $k$, calculate a periodogram:
#
# $$
# I_k(f).
# $$
#
# Then average them:
#
# $$
# \hat S_B(f) =
# \frac{1}{K}
# \sum_{k=0}^{K-1}
# I_k(f).
# $$
#
# That is Bartlett's estimator.
#
# The important point is that Bartlett does not fundamentally invent a new spectral quantity.
#
# It takes the periodogram from Lecture 17 and adds **averaging**.
#
# ---
#
# # 5. What is the price Bartlett pays?
#
# Suppose we originally had $N$ samples.
#
# The full-record DFT frequency spacing is roughly
#
# $$
# \Delta f_{\text{full}} =
# \frac{F_s}{N}.
# $$
#
# After dividing the record into segments of length $L$,
#
# $$
# \Delta f_{\text{segment}} =
# \frac{F_s}{L}.
# $$
#
# Because $L<N$,
#
# $$
# \frac{F_s}{L}
# >
# \frac{F_s}{N}.
# $$
#
# So the frequency grid becomes coarser.
#
# More importantly, the effective observation time per segment decreases.
#
# Since
#
# $$
# T_{\text{segment}} =
# \frac{L}{F_s},
# $$
#
# shorter segments lead to wider spectral main lobes.
#
# Therefore nearby tones become harder to distinguish.
#
# ---
#
# # 6. Bartlett numerical example
#
# The deck uses
#
# $$
# N=4096
# $$
#
# samples and
#
# $$
# K=8.
# $$
#
# Then
#
# $$
# L =
# \frac{4096}{8} =
# 512.
# $$
#
# If
#
# $$
# F_s=1000\text{ Hz},
# $$
#
# the segment-based bin spacing is
#
# $$
# \Delta f =
# \frac{1000}{512}
# \approx
# 1.95\text{ Hz}.
# $$
#
# Using the entire 4096-sample record instead would give
#
# $$
# \Delta f =
# \frac{1000}{4096}
# \approx
# 0.244\text{ Hz}.
# $$
#
# So Bartlett changes the situation from approximately $0.244$ Hz spacing to $1.95$ Hz spacing.
#
# In exchange, it obtains eight periodograms that can be averaged. 
#
# ### Good interpretation
#
# A strong student answer would be:
#
# > Bartlett's method sacrifices roughly a factor of eight in nominal frequency spacing so that eight spectral estimates can be averaged, reducing variance.
#
# That answer connects the arithmetic to the statistical motivation.
#
# ---
#
# # 7. Periodogram versus Bartlett versus Welch
#
# The lecture compares the three approaches conceptually. 
#
# A useful mental picture is:
#
# ### Raw periodogram
#
# Uses one long record.
#
# Result:
#
# > Detailed, but jagged.
#
# ### Bartlett
#
# Breaks the record into shorter non-overlapping sections and averages them.
#
# Result:
#
# > Smoother, but with poorer frequency resolution.
#
# ### Welch
#
# Adds two more ingredients:
#
# * taper each section with a window,
# * allow segments to overlap.
#
# Result:
#
# > Usually a more practical balance between leakage, averaging, and frequency detail.
#
# One important caution is that **smoother does not automatically mean more accurate everywhere**.
#
# A heavily averaged PSD can look beautiful while completely smearing together two physically important nearby spectral peaks.
#
# ---
#
# # 8. Why Welch improves on Bartlett
#
# Welch modifies Bartlett's basic idea in three ways:
#
# 1. use overlapping segments;
# 2. apply a window to each segment;
# 3. compensate correctly for the window's power.
#
# The deck summarizes this as averaging modified periodograms. 
#
# For segment $k$,
#
# $$
# x_k[n],
# $$
#
# apply a window:
#
# $$
# x_{k,w}[n] =
# w[n]x_k[n].
# $$
#
# Then compute
#
# $$
# X_k(f) =
# \sum_n
# w[n]x_k[n]
# e^{-j2\pi fn/F_s}.
# $$
#
# The modified periodogram has the form
#
# $$
# I_k(f)
# \propto
# \frac{|X_k(f)|^2}
# {\sum_n w^2[n]}.
# $$
#
# Finally,
#
# $$
# \hat S_W(f) =
# \frac{1}{K}
# \sum_k I_k(f).
# $$
#
# The normalization involving
#
# $$
# \sum_n w^2[n]
# $$
#
# is necessary because the window changes the signal energy.
#
# Without compensating for that change, the PSD scale would be wrong. 
#
# ---
#
# # 9. Why do we window the segments?
#
# Suppose each segment is simply truncated with a rectangular window.
#
# That produces relatively large spectral sidelobes.
#
# A tone at one frequency can leak substantial energy into other frequency bins.
#
# Using a Hann or Hamming window reduces those sidelobes.
#
# The price is a wider main lobe.
#
# So the window creates another trade-off:
#
# > Lower leakage versus poorer separation of closely spaced frequencies.
#
# The deck summarizes this as:
#
# * rectangular window: narrow main lobe, high sidelobes;
# * Hann/Hamming: lower sidelobes, wider main lobe. 
#
# ---
#
# # 10. A useful two-tone window example
#
# Suppose
#
# $$
# x[n] =
# \cos\left(2\pi100\frac{n}{1000}\right)
# +
# 0.05
# \cos\left(2\pi115\frac{n}{1000}\right).
# $$
#
# The 100 Hz tone is much stronger than the 115 Hz tone.
#
# With a rectangular window, the strong tone's sidelobes may leak enough energy near 115 Hz to partially mask the smaller tone.
#
# A Hann window greatly reduces the sidelobes.
#
# That can make the weak tone easier to detect.
#
# However, if the tones were at 100 Hz and 102 Hz instead, the broader main lobe of the Hann window might cause them to merge.
#
# This illustrates why there is no universally best window.
#
# ---
#
# # 11. What overlap does
#
# Welch allows neighboring segments to reuse samples.
#
# Suppose each segment has length $L$.
#
# With no overlap, the starts might occur at
#
# $$
# 0,\;L,\;2L,\;3L,\ldots
# $$
#
# With 50% overlap, the starts occur at
#
# $$
# 0,\;\frac{L}{2},\;L,\;\frac{3L}{2},\ldots
# $$
#
# So we get more segments from the same total record.
#
# That gives us more periodograms to average.
#
# This sounds like free variance reduction.
#
# But it is not entirely free.
#
# Because overlapping segments contain many of the same samples, their periodograms are statistically correlated.
#
# Therefore the simple relationship
#
# $$
# \operatorname{Var}\{\hat S\}
# \propto
# \frac{1}{K}
# $$
#
# does not hold exactly. 
#
# ---
#
# # 12. Why 50% overlap with Hann is so common
#
# A Hann window strongly downweights the samples near the segment boundaries.
#
# If we used non-overlapping Hann windows, some samples would repeatedly receive very little weight.
#
# By shifting the next window halfway forward, those samples may appear near the center of another window and receive much more weight.
#
# So overlap helps use the available data more efficiently.
#
# This is one reason the combination
#
# > Hann window + 50% overlap
#
# is such a common practical starting point.
#
# The slide correctly calls it a strong default rather than a law. 
#
# ---
#
# # 13. Why 75% overlap does not give four times the information
#
# Suppose changing from 0% overlap to 75% overlap increases the number of segments dramatically.
#
# Students may conclude:
#
# > I now have many more periodograms, so variance must decrease according to $1/K$.
#
# Not exactly.
#
# Adjacent highly overlapping segments may share 75% of their samples.
#
# Therefore their spectral estimates contain substantially redundant information.
#
# The effective number of independent averages is smaller than the literal segment count.
#
# A strong answer would be:
#
# > More overlap gives more estimates, but not proportionally more independent information because neighboring segments are increasingly correlated.
#
# ---
#
# # 14. Segment length is the most important design parameter
#
# This is probably the central practical idea in the lecture.
#
# Welch has several parameters:
#
# * segment length,
# * window,
# * overlap,
# * FFT length.
#
# But segment length is usually the main resolution-versus-variance control.
#
# The deck summarizes the trade-off clearly. 
#
# ### Long segments
#
# Advantages:
#
# * finer frequency grid,
# * narrower main lobes,
# * better ability to resolve close peaks.
#
# Disadvantage:
#
# * fewer segments,
# * fewer averages,
# * higher variance.
#
# ### Short segments
#
# Advantages:
#
# * more segments,
# * more averaging,
# * lower variance.
#
# Disadvantage:
#
# * broader peaks,
# * poorer spectral resolution.
#
# ---
#
# # 15. A concrete segment-length example
#
# Suppose
#
# $$
# F_s=1000\text{ Hz}
# $$
#
# and we have
#
# $$
# N=10000
# $$
#
# samples.
#
# Consider two possible Welch configurations.
#
# ### Configuration A
#
# $$
# L=1000.
# $$
#
# Then nominal spacing is
#
# $$
# \Delta f =
# \frac{1000}{1000} =
# 1\text{ Hz}.
# $$
#
# Without overlap, we obtain roughly 10 segments.
#
# ### Configuration B
#
# $$
# L=250.
# $$
#
# Then
#
# $$
# \Delta f =
# \frac{1000}{250} =
# 4\text{ Hz}.
# $$
#
# Without overlap, we obtain roughly 40 segments.
#
# Configuration B allows roughly four times as many averages.
#
# So it should produce a smoother estimate.
#
# But if the signal contains tones at 100 Hz and 103 Hz, Configuration A is much more likely to distinguish them.
#
# That is the central Welch design trade-off.
#
# ---
#
# # 16. A good quantitative design strategy
#
# The lecture provides an especially useful design recipe. 
#
# Start with the signal, not with an arbitrary Welch default.
#
# Suppose important spectral features are separated by approximately
#
# $$
# \Delta f_{\text{feature}}.
# $$
#
# Choose $L$ such that
#
# $$
# \frac{F_s}{L}
# $$
#
# is comfortably smaller than that separation.
#
# For example, suppose
#
# $$
# F_s=2000\text{ Hz}
# $$
#
# and important features may be separated by
#
# $$
# 10\text{ Hz}.
# $$
#
# We might want
#
# $$
# \frac{F_s}{L}
# \leq
# 2\text{ Hz}.
# $$
#
# Then
#
# $$
# L \geq
# \frac{2000}{2} =
# 1000.
# $$
#
# So a segment length around 1024 would be a reasonable starting point.
#
# Then choose:
#
# * the window based on leakage concerns,
# * the overlap based on averaging efficiency,
# * and validate the result using repeated observations.
#
# ---
#
# # 17. Why "make it smooth" is a bad design objective
#
# Students often look at a PSD estimate and think:
#
# > That looks too jagged. I should smooth it more.
#
# But excessive smoothing can destroy important information.
#
# Suppose a true PSD contains two narrow peaks:
#
# $$
# f_1=120\text{ Hz}
# $$
#
# and
#
# $$
# f_2=126\text{ Hz}.
# $$
#
# A heavily averaged or heavily smoothed estimate might show one broad peak around
#
# $$
# 123\text{ Hz}.
# $$
#
# That curve looks clean and stable.
#
# But it is scientifically less informative because it lost the fact that there are two components.
#
# So a better question is:
#
# > Does the estimator preserve the spectral detail required by my application while providing acceptable variance?
#
# That is a much stronger engineering criterion than visual smoothness.
#
# ---
#
# # 18. Concept check: tones at 120 Hz and 126 Hz
#
# The slide asks:
#
# > You must distinguish tones at 120 Hz and 126 Hz in noisy data sampled at 1 kHz. Which change is most likely to help?
#
# The choices include:
#
# * shorter Welch segments,
# * longer Welch segments,
# * more zero-padding,
# * heavier frequency smoothing.
#
# The correct answer is **longer Welch segments**. 
#
# ### Why?
#
# The tone separation is
#
# $$
# 126-120 =
# 6\text{ Hz}.
# $$
#
# Resolving them requires sufficiently narrow spectral main lobes.
#
# Longer segments mean longer observation time:
#
# $$
# T_{\text{seg}} =
# \frac{L}{F_s}.
# $$
#
# Increasing $L$ gives narrower spectral features.
#
# ### Why not shorter segments?
#
# Shorter segments worsen frequency resolution.
#
# ### Why not zero-padding?
#
# Zero-padding merely samples the same finite-record spectrum more densely.
#
# It does not fundamentally narrow the spectral main lobe.
#
# ### Why not heavier smoothing?
#
# Smoothing would tend to merge the peaks further.
#
# ---
#
# # 19. What would be an excellent student answer?
#
# > I would increase the Welch segment length. The two tones are only 6 Hz apart, so I need better true frequency resolution. Longer segments increase observation time and narrow the spectral main lobes. Zero-padding would give a denser frequency grid but would not provide additional resolving power.
#
# That answer shows understanding rather than simply recognizing choice B.
#
# ---
#
# # 20. Welch and zero-padding are different knobs
#
# This distinction is worth emphasizing again.
#
# Suppose segment length is
#
# $$
# L=256.
# $$
#
# We could compute:
#
# $$
# N_{\text{FFT}}=256,
# $$
#
# or
#
# $$
# N_{\text{FFT}}=2048.
# $$
#
# The 2048-point FFT gives more plotted frequency samples.
#
# But the data in each Welch segment still contains only 256 measured samples.
#
# Therefore the underlying resolution is still governed largely by those 256 samples and the window main lobe.
#
# In practical software, this distinction corresponds roughly to:
#
# * `nperseg`: controls actual segment length;
# * `nfft`: controls FFT sampling density.
#
# Those are not interchangeable.
#
# ---
#
# # 21. Bias versus variance
#
# The lecture is primarily framed around resolution and variance, but there is a useful deeper statistical perspective.
#
# A PSD estimator has both:
#
# * bias,
# * variance.
#
# Windowing and averaging can reduce variance but may increase bias by smoothing spectral features.
#
# A very narrow spectral peak may become spread out.
#
# So estimator design often involves the classical bias-variance trade-off:
#
# $$
# \text{less variance}
# \quad\leftrightarrow\quad
# \text{more smoothing/bias}.
# $$
#
# For students who already know machine learning, this provides a useful conceptual bridge.
#
# It is not exactly the same as model bias and variance, but the general statistical idea is closely related.
#
# ---
#
# # 22. Welch as an ML preprocessing tool
#
# The lecture then makes a very useful transition.
#
# Instead of using the raw waveform directly,
#
# $$
# x[n],
# $$
#
# we can compute a Welch PSD:
#
# $$
# x[n]
# \rightarrow
# \hat S_{xx}(f).
# $$
#
# Then compress the PSD into physically meaningful features such as:
#
# * band powers,
# * log band powers,
# * spectral centroid,
# * roll-off frequency,
# * spectral slopes.
#
# This creates a stable feature vector for a classical classifier. 
#
# A typical pipeline might be:
#
# $$
# \text{waveform}
# \rightarrow
# \text{Welch PSD}
# \rightarrow
# \text{band powers}
# \rightarrow
# \text{standardization}
# \rightarrow
# \text{classifier}.
# $$
#
# ---
#
# # 23. Why Welch features can be useful for ML
#
# Suppose two recordings belong to the same class but differ in the exact waveform.
#
# Their individual time samples could look very different.
#
# However, their average spectral shapes might be very similar.
#
# Welch averaging suppresses some realization-specific randomness.
#
# So instead of asking the classifier to learn from thousands of highly variable waveform samples, we can give it a smaller number of stable spectral features.
#
# That is often advantageous when:
#
# * the classes are distinguished primarily by frequency content,
# * phase is unimportant,
# * time localization is unimportant,
# * the dataset is not enormous.
#
# ---
#
# # 24. Example: rotating machinery
#
# Suppose you are trying to classify motor conditions:
#
# * healthy,
# * bearing defect,
# * shaft imbalance.
#
# A vibration signal may contain characteristic frequency bands.
#
# The raw waveform might be highly variable.
#
# But band powers such as
#
# $$
# P_1 =
# \int_{20}^{40}
# S_{xx}(f)\,df
# $$
#
# and
#
# $$
# P_2 =
# \int_{80}^{120}
# S_{xx}(f)\,df
# $$
#
# could differ consistently between fault classes.
#
# That makes Welch bandpowers potentially useful ML features.
#
# ---
#
# # 25. Example: audio classification
#
# Suppose we want to classify sounds:
#
# * fan,
# * vacuum cleaner,
# * speech,
# * alarm.
#
# We might compute Welch PSDs and then calculate powers in frequency bands:
#
# $$
# P_k =
# \sum_{f\in B_k}
# \hat S_{xx}(f)\Delta f.
# $$
#
# These values summarize how spectral energy is distributed.
#
# For example:
#
# * a fan may be dominated by low-frequency tonal components;
# * an alarm may show strong narrowband peaks;
# * speech may show broader spectral structure.
#
# The resulting feature vector might be
#
# $$
# \mathbf{x} =
# [P_1,P_2,\ldots,P_M].
# $$
#
# That could then feed into an SVM or random forest.
#
# ---
#
# # 26. Why take logarithms of band power?
#
# Power values can span several orders of magnitude.
#
# Instead of using
#
# $$
# P_k,
# $$
#
# we might use
#
# $$
# 10\log_{10}(P_k+\epsilon).
# $$
#
# This compresses the dynamic range.
#
# It can also make multiplicative differences more manageable for many classical ML algorithms.
#
# This is especially useful for audio and vibration features.
#
# ---
#
# # 27. Spectral centroid example
#
# Another possible feature is spectral centroid:
#
# $$
# f_c =
# \frac{\sum_k f_k S[k]}
# {\sum_k S[k]}.
# $$
#
# This can be interpreted roughly as the "center of mass" of the spectrum.
#
# Signals dominated by high frequencies tend to have larger spectral centroids.
#
# Signals dominated by low frequencies tend to have smaller values.
#
# For audio, this often corresponds loosely to perceived "brightness."
#
# ---
#
# # 28. What information PSD features discard
#
# The slide makes an important warning: PSD features intentionally discard phase and precise temporal localization. 
#
# Consider two signals with identical magnitude spectra but different phase.
#
# They can have very different time-domain waveforms.
#
# Yet their PSDs may be identical.
#
# Similarly, suppose a signal contains one short transient.
#
# A PSD can tell us that certain frequency components were present.
#
# It may not tell us **when** they occurred.
#
# So PSD features are excellent when the class depends on long-term spectral content.
#
# They may be poor when the task depends on:
#
# * timing,
# * transient sequence,
# * phase relationships,
# * waveform morphology.
#
# ---
#
# # 29. An example where PSD features might fail
#
# Suppose the task is to distinguish:
#
# > beep followed by chirp
#
# from
#
# > chirp followed by beep.
#
# Both recordings could have nearly identical long-term PSDs.
#
# The frequency content is essentially the same.
#
# Only the temporal ordering differs.
#
# A Welch PSD may therefore fail to distinguish them.
#
# A spectrogram or another time-frequency representation would be much more appropriate.
#
# This is a nice example of the slide's warning that discarding time localization may be desirable—or fatal—depending on the task.
#
# ---
#
# # 30. The example ML pipeline
#
# The slide gives a very sensible pipeline. 
#
# For every record:
#
# 1. compute a Welch PSD using the same settings;
# 2. integrate physically meaningful frequency bands;
# 3. optionally add shape features;
# 4. standardize using training data only;
# 5. train a classifier and cross-validate.
#
# The phrase **same settings** is important.
#
# If one signal is processed using
#
# $$
# L=256
# $$
#
# and another using
#
# $$
# L=2048,
# $$
#
# their resulting features may not be directly comparable.
#
# The feature extraction pipeline must be consistent across records.
#
# ---
#
# # 31. Why training-only standardization matters
#
# Suppose a feature has mean
#
# $$
# \mu
# $$
#
# and standard deviation
#
# $$
# \sigma.
# $$
#
# Standardization is
#
# $$
# z =
# \frac{x-\mu}{\sigma}.
# $$
#
# The values of $\mu$ and $\sigma$ must be estimated from the training set only.
#
# If they are computed from the entire dataset, including the test data, then information about the test distribution has influenced the training pipeline.
#
# That is **dataset leakage**.
#
# The slide cleverly distinguishes two different meanings of leakage:
#
# ### Spectral leakage
#
# Energy from one frequency spreads to nearby frequencies because of finite observation and windowing.
#
# ### Dataset leakage
#
# Information from validation/test data leaks into model training or preprocessing.
#
# They are entirely different phenomena, but both can invalidate conclusions. 
#
# ---
#
# # 32. Why standardization is especially important for SVMs
#
# Suppose we have two features:
#
# $$
# x_1 =
# \text{band power}
# $$
#
# with numerical scale around
#
# $$
# 10^5,
# $$
#
# and
#
# $$
# x_2 =
# \text{spectral centroid}
# $$
#
# with numerical scale around
#
# $$
# 10^2.
# $$
#
# An SVM based on Euclidean geometry can become dominated by the feature with the larger numerical scale.
#
# Standardizing makes the features more comparable.
#
# Random forests are generally much less sensitive to scaling, but using a consistent pipeline is still valuable.
#
# ---
#
# # 33. A good coding demo sequence
#
# The deck suggests comparing custom Bartlett with `scipy.signal.welch` and sweeping parameters. 
#
# A particularly effective demonstration would proceed in this order.
#
# ### Step 1: Generate noisy two-tone data
#
# For example,
#
# $$
# x[n] =
# \cos\left(2\pi120\frac{n}{1000}\right)
# +
# 0.7
# \cos\left(2\pi126\frac{n}{1000}\right)
# +
# v[n].
# $$
#
# ### Step 2: Raw periodogram
#
# Students should predict:
#
# > high spectral detail, but noisy.
#
# ### Step 3: Bartlett
#
# Use several short non-overlapping segments.
#
# Students should predict:
#
# > smoother estimate, but perhaps the two peaks begin to merge.
#
# ### Step 4: Welch with Hann and 50% overlap
#
# Students should predict:
#
# > better leakage behavior and a stable estimate, with resolution determined strongly by segment length.
#
# ### Step 5: Change only segment length
#
# Compare something like
#
# $$
# L=128,\;256,\;512,\;1024.
# $$
#
# Ask students:
#
# > At what point can you clearly distinguish 120 Hz from 126 Hz?
#
# That makes the resolution trade-off concrete.
#
# ---
#
# # 34. Student prediction questions before running the notebook
#
# The deck explicitly recommends prediction before computation. 
#
# Some useful questions are:
#
# > If I cut the segment length in half, what happens to the PSD?
#
# Good answer:
#
# > I get more segments and therefore potentially lower variance, but frequency resolution becomes worse.
#
# > If I double the segment length?
#
# Good answer:
#
# > Peaks should become sharper and close frequencies easier to distinguish, but fewer segments are available for averaging, so the estimate may look noisier.
#
# > If I change overlap from 50% to 75%?
#
# Good answer:
#
# > I get more segment estimates, but they are more strongly correlated, so the effective variance reduction is less than what the raw increase in segment count suggests.
#
# > If I increase only `nfft`?
#
# Good answer:
#
# > The displayed frequency grid becomes denser, but the true spectral resolution does not fundamentally improve.
#
# ---
#
# # 35. An important subtlety: frequency spacing is not identical to resolution
#
# Suppose
#
# $$
# L=256
# $$
#
# at
#
# $$
# F_s=1000\text{ Hz}.
# $$
#
# The nominal bin spacing is
#
# $$
# \frac{1000}{256}
# \approx
# 3.91\text{ Hz}.
# $$
#
# But if a Hann window is used, the effective peak width is wider than one DFT bin.
#
# So one should not interpret
#
# $$
# 3.91\text{ Hz}
# $$
#
# as meaning that every pair of tones separated by 3.91 Hz can be cleanly resolved.
#
# Actual resolution depends on:
#
# * segment duration,
# * window main-lobe width,
# * tone amplitudes,
# * noise level.
#
# This is why the deck says to choose $F_s/L$ **comfortably smaller** than the separation of important features rather than merely equal to it. 
#
# ---
#
# # 36. A useful instructor question
#
# Ask students:
#
# > Suppose two spectral peaks are 8 Hz apart. Should I choose parameters giving $\Delta f=8$ Hz?
#
# A weak answer might say yes.
#
# A better answer is:
#
# > Probably not. The nominal frequency spacing should be significantly finer than 8 Hz because the window main lobe occupies multiple frequency samples, so equal bin spacing does not guarantee the peaks can be resolved.
#
# That tests whether they understand the difference between frequency-grid sampling and actual resolving capability.
#
# ---
#
# # 37. What is the "best" Welch configuration?
#
# There is no universal best configuration.
#
# The answer depends on the scientific question.
#
# If you are detecting two closely spaced tones, prioritize longer segments.
#
# If you only care about broad frequency bands and want reliable bandpower estimates, shorter segments and more averaging may be better.
#
# If leakage from strong tones is a major concern, a tapered window may be necessary.
#
# If the record is short, large overlap may help make efficient use of the available samples.
#
# The appropriate question is therefore not:
#
# > What settings make the nicest-looking PSD?
#
# It is:
#
# > What settings preserve the spectral information relevant to the task while achieving acceptable statistical stability?
#
# ---
#
# # 38. Good answer to the lecture's core question
#
# The lecture asks:
#
# > How much frequency detail are we willing to trade for a PSD estimate that is statistically more stable?
#
# A strong answer is:
#
# > Averaging multiple segment periodograms reduces estimator variance, but creating more averages requires shorter segments, which broadens spectral features and reduces frequency resolution. The appropriate trade-off depends on the minimum frequency separation that must be resolved and the amount of estimator variability that can be tolerated.
#
# That captures the essential engineering design problem.
#
# ---
#
# # 39. Good answer to the exit ticket
#
# The slide asks:
#
# > If a Welch PSD looks too noisy, name one change that reduces variance and state the price you pay. 
#
# ### Minimum acceptable answer
#
# > Use shorter segments so that more periodograms can be averaged. The cost is poorer frequency resolution.
#
# ### Good answer
#
# > Reduce `nperseg`. This creates more Welch segments from the same record, so more modified periodograms can be averaged and the estimate becomes less variable. The trade-off is broader spectral peaks and poorer ability to distinguish nearby frequency components.
#
# ### Another good answer
#
# > Increase overlap. This gives more segment estimates to average, which can reduce variance. However, neighboring segments become more correlated, so the variance reduction is not proportional to the total number of segments, and computational cost increases.
#
# ### Excellent answer
#
# > The most direct resolution-variance trade-off is to shorten the Welch segment length. That increases the number of averages and generally lowers variance, but it shortens the observation time of each spectral estimate, broadening the window main lobe and reducing true frequency resolution. So I would only shorten the segments as far as the application's required frequency separation allows.
#
# ---
#
# # 40. The conceptual thread I would emphasize throughout the lecture
#
# I would keep returning to three different knobs:
#
# **Segment length controls the main resolution-variance trade-off.**
#
# $$
# \text{long segments}
# \rightarrow
# \text{better resolution, fewer averages}
# $$
#
# $$
# \text{short segments}
# \rightarrow
# \text{poorer resolution, more averages}
# $$
#
# **Window controls the leakage-resolution trade-off.**
#
# $$
# \text{lower sidelobes}
# \leftrightarrow
# \text{wider main lobe}
# $$
#
# **Overlap controls averaging efficiency.**
#
# $$
# \text{more overlap}
# \rightarrow
# \text{more estimates}
# $$
#
# but also
#
# $$
# \text{more overlap}
# \rightarrow
# \text{more correlation between estimates}.
# $$
#
# Those are three distinct effects.
#
# If students understand which parameter controls which phenomenon, they are much less likely to treat Welch as a black-box function call.
#
# The most important takeaway is probably:
#
# > **Welch's method is not simply "a smoother periodogram." It is an estimator design strategy in which segment length, window choice, and overlap are deliberately chosen to balance spectral resolution, leakage, and statistical variance.**
#
# That distinction is especially useful for your course because it connects classical DSP estimator design directly to the later ML idea that preprocessing and feature extraction should be chosen according to the information the task actually requires.
#
# -----------------------
#

# %% [markdown]
# ## Week 9, Lectures 17 and 18, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# Gemini Pro:
#
# Here are strong exit ticket questions and answers for these lectures, calibrated for senior and graduate-level understanding. They bridge the gap between classical theoretical signal processing and modern machine learning feature extraction.
#
# ### Lecture 17: Power Spectral Density and the Periodogram
#
# **Question 1: The Statistical Flaw** (Directly from your slide wrap-up)
#
# * **Question:** Why does a longer observation record not automatically make a raw periodogram a low-variance PSD estimator?
#
#
# * **Answer:** Increasing the record length $N$ yields a finer frequency grid by reducing the FFT bin spacing, but the estimator remains inconsistent. For a white noise process, the squared magnitude of the DFT coefficient at a non-DC bin remains random, and its coefficient of variation stays around 1, meaning the variance does not shrink as $N \rightarrow \infty$.
#
#
#
# **Question 2: Filtering and Colored Noise** (Adapted from your concept check)
#
# * **Question:** If wide-sense stationary white noise is passed through a stable LTI filter that has a deep notch at frequency $\omega_0$, what is the expected shape of the output PSD at $\omega_0$?
#
#
# * **Answer:** The output PSD will also feature a deep notch at $\omega_0$. For a stable LTI system, the output PSD is scaled by the squared magnitude of the filter's frequency response: $S_{yy}(e^{j\omega}) = S_{xx}(e^{j\omega}) \vert{}H(e^{j\omega})\vert{}^2$. Therefore, the filter imposes its spectral structure on the white noise.
#
#
#
# **Question 3: Interpreting the Wiener-Khinchin Theorem**
#
# * **Question:** How does the Wiener-Khinchin theorem relate the time-domain properties of a wide-sense stationary (WSS) process to its frequency-domain properties?
#
#
# * **Answer:** It states that the power spectral density (PSD) and the autocorrelation sequence of a WSS process form a Discrete-Time Fourier Transform (DTFT) pair. It translates the measurement of time-lag similarity (autocorrelation) into an equivalent distribution of average power organized by frequency.
#
#
#
# ---
#
# ### Lecture 18: Non-Parametric Spectral Estimation
#
# **Question 4: The Welch Trade-off** (Directly from your slide wrap-up)
#
# * **Question:** If a Welch PSD estimate looks too noisy, name one specific parameter change that reduces variance and state the mathematical or practical price you pay.
#
#
# * **Answer:** You can decrease the segment length. Doing this yields more segments to average together, which reduces the estimator's variance. However, the price paid is decreased frequency resolution; the spectral peaks become broader, making it harder to separate nearby frequency components.
#
#
#
# **Question 5: Resolving Tones** (Adapted from your concept check)
#
# * **Question:** You need to extract bandpower features to classify a signal using a Support Vector Machine, and you must distinguish two close tones at 120 Hz and 126 Hz (sampled at 1 kHz). Should you increase zero-padding, or increase the Welch segment length? Explain why.
#
#
# * **Answer:** You must increase the Welch segment length. Longer segments narrow the spectral main lobe, yielding the true frequency resolution needed to separate the close tones. Increasing zero-padding only interpolates the existing spectral estimate on a finer grid; it does not create new information or improve actual resolution.
#
#
#
# **Question 6: Overlap and Statistical Independence**
#
# * **Question:** In Welch's method, increasing the segment overlap from 0% to 50% gives us more segments to average. Why doesn't the estimator variance decrease by exactly $1/K$ (where $K$ is the new number of segments)?
#
#
# * **Answer:** When segments overlap, they reuse the same data samples. Therefore, the resulting periodograms are correlated with one another, rather than being statistically independent, reducing the efficiency of the averaging.
#
#
#
# ---
#
# Would you like me to draft a SciPy or PyTorch coding assignment to pair with these specific spectral estimation concepts?

# %% [markdown]
# Yes, please
#
# Here is a hands-on coding assignment utilizing SciPy and PyTorch that reinforces the transition from classical spectral estimation to machine learning feature pipelines. It bridges the theoretical rigor of Oppenheim and Schafer with the practical ML workflows seen in Raschka's texts.
#
# ## Spectral Estimation and ML Feature Pipelines
#
# **Objective:** Implement non-parametric spectral estimators to analyze wide-sense stationary (WSS) processes and extract stable bandpower features for a downstream classification task.
#
# ### Part 1: The Raw Periodogram & Filtering
#
# 1. Generate a 10,000-sample realization of zero-mean white Gaussian noise.
# 2. Design a low-pass FIR filter using `scipy.signal.firwin` with a normalized cutoff frequency of $\omega_c = 0.2\pi$. Pass the white noise through this filter.
# 3. Compute the raw periodogram of the filtered output.
# 4. **Deliverable:** Plot the periodogram against the theoretical scaled squared-magnitude response, $\sigma^2\vert{}H(e^{j\omega})\vert{}^2$. In a short paragraph, explain why the empirical estimate exhibits large random fluctuations despite the large sample size $N$.
#
# ### Part 2: Variance Reduction via Welch's Method
#
# Utilize `scipy.signal.welch` (or `torchaudio.transforms.Spectrogram`) on the filtered noise from Part 1 to observe the effects of averaging.
#
# * Compute the PSD using a Hann window with 50% overlap.
# * Sweep the segment length: $L \in \{128, 512, 2048\}$.
# * **Deliverable:** Plot the three PSD estimates on a single logarithmic scale. Discuss the quantitative trade-off observed between frequency resolution (main-lobe width) and estimator variance. Which segment length best captures the true filter shape?
#
# ### Part 3: Bandpower Features for Machine Learning
#
# Generate a dataset of 200 noisy synthetic signals sampled at 1 kHz (100 from Class 0, 100 from Class 1). Class 0 has dominant spectral energy between 60–130 Hz, while Class 1 has dominant energy between 180–270 Hz.
#
# * Compute the Welch PSD for every record.
# * Integrate the PSD over the two specific frequency bands to create a 2D feature vector for each signal.
# * Standardize these features (fitting the scaler on training data only to avoid leakage).
# * Train a linear classifier using PyTorch to separate the two classes based on these stable features.
# * **Deliverable:** Provide a scatter plot of the 2D feature space colored by class label, and report the classification accuracy.
#
# ---
#
# Would you like me to draft a grading rubric for this assignment, or perhaps an automated test script to help you check their feature extraction logic?
#
# Give me a solution in Python
#
# Here is the complete Python solution implementing the three parts of the assignment using NumPy, SciPy, Scikit-Learn, and PyTorch.
#
# ### Python Solution
#
# ```python
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # ==========================================
# # PART 1: The Raw Periodogram & Filtering
# # ==========================================
# np.random.seed(42)
# torch.manual_seed(42)
#
# # 1. Generate 10,000 samples of white Gaussian noise
# N = 10000
# fs = 1000  # Assumed sampling rate in Hz
# x = np.random.randn(N)
#
# # 2. Design low-pass FIR filter (cutoff = 0.2*pi, which is 0.2 of Nyquist)
# numtaps = 101
# b = signal.firwin(numtaps, cutoff=0.2)
# y = signal.lfilter(b, 1, x)
#
# # 3. Compute raw periodogram
# f_per, Pxx_per = signal.periodogram(y, fs=fs, window='boxcar', scaling='density')
#
# # Theoretical frequency response scaled by input noise variance (var=1)
# w, h = signal.freqz(b, worN=8000, fs=fs)
# H_squared = np.abs(h)**2 * (2 / fs)  # Scaled for one-sided density
#
# plt.figure(figsize=(10, 4))
# plt.semilogy(f_per, Pxx_per, color='lightblue', label='Raw Periodogram')
# plt.semilogy(w, H_squared, color='red', linewidth=2, label=r'Theoretical $|H(e^{j\omega})|^2$')
# plt.title('Part 1: Raw Periodogram vs. Theoretical PSD')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V**2/Hz)')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# """
# Part 1 Explanation:
# The empirical estimate exhibits large random fluctuations because the raw periodogram 
# is an inconsistent estimator. For a random noise process, increasing N increases the 
# frequency grid density (finer bins) but does not decrease the variance of the squared 
# magnitude at any given frequency bin.
# """
#
# # ==========================================
# # PART 2: Variance Reduction via Welch's Method
# # ==========================================
# segment_lengths = [128, 512, 2048]
#
# plt.figure(figsize=(10, 4))
# for L in segment_lengths:
#     f_welch, Pxx_welch = signal.welch(y, fs=fs, window='hann', nperseg=L, noverlap=L//2)
#     plt.semilogy(f_welch, Pxx_welch, label=f'Welch (L={L})')
#
# plt.semilogy(w, H_squared, 'k--', linewidth=2, label='Theoretical')
# plt.title("Part 2: Welch's Method with Varying Segment Lengths")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (V**2/Hz)')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# """
# Part 2 Explanation:
# As segment length (L) increases, the frequency resolution improves (narrower main-lobe), 
# allowing the estimate to track the sharp drop-off of the filter more accurately. 
# However, larger L means fewer segments are averaged, resulting in higher estimator variance 
# (more jagged lines). L=512 offers a strong balance of smoothness and resolution here.
# """
#
# # ==========================================
# # PART 3: Bandpower Features & PyTorch Classifier
# # ==========================================
# n_samples = 200
# record_length = 1000
# t = np.arange(record_length) / fs
#
# # Generate synthetic dataset
# X_signals = []
# y_labels = []
#
# for i in range(n_samples):
#     noise = np.random.randn(record_length) * 0.5
#     label = 0 if i < 100 else 1
#     
#     # Class 0: Dominant energy 60-130 Hz
#     # Class 1: Dominant energy 180-270 Hz
#     f_center = np.random.uniform(80, 110) if label == 0 else np.random.uniform(200, 250)
#     sig = noise + np.sin(2 * np.pi * f_center * t)
#     
#     X_signals.append(sig)
#     y_labels.append(label)
#
# X_signals = np.array(X_signals)
# y_labels = np.array(y_labels)
#
# # Feature Extraction: Compute bandpower using Welch's method
# features = []
# band_0_idx = None
# band_1_idx = None
#
# for sig in X_signals:
#     f, Pxx = signal.welch(sig, fs=fs, window='hann', nperseg=256, noverlap=128)
#     
#     # Identify indices for integration bands once
#     if band_0_idx is None:
#         band_0_idx = np.logical_and(f >= 60, f <= 130)
#         band_1_idx = np.logical_and(f >= 180, f <= 270)
#         
#     bp_0 = np.trapz(Pxx[band_0_idx], f[band_0_idx])
#     bp_1 = np.trapz(Pxx[band_1_idx], f[band_1_idx])
#     features.append([bp_0, bp_1])
#
# features = np.array(features)
#
# # Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.3, random_state=42)
#
# # Standardize features (fit ONLY on training data to prevent leakage)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Convert to PyTorch Tensors
# X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
# y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
#
# # Define a simple linear classifier (Logistic Regression in PyTorch)
# model = nn.Linear(2, 1)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # Training Loop
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_t)
#     loss = criterion(outputs, y_train_t)
#     loss.backward()
#     optimizer.step()
#
# # Evaluation
# model.eval()
# with torch.no_grad():
#     predictions = torch.sigmoid(model(X_test_t))
#     predicted_classes = (predictions >= 0.5).float()
#     accuracy = (predicted_classes == y_test_t).sum().item() / y_test_t.shape[0]
#
# print(f"Test Set Accuracy: {accuracy * 100:.1f}%")
#
# # Plot feature space
# plt.figure(figsize=(6, 5))
# plt.scatter(X_train_scaled[y_train==0, 0], X_train_scaled[y_train==0, 1], alpha=0.7, label='Class 0 (60-130 Hz)')
# plt.scatter(X_train_scaled[y_train==1, 0], X_train_scaled[y_train==1, 1], alpha=0.7, label='Class 1 (180-270 Hz)')
# plt.title('Standardized Bandpower Features')
# plt.xlabel('Band 0 Power (Standardized)')
# plt.ylabel('Band 1 Power (Standardized)')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# ```
#
# -------------------
#

# %% [markdown]
# ## Week 9, Lectures 17 and 18, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# ChatGPT:
#
# For these two lectures, I would use exit tickets that test whether students can distinguish **theoretical PSD from a finite-data estimate**, and then distinguish **resolution improvements from variance reduction**. That is the conceptual thread connecting Lecture 17 and Lecture 18. Lecture 17 emphasizes PSD, the periodogram, leakage, zero-padding, and the inconsistency of the raw periodogram; Lecture 18 develops Bartlett and Welch as ways to trade some frequency detail for lower estimator variance.  
#
# ## Lecture 17: Power Spectral Density and the Periodogram
#
# ### Exit Ticket 1 — Best overall conceptual question
#
# **Question:**
# A white-noise process has theoretical PSD
#
# $$
# S_{xx}(e^{j\omega})=\sigma^2.
# $$
#
# You compute the periodogram of one long realization and it looks very jagged rather than flat. Is something wrong? Explain.
#
# **Good answer:**
# No. The theoretical PSD is an ensemble property of the random process, while a periodogram is a random estimate computed from one finite realization. Even for a long record, individual periodogram values have large random fluctuations, so a single periodogram need not look flat. 
#
# **What you are testing:**
# Whether students understand the critical distinction
#
# $$
# \text{true PSD}\neq\text{one realization's periodogram}.
# $$
#
# This is probably my first choice for Lecture 17.
#
# ---
#
# ### Exit Ticket 2 — Longer record misconception
#
# **Question:**
# Why does increasing the record length $N$ not automatically make a raw periodogram a low-variance PSD estimator?
#
# **Good answer:**
# Increasing $N$ provides finer frequency spacing,
#
# $$
# \Delta f=\frac{F_s}{N},
# $$
#
# and improves frequency detail, but the variance of the raw periodogram at a fixed frequency does not generally approach zero as $N$ increases. Therefore the raw periodogram remains statistically noisy.  
#
# **Excellent answer:**
# For white Gaussian noise, the periodogram has approximately
#
# $$
# E\{I\}\approx\sigma^2
# $$
#
# and
#
# $$
# \operatorname{Var}\{I\}\approx\sigma^4,
# $$
#
# so its relative scatter remains of order one even for large $N$.
#
# **What you are testing:**
# Whether students distinguish **frequency resolution** from **estimator variance**.
#
# ---
#
# ### Exit Ticket 3 — Zero-padding
#
# **Question:**
# A student changes a 512-point FFT to a 4096-point FFT by adding zeros to the same 512 measured samples. What improves, and what does not?
#
# **Good answer:**
# The plotted frequency grid becomes denser and the spectrum appears smoother because zero-padding interpolates the finite-record spectrum. It does not add new measured information and therefore does not fundamentally improve true frequency resolution or reduce periodogram variance. 
#
# **What you are testing:**
# One of the most common DSP misconceptions:
#
# $$
# \text{more FFT points}\neq\text{more information}.
# $$
#
# ---
#
# ### Exit Ticket 4 — LTI filtering of a PSD
#
# **Question:**
# A WSS input passes through an LTI filter with frequency response $H(e^{j\omega})$. Write the relationship between input and output PSDs and explain physically what it means.
#
# **Good answer:**
#
# $$
# S_{yy}(e^{j\omega}) =
# S_{xx}(e^{j\omega})
# |H(e^{j\omega})|^2.
# $$
#
# The filter scales the input power at each frequency by the squared magnitude response. Frequencies in the passband retain more power, while frequencies strongly attenuated by the filter have lower output PSD. 
#
# **Follow-up:**
# If $H(e^{j\omega_0})=0$, what happens?
#
# **Answer:**
#
# $$
# S_{yy}(e^{j\omega_0})=0,
# $$
#
# assuming the model applies.
#
# ---
#
# ### Exit Ticket 5 — Leakage
#
# **Question:**
# Why can a sinusoid at one frequency produce nonzero periodogram values across several neighboring frequencies?
#
# **Good answer:**
# Because observing only a finite record is equivalent to multiplying the signal by a finite-duration window. Multiplication in time corresponds to convolution in frequency, so the sinusoid's ideal spectral line is spread according to the window spectrum. This is spectral leakage. 
#
# **Strong follow-up:**
# What happens if we replace the rectangular window with a Hann window?
#
# **Answer:**
# The Hann window lowers sidelobes and therefore reduces distant leakage, but its wider main lobe reduces the ability to distinguish closely spaced frequency components.
#
# ---
#
# ### Exit Ticket 6 — Same process, different periodograms
#
# **Question:**
# Two long data records are generated independently from exactly the same WSS process. Their periodograms look noticeably different. Does this demonstrate that the underlying PSD changed?
#
# **Good answer:**
# No. The periodogram itself is random and has high variance. Different realizations of the same process can therefore produce noticeably different periodograms. More averaging or statistical confidence analysis would be needed before concluding that the process PSD changed. 
#
# **Why I like this question:**
# It reveals whether students are treating the periodogram as a deterministic spectrum rather than as an estimator.
#
# ---
#
# ## Lecture 18: Bartlett, Welch, and Non-Parametric Spectral Estimation
#
# ### Exit Ticket 1 — Best overall question
#
# **Question:**
# Your Welch PSD estimate is too noisy. Name one change you could make to reduce its variance, and state the price you pay.
#
# **Good answer:**
# Use shorter segments. This provides more segments and therefore more periodograms to average, reducing estimator variance. The cost is poorer frequency resolution because each segment has a shorter observation time and a wider spectral main lobe. 
#
# This is essentially the exit ticket already built into the lecture, and I think it is an excellent one. 
#
# ---
#
# ### Exit Ticket 2 — Segment-length prediction
#
# **Question:**
# Suppose you double the Welch segment length while keeping the total data record fixed. Predict what happens to:
#
# 1. frequency resolution,
# 2. number of averages,
# 3. variance of the PSD estimate.
#
# **Good answer:**
# Frequency resolution improves because the segments are longer. However, fewer segments fit into the same total data record, so fewer periodograms are averaged. Therefore estimator variance generally increases. 
#
# A compact student answer could be:
#
# $$
# L\uparrow
# \Rightarrow
# \text{resolution improves}
# \Rightarrow
# K\downarrow
# \Rightarrow
# \text{variance tends to increase}.
# $$
#
# ---
#
# ### Exit Ticket 3 — Bartlett calculation
#
# **Question:**
# You have
#
# $$
# N=4096
# $$
#
# samples at
#
# $$
# F_s=1000\text{ Hz}.
# $$
#
# Bartlett's method uses $K=8$ equal non-overlapping segments.
#
# Find the segment length and nominal frequency spacing.
#
# **Answer:**
#
# $$
# L=\frac{N}{K} =
# \frac{4096}{8} =
# 512.
# $$
#
# Then
#
# $$
# \Delta f =
# \frac{F_s}{L} =
# \frac{1000}{512}
# \approx1.95\text{ Hz}.
# $$
#
# The full-record periodogram would instead have spacing
#
# $$
# \frac{1000}{4096}
# \approx0.244\text{ Hz}.
# $$
#
# Thus Bartlett sacrifices frequency detail to obtain multiple estimates for averaging. 
#
# **What you are testing:**
# Whether students can connect the numerical calculation to the conceptual trade-off.
#
# ---
#
# ### Exit Ticket 4 — Bartlett versus Welch
#
# **Question:**
# What are the two main additions Welch makes to Bartlett's method, and why are they useful?
#
# **Good answer:**
# Welch uses overlapping segments and applies a tapering window to each segment. The window reduces spectral leakage, while overlap allows more modified periodograms to be formed and averaged from the same finite record. 
#
# **Excellent answer:**
# Overlap does not create independent new data, so the variance reduction is less than what would be predicted by simply counting the number of overlapping segments.
#
# ---
#
# ### Exit Ticket 5 — Why overlap is not free information
#
# **Question:**
# Why does increasing Welch overlap from 50% to 75% not reduce variance exactly according to $1/K$, even though it produces more segments?
#
# **Good answer:**
# Because overlapping segments share samples. Their periodograms are therefore correlated rather than independent. Increasing the number of overlapping segments increases the number of estimates being averaged, but not the amount of independent information by the same factor. 
#
# **What you are testing:**
# Whether students understand the statistical reason for overlap rather than viewing it as free data multiplication.
#
# ---
#
# ### Exit Ticket 6 — Close tones
#
# **Question:**
# A signal sampled at
#
# $$
# F_s=1000\text{ Hz}
# $$
#
# contains noisy tones at 120 Hz and 126 Hz. Which change is most likely to help distinguish them?
#
# A. Shorter Welch segments
# B. Longer Welch segments
# C. More zero-padding only
# D. More frequency smoothing
#
# **Answer:**
# **B. Longer Welch segments.**
#
# **Explanation:**
# The tones are only 6 Hz apart. Longer segments give longer observation times and narrower spectral main lobes, improving true frequency resolution. Zero-padding merely samples the same spectral estimate more densely, while stronger smoothing or shorter segments would tend to blur the peaks together. 
#
# This is probably my favorite quick multiple-choice exit ticket for Lecture 18.
#
# ---
#
# ### Exit Ticket 7 — Window trade-off
#
# **Question:**
# Compared with a rectangular window, what is the main advantage and disadvantage of a Hann window in Welch PSD estimation?
#
# **Good answer:**
# A Hann window has lower sidelobes, so it reduces spectral leakage from strong frequency components into distant frequencies. The cost is a wider main lobe, which reduces the ability to resolve closely spaced spectral components. 
#
# ---
#
# ### Exit Ticket 8 — ML connection
#
# **Question:**
# Why might Welch band powers be better features for an SVM than the raw waveform samples?
#
# **Good answer:**
# Welch averaging reduces some realization-to-realization spectral variability, and band powers compress many PSD bins into a smaller set of interpretable features. This can provide a more stable feature vector when class differences are primarily spectral. 
#
# **Important qualification:**
# PSD features discard phase and precise time localization, so they may be poor features if the classification problem depends on when an event occurs or on temporal ordering.
#
# ---
#
# ### Exit Ticket 9 — Signal-processing leakage versus ML leakage
#
# **Question:**
# The lecture uses the word "leakage" in two very different ways. Explain both.
#
# **Good answer:**
#
# **Spectral leakage:** finite observation and windowing spread energy from one frequency into neighboring frequencies.
#
# **Dataset leakage:** information from validation or test data improperly influences training or preprocessing, such as computing normalization statistics using the full dataset rather than the training set only. 
#
# This is particularly useful for this DSP/ML course because it reinforces terminology from both sides of the course.
#
# ---
#
# ## My recommended actual exit tickets
#
# If you want only **one question at the end of each lecture**, I would use these:
#
# **Lecture 17**
#
# > You double the amount of measured data and also compute a much larger FFT. Explain separately what happens to frequency resolution, the plotted frequency grid, and raw-periodogram variance.
#
# A strong answer should distinguish:
#
# $$
# \text{longer measured record}
# \rightarrow
# \text{better true frequency detail}
# $$
#
# $$
# \text{zero-padding}
# \rightarrow
# \text{denser displayed frequency grid only}
# $$
#
# $$
# \text{raw periodogram}
# \rightarrow
# \text{still high variance}.
# $$
#
# That single question tests several of the lecture's most important ideas simultaneously. 
#
# **Lecture 18**
#
# > You can choose either longer or shorter Welch segments. Explain which you would choose if your priority is (a) distinguishing two closely spaced tones and (b) obtaining a stable estimate of broad-band power.
#
# A strong answer is:
#
# > For closely spaced tones, use longer segments because they give better frequency resolution. For stable broad-band power estimates, shorter segments may be preferable because they provide more segments to average and therefore lower estimator variance. The correct choice depends on the spectral information the application needs.
#
# That gets directly at the lecture's central **resolution-versus-variance design trade-off**. 
#
# For senior/first-year graduate ECE students, I would favor these **explain-and-predict** questions over definition questions. They reveal whether students can reason about an estimator configuration they have not seen before, rather than merely remembering that “Welch is smoother than the periodogram.”
#
# -----------------------
#
