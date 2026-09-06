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
#

# %%
