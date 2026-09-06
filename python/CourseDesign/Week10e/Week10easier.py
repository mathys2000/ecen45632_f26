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
# # Week 10 easier
#
# **Week 10: Time-Frequency Analysis.** The Discrete Fourier Transform (DFT), fast convolution, and the Short-Time Fourier Transform (STFT).
#

# %% [markdown]
# ## Week 10: Time-Frequency Analysis
#
# **Objective:** Handle non-stationary signals where frequency content evolves over time, introducing practical computational methods and multi-dimensional feature representations.
#
# ### Lecture 19: The DFT and Fast Convolution (75 min)
#
# * **The Discrete Fourier Transform (20 min):** Reviewing the DFT as a sampled DTFT: $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j\frac{2\pi}{N}kn}$.
# * **Linear vs. Circular Convolution (20 min):** Mathematical distinction between time-domain linear convolution and DFT-domain circular convolution. Padding requirements to make them equivalent.
# * **Block Processing for Streaming Data (35 min):**
# * Overlap-Add method.
# * Overlap-Save method.
# * Why this matters: Efficiently applying FIR filters to continuous data streams, a critical preprocessing step for real-time inference pipelines.
#
#
#
# ### Lecture 20: The STFT and Spectrograms (75 min)
#
# * **The Need for Time-Frequency Localization (15 min):** Analyzing non-stationary signals (e.g., speech, biomedical signals) where global PSD fails.
# * **The Short-Time Fourier Transform (25 min):**
# * Mathematical formulation: $X(n, \omega) = \sum_{m=-\infty}^{\infty} x[m] w[n-m] e^{-j\omega m}$.
# * The uncertainty principle of signal processing: The fundamental trade-off between time resolution (narrow window) and frequency resolution (wide window).
#
#
# * **Spectrograms (20 min):** Visualizing the squared magnitude of the STFT, $\vert{}X(n, \omega)\vert{}^2$.
# * **ML Bridge: Deep Learning with Audio/Series (15 min):** Converting 1D time-series data into 2D spectrogram images. Discussing how these structured 2D matrices serve as standard inputs for Convolutional Neural Networks (CNNs) in PyTorch, mapping classical DSP directly to modern ML architectures.
#
# --------------------------------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# ChatGPT: (Medium)
#
# Done. I created the complete Week 10 Time-Frequency Analysis package.
#
# ### Lecture slides
#
# * [Lecture 19 — The DFT and Fast Convolution](sandbox:/mnt/data/week10_time_frequency/Week10_Lecture19_DFT_Fast_Convolution.pptx)
# * [Lecture 20 — The STFT and Spectrograms](sandbox:/mnt/data/week10_time_frequency/Week10_Lecture20_STFT_Spectrograms.pptx)
#
# The decks are structured for two 75-minute classes and include worked examples, visualizations, concept checks, instructor prompts, ML connections, and exit-ticket questions.
#
# Lecture 19 develops the progression
#
# $$
# \text{DFT}
# \rightarrow
# \text{circular convolution}
# \rightarrow
# \text{zero-padding}
# \rightarrow
# \text{fast linear convolution}
# \rightarrow
# \text{streaming OLA/OLS}.
# $$
#
# I emphasized the distinction between **linear convolution and circular convolution**, including the condition
#
# $$
# N_{\text{FFT}} \geq L_x + L_h - 1
# $$
#
# and, for overlap-add,
#
# $$
# N_{\text{FFT}} \geq B + L_h - 1.
# $$
#
# The streaming section compares overlap-add and overlap-save explicitly and connects FFT block size to throughput, memory, and inference latency.
#
# Lecture 20 develops the progression
#
# $$
# \text{global spectrum}
# \rightarrow
# \text{local windowing}
# \rightarrow
# \text{STFT}
# \rightarrow
# \text{spectrogram}
# \rightarrow
# \text{CNN input}.
# $$
#
# The deck gives substantial attention to the **time-frequency resolution trade-off**, hop size versus window length, spectrogram interpretation, log-power scaling, normalization, and physically meaningful versus inappropriate image-style augmentation.
#
# ### Python material
#
# * [Jupyter notebook — Week 10 DSP + ML Demos](sandbox:/mnt/data/week10_time_frequency/Week10_Time_Frequency_DSP_ML_Demos.ipynb)
# * [Plain Python version](sandbox:/mnt/data/week10_time_frequency/Week10_Time_Frequency_DSP_ML_Demos.py)
#
# The code includes:
#
# * DFT samples versus the underlying finite-record DTFT,
# * linear versus circular convolution,
# * properly zero-padded FFT convolution,
# * complete overlap-add implementation,
# * complete overlap-save implementation,
# * numerical comparison with direct FIR convolution,
# * global-spectrum versus STFT analysis of a chirp,
# * STFT window-length experiments,
# * spectrogram generation,
# * PyTorch `torch.stft`,
# * conversion to `[batch, channel, frequency, time]`,
# * and a small `Conv2d` network operating directly on spectrogram tensors.
#
# I executed both the script and notebook successfully. The direct, one-shot FFT, overlap-add, and overlap-save implementations agree to approximately machine precision.
#
# ### Homework and weekly quiz
#
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week10_time_frequency/Week10_Time_Frequency_Homework_Quiz_Solutions.docx)
#
# The packet contains eight homework problems with complete solutions and grading guidance. Topics include DFT frequency grids, circular-convolution aliasing, OLA/OLS bookkeeping, STFT parameter selection, spectrogram interpretation, PyTorch tensor design, and an integrated latency-constrained DSP/ML front-end design problem.
#
# It also contains an eight-question weekly quiz with complete solutions and differentiation guidance for the combined first-year/senior cohort.
#
# ### Complete package
#
# * [Download the entire Week 10 package](sandbox:/mnt/data/week10_time_frequency/Week10_Time_Frequency_DSP_ML_Package.zip)
#
# A theme I reinforced throughout the week is that **more FFT points do not automatically mean more physical resolution**. In Lecture 19, zero-padding mainly gives denser frequency samples; in Lecture 20, true time-frequency resolution is fundamentally controlled by the observation/window duration.
#
# ---------------------------
#

# %% [markdown]
#

# %%
