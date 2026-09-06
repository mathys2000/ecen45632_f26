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
# # Week 8 easier
#
# **Week 8: Random Signals.** Autocorrelation, cross-correlation, and the response of LTI systems to wide-sense stationary random inputs.
#

# %% [markdown]
# ## Week 8: Random Signals
#
# **Objective:** Transition from deterministic signal processing to stochastic frameworks, establishing the mathematical foundation for noise modeling and statistical feature extraction.
#
# ### Lecture 15: Characterizing Random Signals (75 min)
#
# * **Introduction to Stochastic Processes (15 min):** Defining random variables in discrete time. The concept of an ensemble versus a single realization.
# * **Stationarity (20 min):** Strict-sense stationarity (SSS) vs. Wide-Sense Stationarity (WSS). Emphasize why WSS is the standard assumption for most DSP and ML time-series applications.
# * **Time-Domain Statistics (25 min):**
# * Expected value and mean: $\mu_x = \mathbb{E}[x[n]]$.
# * Autocorrelation sequence: $r_{xx}[m] = \mathbb{E}[x[n]x[n+m]]$.
# * Cross-correlation: $r_{xy}[m] = \mathbb{E}[x[n]y[n+m]]$.
# * Properties of the autocorrelation matrix (symmetry, positive semi-definiteness).
#
#
# * **ML Bridge (15 min):** Using cross-correlation for template matching and feature alignment before feeding data into predictive models (connecting to Scikit-Learn preprocessing).
#
# ### Lecture 16: LTI Systems with Random Inputs (75 min)
#
# * **Mean and Variance Through an LTI System (20 min):** Deriving the output mean $\mu_y = \mu_x \sum_{n} h[n]$ and the impact of the DC gain.
# * **Input-Output Correlation Dynamics (30 min):**
# * Cross-correlation of input and output: $r_{xy}[m] = r_{xx}[m] * h[-m]$.
# * Output autocorrelation: $r_{yy}[m] = r_{xx}[m] * h[m] * h[-m]$, or in terms of the deterministic autocorrelation of the impulse response, $r_{yy}[m] = r_{xx}[m] * r_{hh}[m]$.
#
#
# * **White Noise & Filtering (25 min):** Defining white noise ($r_{xx}[m] = \sigma^2 \delta[m]$). Analyzing the output of an LTI system driven by white noise (coloring the noise). Application: Generating synthetic noise distributions for data augmentation in PyTorch.
#
# ----------------------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# ChatGPT (Medium):
#
# Done. I created the complete Week 8 Random Signals package.
#
# ### Lecture slides
#
# * [Lecture 15 — Characterizing Random Signals](sandbox:/mnt/data/week8_random_signals/Week8_Lecture15_Characterizing_Random_Signals.pptx)
# * [Lecture 16 — LTI Systems with Random Inputs](sandbox:/mnt/data/week8_random_signals/Week8_Lecture16_LTI_Systems_Random_Inputs.pptx)
#
# Both decks are paced for 75 minutes and include conceptual checks, worked examples, DSP/ML connections, figures, instructor notes, and exit questions.
#
# ### Python material
#
# * [Jupyter notebook — Week 8 Random Signals DSP + ML Demos](sandbox:/mnt/data/week8_random_signals/Week8_Random_Signals_DSP_ML_Demos.ipynb)
# * [Plain Python version](sandbox:/mnt/data/week8_random_signals/Week8_Random_Signals_DSP_ML_Demos.py)
#
# The code covers ensemble realizations, stationarity diagnostics, autocorrelation, correlation matrices and positive semidefiniteness, template matching, Scikit-Learn correlation features, LTI mean/variance propagation, white-noise coloring, PSDs, and PyTorch white/colored-noise augmentation. The notebook was executed successfully end-to-end.
#
# ### Homework and weekly quiz
#
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week8_random_signals/Week8_Random_Signals_Homework_Quiz_Solutions.docx)
#
# The packet includes 8 homework problems, complete worked solutions, grading guidance, an 8-question weekly quiz with solutions, and differentiation notes for the mixed first-year/senior audience.
#
# ### Complete package
#
# * [Download the entire Week 8 package](sandbox:/mnt/data/week8_random_signals/Week8_Random_Signals_DSP_ML_Package.zip)
#
# One technical point I made explicit in Lecture 16: with your Lecture 15 definition
#
# $$
# r_{xy}[m]=E\{x[n]y[n+m]\},
# $$
#
# the consistent real-signal input-output relationship is
#
# $$
# r_{xy}[m]=r_{xx}[m]*h[m].
# $$
#
# The often-seen $h[-m]$ form corresponds to the alternate convention $E{x[n+m]y[n]}$. The slides show both conventions so students do not accidentally mix them.
#
# -------------------
#

# %% [markdown]
#
