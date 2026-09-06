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
# # Week 7 easier
#
# **Week 7: Polyphase Structures.** Efficient implementations of multi-rate filters using polyphase decomposition. 
#

# %% [markdown]
# ## Week 7: Polyphase Structures
#
# **Lecture 13: Polyphase Decomposition and Efficient Decimation**
# *Oppenheim & Schafer: Chapter 4 (4.7)*
#
# * **0:00 – 0:20 | The Polyphase Concept:** Decomposing an impulse response into $M$ sub-sequences.
# * Equation: $H(z) = \sum_{k=0}^{M-1} z^{-k} E_k(z^M)$.
# * Visualizing how a single FIR filter splits into parallel branches.
#
#
# * **0:20 – 0:55 | Efficient Decimation and Noble Identities:**
# * The naive approach: Computing outputs that the downsampler immediately throws away.
# * First Noble Identity: Commuting the downsampler and the filter.
# * Moving the downsampler *before* the polyphase filters $E_k(z)$ so that all arithmetic runs at the lower sampling rate, achieving an $M$-fold computational savings.
#
#
# * **0:55 – 1:15 | ML Context – Grouped Convolutions:** How polyphase parallelization relates conceptually to grouped convolutions in PyTorch and highly efficient edge-deployment architectures (e.g., MobileNet).
#
# **Lecture 14: Efficient Interpolation and Filter Banks**
# *Oppenheim & Schafer: Chapter 4 (4.7, 4.8)*
#
# * **0:00 – 0:35 | Efficient Interpolation:**
# * Second Noble Identity: Commuting the upsampler and the filter.
# * Moving the upsampler *after* the polyphase filters. Computing outputs without multiplying by the inserted zeros, achieving an $L$-fold computational savings.
#
#
# * **0:35 – 0:55 | Two-Channel Filter Banks:** Introduction to Analysis and Synthesis banks. Splitting a signal into lowpass and highpass subbands, downsampling, and perfectly reconstructing them.
# * **0:55 – 1:15 | ML Context – Learned Filter Banks:** Transitioning from fixed polyphase filter banks to learned filter banks in ML. Using fixed subband decompositions (like wavelet scattering transforms or Mel-filter banks) as highly efficient, non-trainable feature extractors that feed into lightweight PyTorch classifiers.
#
# -----------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# ChatGPT: (Medium)
#
# Done. I created the complete Week 7 Polyphase Structures package.
#
# ### Lecture slides
#
# * [Lecture 13 — Polyphase Decomposition and Efficient Decimation](sandbox:/mnt/data/week7_polyphase/Week7_Lecture13_Polyphase_Efficient_Decimation.pptx)
# * [Lecture 14 — Efficient Interpolation and Filter Banks](sandbox:/mnt/data/week7_polyphase/Week7_Lecture14_Efficient_Interpolation_Filter_Banks.pptx)
#
# The decks include pacing for the 75-minute classes, worked examples, Noble-identity diagrams, computational-savings examples, concept checks, instructor notes, ML connections, and exit questions.
#
# A point I treated carefully is that the Noble identities do **not** allow an arbitrary filter to move through a rate changer. Lecture 13 explicitly distinguishes
#
# $$
# G(z^M)\rightarrow\downarrow M
# $$
#
# from an arbitrary $G(z)$, and Lecture 14 makes the analogous distinction for interpolation. This prevents a common conceptual error.
#
# The ML sections also distinguish the useful analogy from an exact equivalence: polyphase decomposition partitions **time/tap phases modulo the rate-change factor**, whereas grouped convolution partitions **feature channels**.
#
# ### Python material
#
# * [Jupyter notebook — Week 7 Polyphase DSP + ML Demos](sandbox:/mnt/data/week7_polyphase/Week7_Polyphase_DSP_ML_Demos.ipynb)
# * [Plain Python version](sandbox:/mnt/data/week7_polyphase/Week7_Polyphase_DSP_ML_Demos.py)
#
# The code includes:
#
# * polyphase decomposition and reconstruction,
# * naive versus polyphase decimation,
# * numerical verification of the First Noble Identity,
# * PyTorch grouped `Conv1d`,
# * numerical verification of the Second Noble Identity,
# * efficient interpolation with `scipy.signal.upfirdn`,
# * a perfectly reconstructing Haar two-channel bank,
# * fixed FIR subband features with Scikit-Learn,
# * and a lightweight PyTorch classifier after a fixed DSP front end.
#
# The notebook was executed successfully end-to-end.
#
# ### Homework and weekly quiz
#
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week7_polyphase/Week7_Polyphase_Homework_Quiz_Solutions.docx)
#
# The packet contains eight homework problems with complete solutions and grading guidance, including:
#
# * polyphase decomposition,
# * operation-count savings,
# * a Noble-identity counterexample,
# * the efficient-decimator derivation,
# * interpolation phases,
# * Haar perfect reconstruction,
# * two-channel perfect-reconstruction conditions,
# * and DSP versus grouped/depthwise CNN architecture design.
#
# It also contains an eight-question weekly quiz with solutions plus differentiation notes for the mixed first-year/senior cohort.
#
# ### Complete package
#
# * [Download the entire Week 7 package](sandbox:/mnt/data/week7_polyphase/Week7_Polyphase_DSP_ML_Package.zip)
#
# ------------------------------
#

# %% [markdown]
#
