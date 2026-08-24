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
# # Exit Ticket Answers
#
# What are the hallmarks of good exit ticket answers?

# %% [markdown]
# Characterize the features of good exit ticket answers
#
# ChatGPT (Medium):
#
# Good exit ticket answers should be **short, diagnostic, and explanatory**. The goal is not to see whether students can reproduce lecture notes; it is to see whether they understand the key idea well enough to express it correctly with minimal prompting.
#
# A strong exit-ticket answer usually has these features:
#
# * **Correct conclusion.** The answer reaches the right result: stable/unstable, aliases to 3 kHz, DTFT exists/does not exist, zero creates a notch, etc.
# * **One essential reason.** Students should give the key reasoning, not just the final word. For example, “Stable because the unit circle lies in the ROC” is much more informative than simply “stable.”
# * **Uses the right concept or vocabulary.** Terms such as ROC, unit circle, pole, zero, aliasing, sinc, quantization, and causality should be used accurately.
# * **Shows the relevant connection.** Good answers connect two ideas: “causal ⇒ exterior ROC,” “zero on unit circle ⇒ numerator becomes zero,” or “adding one bit halves (\Delta) ⇒ about 6 dB SQNR improvement.”
# * **Avoids irrelevant detail.** An exit ticket should not require a full derivation unless derivation itself is the learning objective. Usually one equation plus one sentence is enough.
# * **Makes the reasoning visible.** You want enough evidence to distinguish understanding from guessing.
#
# For your DSP lectures, I would think of answers in roughly three quality levels:
#
# **Weak answer:**
# “Stable.”
#
# **Acceptable answer:**
# “Stable because all the poles are inside the unit circle.”
#
# **Strong answer:**
# “For a causal rational system, the ROC is outside the outermost pole. Since all poles are inside (|z|=1), the ROC includes the unit circle, so the system is BIBO stable.”
#
# The strong answer is better because it exposes the reasoning chain rather than only recalling a shortcut.
#
# Likewise, for a pole-zero question:
#
# **Weak:**
# “A zero causes a notch.”
#
# **Acceptable:**
# “A zero on the unit circle makes the response zero at that frequency.”
#
# **Strong:**
# “When (e^{j\omega}) reaches the angle of a zero on the unit circle, the distance to that zero is zero, so the numerator of (H(e^{j\omega})) becomes zero and a notch occurs.”
#
# For a numerical exit ticket:
#
# **Weak:**
# “3 kHz.”
#
# **Acceptable:**
# “3 kHz because 9 kHz aliases when sampled at 12 kHz.”
#
# **Strong:**
# “The Nyquist frequency is 6 kHz, so 9 kHz folds about 6 kHz to (12-9=3) kHz.”
#
# A useful grading principle is therefore:
#
# $$\boxed{\text{good exit-ticket answer} =
# \text{correct result}
# +
# \text{key justification}}$$
#
# You generally do **not** need complete formal derivations, polished prose, or every intermediate algebraic step. In fact, if students need five minutes to write a paragraph, the question is probably doing more than an exit ticket should.
#
# For your lectures, I would usually define a “good answer” as something that can be written in **one to three sentences or a few lines of math** and that demonstrates the central conceptual relationship you were trying to teach.
#
#

# %%
