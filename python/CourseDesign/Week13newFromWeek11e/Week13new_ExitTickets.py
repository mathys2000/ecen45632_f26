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
# # Week13new ExitTickets
#
# Week13new from Week11e, exit ticket questions based on lecture 25 and 26 slides.
#

# %% [markdown]
# Week13(new)Lecture25.pptx
# Week13(new)Lecture26.pptx
#
# Give me some good exit ticket questions for these two lectures
#
# ChatGPT (Medium):
#
# # **Week 13: Exit Ticket Questions with Answers**
#
# **ECEN 4/5632 — Theory and Application of Digital Filters**
#
# Based on your two uploaded lectures:
#
# - **Lecture 25:** Feature Engineering — Mel Spectrograms and MFCCs (November 10, 2026)
# - **Lecture 26:** Machine Learning Foundations — Supervised Learning, Data Splits, and Scikit-Learn (November 12, 2026)
#
# For senior and first-year graduate ECE students, I recommend exit tickets that require students to **explain a design choice, predict a consequence, or diagnose a mistake**, rather than simply recall definitions.
#
# Below are six candidate questions for each lecture, with expected answers and teaching notes. Most should take 2–4 minutes each. I would typically assign two questions per lecture.
#
# ## **Lecture 25 — Feature Engineering: Mel Spectrograms and MFCCs**
#
# ### **Question 1: What information does the Mel filterbank discard?**
#
# **Question:**
#
# A student argues:
#
# > "A Mel spectrogram contains essentially the same information as the STFT, except that the frequency axis is expressed in Mel instead of Hz."
#
# Is this statement correct? Explain what happens mathematically when the STFT power spectrum is converted to a Mel spectrogram.
#
# **Expected answer:**
#
# The statement is incorrect. Mel conversion is not simply a change in the frequency-axis labels.
#
# The Mel filterbank computes
#
# \\[ S\_{\mathrm{mel}}[b,m]=\sum_k H_b[k]|X[m,k]|^2 \\]
#
# where each Mel band combines information from multiple STFT frequency bins.
#
# Consequently:
#
# - The frequency representation has fewer dimensions.
# - Detailed spectral structure within each Mel band is lost.
# - Different STFT power spectra can produce the same Mel-band energies.
# - The time-frame structure is preserved.
#
# In matrix form,
#
# \\[ \mathbf S\_{\mathrm{mel}}=\mathbf H\mathbf P \\]
#
# The filterbank transformation is generally not invertible because the number of Mel bands is smaller than the number of STFT bins.
#
# **What makes a good answer:** Students should distinguish *frequency warping* from *frequency-domain pooling and dimensionality reduction*.
#
# **Teaching value:** Tests whether students understand what information is actually lost, rather than simply memorizing that Mel filters are perceptually motivated.
#
# ### **Question 2: Why do we use logarithms before calculating MFCCs?**
#
# **Question:**
#
# Two Mel frequency bands have powers of 100 and 1, respectively. After the signal amplitude doubles, what happens to their powers and their power levels in dB? Why is logarithmic compression useful for machine learning?
#
# **Expected answer:**
#
# Doubling amplitude multiplies power by four:
#
# \\[ P'\_1=400,\qquad P'\_2=4 \\]
#
# The dB change for either band is
#
# \\[ 10\log\_{10}(4)\approx 6.02\ \mathrm{dB} \\]
#
# Thus both bands increase by approximately 6 dB, despite having very different absolute power values.
#
# Logarithmic compression converts multiplicative changes in power into additive changes, reduces the dynamic range of the features, and can make relatively weak spectral components easier to represent.
#
# **What makes a good answer:** Recognizes the difference between amplitude and power scaling and explains why the logarithm helps beyond just saying "human hearing is logarithmic."
#
# **Teaching value:** Reinforces the connection between DSP power calculations, decibels, and feature engineering.
#
# ### **Question 3: Why use a DCT to obtain MFCCs?**
#
# **Question:**
#
# Why do we apply a Discrete Cosine Transform (DCT) to the log-Mel spectrum? Why is it often reasonable to retain only the first 20 MFCCs rather than all 64 coefficients from a 64-band Mel spectrogram?
#
# **Expected answer:**
#
# The DCT expresses the log-Mel spectrum as a sum of cosine basis functions across Mel-band index.
#
# Because neighboring Mel bands are correlated and the spectral envelope is often relatively smooth, much of the useful envelope information can be concentrated in low-order DCT coefficients.
#
# - Low-order coefficients represent broad spectral-envelope structure.
# - Higher-order coefficients describe finer variation across Mel bands.
# - Retaining only the first 20 coefficients reduces dimensionality and may suppress irrelevant fine structure.
#
# However, truncating coefficients is lossy and may remove useful information for some classification tasks.
#
# The DCT provides approximate decorrelation for many audio datasets, not guaranteed statistical independence.
#
# **What makes a good answer:** Explains both the mathematical role of the DCT and the tradeoff associated with coefficient truncation.
#
# ### **Question 4: Predict the dimensions of an audio feature vector**
#
# **Question:**
#
# We process a 4-second audio clip sampled at 22,050 Hz using
#
# ```
# n_fft = 2048
# hop_length = 512
# n_mels = 64n_mfcc = 20
# ```
#
# Assuming the default centered STFT in librosa, determine:
#
# 1. The number of waveform samples.
# 2. The dimensions of the one-sided STFT.
# 3. The dimensions of the Mel spectrogram.
# 4. The dimensions of the MFCC matrix.
# 5. The length of the feature vector containing the mean and standard deviation of each MFCC.
#
# **Expected answer:**
#
# | Representation  | Dimensions     |
# | --------------- | -------------- |
# | Waveform        | 88,200 samples |
# | STFT            | 1025 × 173     |
# | Mel spectrogram | 64 × 173       |
# | MFCC matrix     | 20 × 173       |
# | MFCC mean + std | 40             |
#
# The number of STFT time frames is
#
# \\[ T=1+\left\lfloor\frac{88200}{512}\right\rfloor=173 \\]
#
# under the default centering convention.
#
# **What makes a good answer:** Students can explain where each dimension comes from, especially why a 2048-point real-input FFT produces 1025 nonnegative-frequency bins.
#
# **Teaching value:** Excellent practical debugging question before students use the resulting features in scikit-learn.
#
# ### **Question 5: Can MFCC averages distinguish two different sirens?**
#
# **Question:**
#
# Consider two recordings:
#
# - Recording A: A siren sweeps from 500 Hz to 1500 Hz.
# - Recording B: A siren sweeps from 1500 Hz to 500 Hz.
#
# Both recordings spend equal amounts of time at corresponding frequencies.
#
# Could the two recordings have nearly identical mean MFCC feature vectors? Would adding MFCC standard deviations solve the problem?
#
# **Expected answer:**
#
# Yes. The mean MFCC vector averages each coefficient over time, so it does not preserve the sequence in which spectral patterns occurred.
#
# If the two recordings contain the same collection of spectral patterns in reverse order, they can have identical MFCC means.
#
# Adding the standard deviation does not solve the problem because standard deviation is also insensitive to temporal ordering.
#
# To discriminate the recordings, we need features that preserve temporal evolution, such as the full MFCC matrix or Mel spectrogram, or suitably designed temporal-difference features.
#
# **What makes a good answer:** Students recognize that mean and standard deviation summarize a distribution but do not describe its ordering.
#
# **Teaching value:** One of my preferred questions for this lecture. It leads directly to Lecture 26 and motivates next week's CNN discussion.
#
# ### **Question 6: When is converting stereo to mono a bad idea?**
#
# **Question:**
#
# Your audio classifier must determine whether a vehicle passes a microphone array from left to right or right to left.
#
# Would you convert the stereo recording to mono before extracting Mel spectrograms or MFCCs? Explain.
#
# **Expected answer:**
#
# No, not without checking the effect on the relevant information.
#
# For a simple downmix,
#
# \\[ y\_{\mathrm{mono}}[n]=\frac{y_L[n]+y_R[n]}{2} \\]
#
# the two channels are combined into one signal. This can destroy or obscure interchannel timing and level differences that indicate the direction of movement.
#
# For a directional classification task, we should preserve stereo information or calculate spatial features such as interchannel time differences.
#
# For a task such as detecting whether a siren is present, mono conversion may be perfectly acceptable.
#
# **What makes a good answer:** Recognizes that preprocessing should be determined by the classification task, not by a universal recipe.
#
# ## **Lecture 26 — Machine Learning Foundations and Scikit-Learn**
#
# ### **Question 1: Why can 99% classification accuracy be misleading?**
#
# **Question:**
#
# A student develops an audio classifier using a recording of 100 siren events. Each recording is divided into 20 short clips.
#
# The student randomly splits the resulting 2000 clips into training and test sets and obtains 99% test accuracy.
#
# Why might this result be misleading, even if the classifier was never explicitly trained on the test clips?
#
# **Expected answer:**
#
# Different clips from the same original recording are highly correlated.
#
# If clips from the same source recording appear in training and testing, the classifier might learn recording-specific characteristics such as microphone response, background noise, acoustic environment, or source identity.
#
# It may therefore recognize the recording rather than learn to identify sirens generally.
#
# The correct approach is to split the dataset by original source recording, keeping all segments derived from a source in the same partition.
#
# **What makes a good answer:** Explicitly identifies *source-level data leakage* and explains why it produces an overly optimistic estimate of generalization.
#
# **Teaching value:** Perhaps the single most important exit ticket for Lecture 26.
#
# ### **Question 2: What is wrong with standardizing the entire dataset first?**
#
# **Question:**
#
# Consider the following Python code:
#
# ```
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# ```
#
# Identify the problem and explain how to correct it.
#
# **Expected answer:**
#
# The scaler calculates the mean and standard deviation from the entire dataset before splitting it.
#
# Therefore, information about the test-data distribution influences the preprocessing of the training data, producing data leakage.
#
# The correct procedure is to split first:
#
# ```
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# ```
#
# The scaler must be fitted only on training data.
#
# For the UrbanSound8K application, students should also use the predefined source-aware folds rather than relying on an ordinary random split.
#
# A scikit-learn `Pipeline` is another good solution.
#
# **What makes a good answer:** Distinguishes `fit_transform()` from `transform()` and identifies which data may be used to estimate preprocessing parameters.
#
# ### **Question 3: What exactly does scikit-learn learn?**
#
# **Question:**
#
# We extract 20 MFCCs from every recording, compute each coefficient's mean and standard deviation, and use the resulting vectors to train a Random Forest.
#
# What are the input features, what are the target labels, and what does the Random Forest actually learn?
#
# **Expected answer:**
#
# Each recording produces a 40-dimensional feature vector:
#
# \\[ \mathbf x_i= [\mu_1,\ldots,\mu\_{20},\sigma_1,\ldots,\sigma\_{20}] \\]
#
# The complete feature matrix is
#
# \\[ \mathbf X\in\mathbb R^{N\times40} \\]
#
# and the target vector contains the class label for each recording.
#
# The Random Forest learns a collection of decision trees whose feature thresholds and branching rules help separate the training classes.
#
# It does **not** learn the MFCC extraction process. Those features are calculated beforehand using fixed DSP operations.
#
# It also does not directly analyze the original waveform unless waveform-derived features are supplied.
#
# **What makes a good answer:** Clearly separates hand-engineered feature extraction from learned classifier parameters.
#
# ### **Question 4: Why do we need three datasets?**
#
# **Question:**
#
# A student proposes using the training set to fit a classifier and the test set to compare several hyperparameter choices.
#
# The best-performing classifier on the test set will be reported as the final result.
#
# What is wrong with this procedure?
#
# **Expected answer:**
#
# Repeatedly evaluating models on the test set and selecting the best one makes the test data part of model development.
#
# Even though the classifier parameters were not directly fitted to those samples, the model-selection decisions were influenced by the test performance.
#
# The proper roles are:
#
# - **Training:** Fit model parameters and data-dependent preprocessing.
# - **Validation:** Select hyperparameters and compare feature representations.
# - **Test:** Estimate generalization performance after all model choices have been finalized.
#
# The test set should not be used repeatedly as another validation set.
#
# **What makes a good answer:** Understands that leakage can happen through model selection, not only through direct training.
#
# ### **Question 5: Why is 90% accuracy not necessarily good?**
#
# **Question:**
#
# A classifier is evaluated on a test set containing 100 recordings:
#
# - 90 recordings are labeled "background noise."
# - 10 recordings are labeled "siren."
#
# The classifier predicts "background noise" for every recording.
#
# What is its accuracy? Is this a useful siren detector? Which additional metric should you examine?
#
# **Expected answer:**
#
# The classifier correctly labels 90 out of 100 recordings:
#
# \\[ \mathrm{Accuracy}=\frac{90}{100}=90\\% \\]
#
# However, it never detects a siren.
#
# Its siren recall is
#
# \\[ \mathrm{Recall}\_{\mathrm{siren}}= \frac{TP}{TP+FN}=\frac{0}{0+10}=0 \\]
#
# The classifier is therefore useless for detecting sirens, despite its apparently high accuracy.
#
# A confusion matrix and class-specific recall immediately reveal this problem. Macro-F1 is also useful because it gives equal importance to each class.
#
# **What makes a good answer:** Recognizes why class imbalance can hide poor performance and identifies recall as especially important when missing a siren is costly.
#
# ### **Question 6: Why might adding more trees not solve the problem?**
#
# **Question:**
#
# A Random Forest trained on mean and standard deviation MFCC features performs poorly when distinguishing rising and falling sirens.
#
# A student proposes increasing the number of trees from 100 to 1000.
#
# Why might this fail, and what alternative would you suggest?
#
# **Expected answer:**
#
# The main limitation may not be the classifier capacity but the feature representation.
#
# Mean and standard deviation MFCCs discard temporal ordering. Consequently, a rising and falling frequency trajectory can have very similar feature vectors.
#
# Increasing the number of trees cannot reliably recover information that has already been discarded.
#
# Better approaches include using temporal-difference features, retaining the MFCC time sequence, or using a CNN with a Mel-spectrogram as input.
#
# **What makes a good answer:** Distinguishes the limitations of a machine-learning algorithm from the limitations of its input representation.
#
# **Teaching value:** Excellent bridge to CNNs because it forces students to think about why a more sophisticated model might be justified.
#
# ## **My recommended exit tickets**
#
# For a typical 5-minute exit ticket at the end of each 75-minute lecture, I would select:
#
# Lecture 25 — Feature Engineering
#
# Q5
#
# **Can MFCC averages distinguish rising and falling sirens?**
#
# Tests whether students understand information loss through temporal averaging.
#
# Q1
#
# **What information does the Mel filterbank discard?**
#
# Tests mathematical understanding of frequency pooling and irreversible dimensionality reduction.
#
# Lecture 26 — Machine Learning Foundations
#
# Q1
#
# **Why can 99% classification accuracy be misleading?**
#
# Tests understanding of source-level leakage and genuine generalization.
#
# Q6
#
# **Why might adding more trees not solve the problem?**
#
# Tests whether students distinguish problems with the model from problems with the feature representation.
#
# ## **Suggested grading rubric**
#
# For these short conceptual exit tickets, a simple 3-point rubric is sufficient.
#
# | Score                | Criteria                                                                                                           |
# | -------------------- | ------------------------------------------------------------------------------------------------------------------ |
# | **3 — Excellent**    | Correct conclusion, technically sound explanation, and clear identification of the underlying DSP or ML principle. |
# | **2 — Satisfactory** | Correct conclusion and a mostly correct explanation, but missing an important detail.                              |
# | **1 — Partial**      | Some relevant understanding, but explanation is incomplete or includes a significant misconception.                |
# | **0 — Incorrect**    | Incorrect conclusion or no meaningful explanation.                                                                 |
#
# I particularly recommend using **Lecture 25, Question 5** and **Lecture 26, Question 6** together. They reinforce one of the most important connections between DSP and machine learning: **the quality of a classifier depends not only on the learning algorithm, but also on what information the signal-processing front end preserves or discards.** This gives students a strong conceptual foundation for your subsequent lectures on CNN-based audio classification.

# %%
