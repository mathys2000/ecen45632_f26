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
# # Week 11 easier
#
# **Week 11: Feature Engineering for Audio.** Translating the STFT into Mel-spectrograms and MFCCs. Introduction to Scikit-Learn (Raschka). Loading and preprocessing the Urban Sound Dataset.
#

# %% [markdown]
# Week 11 easier
#
# Can you map out two 75 min lectures for week 11
#
# Gemini Pro:
#
# Here is the detailed breakdown for the two 75-minute lectures in Week 11.
#
# This week marks the official transition from classic DSP to Machine Learning. We shift textbooks (opening Raschka) and change our mindset. We are no longer trying to perfectly reconstruct a signal; instead, we are trying to extract the minimum amount of "perceptually relevant" information needed for an algorithm to make a decision.
#
# ## Lecture 21: Feature Engineering (From Physics to Perception)
#
# **Objective:** Transform the raw, linear STFT spectrogram into human-perception-aligned features (Mel-spectrograms and MFCCs) using `librosa`, and standardize audio tensors for machine learning.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 15 min** | **The Problem with Linear Frequency** | Motivation: The STFT divides frequencies linearly. But human hearing is logarithmic—we can easily distinguish 100 Hz from 200 Hz, but 10,000 Hz and 10,100 Hz sound identical. Feeding a raw STFT to a model wastes massive memory on high-frequency bins that don't help classification. |
# | **15 - 35 min** | **The Mel Scale & Filterbanks** | Introduce the Mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another. Show how to multiply the linear STFT magnitude matrix by a set of triangular Mel-filterbanks to compress thousands of narrow frequency bins into ~40-128 wide, perceptually relevant bands. |
# | **35 - 55 min** | **MFCCs (The Classic Feature)** | Mel-Frequency Cepstral Coefficients. Take the Discrete Cosine Transform (DCT) of the log-Mel spectrogram. Explain why: earlier ML algorithms (like Gaussian Mixture Models and Naive Bayes) assumed input features were uncorrelated. The DCT mathematically decorrelates the overlapping Mel bands. |
# | **55 - 75 min** | **Standardizing the UrbanSound Data** | *Live Code-Along:* Load an audio file from the UrbanSound8K dataset. Demonstrate the crucial preprocessing steps: downmixing stereo to mono, resampling everything to 22.05 kHz, and padding/truncating clips to exactly 4.0 seconds so every feature matrix $X$ has the exact same dimensions. |
#
# ### Key Equations to Board
#
# * **The Mel Scale Conversion:**
#
# $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
#
#
# * **Mel-Spectrogram Calculation:**
#
# $$S_{\text{mel}}[b, m] = \sum_{k} H_b[k] \vert{}X[m, k]\vert{}^2$$
#
#
#
# *(where $H_b[k]$ is the triangular weighting function for the $b$-th Mel band).*
# * **The Cepstrum (MFCC):**
#
# $$C[n] = \text{DCT}\{ \log(S_{\text{mel}}) \}$$
#
#
#
# ---
#
# ## Lecture 22: Machine Learning Foundations & Scikit-Learn
#
# **Objective:** Introduce classical supervised learning concepts from Raschka, explain the critical importance of data splitting (especially for audio), and implement baseline classifiers using Scikit-Learn.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 20 min** | **Supervised Learning Basics** | Define the ML paradigm: instead of writing rules to map $X \to y$, we provide $X$ (features) and $y$ (labels: "siren", "dog bark") and let the algorithm learn the mapping function. Define the feature matrix $\mathbf{X}$ (shape: `[n_samples, n_features]`) and target vector $\mathbf{y}$. |
# | **20 - 40 min** | **The Golden Rule: Data Splitting** | Read from Raschka Ch. 6. Explain Train, Validation, and Test splits. *Crucial Audio Warning:* Discuss "Data Leakage." In UrbanSound8K, a 10-second recording of a siren is sliced into multiple 4-second clips. If you randomly shuffle clips, slices from the *same physical siren* end up in both training and testing sets, ruining the evaluation. They MUST use the pre-defined folds. |
# | **40 - 55 min** | **Flattening 2D Audio for 1D Models** | Traditional ML models (SVMs, Random Forests) expect a 1D array of features per sample, but our MFCCs are 2D `[n_mels, n_time_frames]`. Discuss strategies: flatten the entire matrix into a giant 1D vector (preserves time, explodes dimensionality), or take the mathematical mean across the time axis (drastically reduces size, destroys temporal sequence). |
# | **55 - 75 min** | **The Scikit-Learn API** | *Live Python Demo:* Introduce the elegant consistency of Scikit-Learn. Instantiate a model: `clf = RandomForestClassifier()`. Train it: `clf.fit(X_train, y_train)`. Test it: `y_pred = clf.predict(X_test)`. Evaluate it using a confusion matrix. |
#
# ### Key Concepts to Board
#
# * **The Supervised Learning Mapping:**
#
# $$\hat{y} = f(\mathbf{x}; \boldsymbol{\theta})$$
#
#
#
# *(where $\mathbf{x}$ is the input feature vector, $\boldsymbol{\theta}$ are the learned model weights, and $\hat{y}$ is the predicted class).*
# * **Feature Matrix Shape (Scikit-Learn Standard):**
#
# $$\mathbf{X} \in \mathbb{R}^{N \times D}$$
#
#
#
# *(where $N$ is the number of audio clips, and $D$ is the number of flattened MFCC features).*
# * **Time-Averaging Features (Dimensionality Reduction):**
#
# $$\bar{c}[b] = \frac{1}{T} \sum_{t=1}^{T} C[b, t]$$
#
#
#
# > **Instructor Tip for PyTorch Prep:** End the week by looking at the accuracy of the baseline Random Forest model. It will likely max out around 60-65% on the UrbanSound dataset. Ask the students *why* it hit a ceiling. Guide them to the realization: by flattening the spectrogram or taking the mean, we destroyed the spatial patterns (like the visual "curve" of a dog bark's pitch). Next week, we will use Convolutional Neural Networks (CNNs) in PyTorch to look at the 2D spectrogram as a complete picture, preserving those spatial relationships.
#
#
# ----------------------------------
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# Chat GPT: (worked for 11 min 51 sec)
#
# Done. I created the complete **Week 11 DSP → ML transition package** for the two 75-minute classes.
#
# ### Complete package
#
# [Download the complete Week 11 package](sandbox:/mnt/data/week11_dsp_ml/Week11_DSP_ML_Complete_Package.zip)
#
# ### Individual files
#
# * [Lecture 21 — Feature Engineering: From Physics to Perception](sandbox:/mnt/data/week11_dsp_ml/Week11_Lecture21_Feature_Engineering_Mel_MFCC.pptx) — 22 slides.
# * [Lecture 22 — Machine Learning Foundations & Scikit-Learn](sandbox:/mnt/data/week11_dsp_ml/Week11_Lecture22_ML_Foundations_ScikitLearn.pptx) — 22 slides.
# * [Week 11 Jupyter Notebook](sandbox:/mnt/data/week11_dsp_ml/Week11_DSP_ML_Code.ipynb) — executed end-to-end with synthetic examples and an optional UrbanSound8K workflow.
# * [Standalone Python Script](sandbox:/mnt/data/week11_dsp_ml/Week11_DSP_ML_Code.py)
# * [Homework + Weekly Quiz + Complete Solutions](sandbox:/mnt/data/week11_dsp_ml/Week11_Homework_Quiz_Solutions.docx)
#
# ### Some deliberate additions
#
# **Lecture 21** makes the conceptual transition explicit:
#
# $$\text{STFT}
# \rightarrow
# \text{Mel pooling}
# \rightarrow
# \log
# \rightarrow
# \text{DCT}
# \rightarrow
# \text{MFCCs}.$$
#
# I included a matrix interpretation,
#
# $$S_{\mathrm{mel}}=HP,$$
#
# so students can see that Mel filtering is initially just a linear transformation of STFT power, followed by nonlinear log compression.
#
# I also added an important implementation nuance: the familiar
#
# $$m=2595\log_{10}(1+f/700)$$
#
# is the HTK-style Mel formula, while current `librosa` defaults to the Slaney convention (`htk=False`). ([Librosa][1]) `librosa.feature.mfcc` also uses a DCT, with type II as the current default. ([Librosa][2])
#
# The slides deliberately say that the DCT **approximately** decorrelates/compacts log-Mel information rather than claiming statistical independence.
#
# **Lecture 22** puts substantial emphasis on leakage. The sequence is:
#
# $$\text{audio source}
# \rightarrow
# \text{source-aware folds}
# \rightarrow
# \text{feature extraction}
# \rightarrow
# X,y
# \rightarrow
# \text{fit}
# \rightarrow
# \text{held-out evaluation}.$$
#
# The course example uses folds **1–8 for training, 9 for validation, and 10 for test** while making clear that this is a convenient course convention; rotating predefined folds gives a stronger evaluation.
#
# I also changed one assumption from your outline: the slides do **not** promise that the Random Forest will “max out at 60–65%.” Its result depends heavily on the feature representation, split policy, and hyperparameters. The point of the baseline is instead to establish what compact hand-engineered features can accomplish before the CNN is introduced.
#
# The scikit-learn examples follow the current estimator/Pipeline conventions, including `fit`, `predict`, `RandomForestClassifier`, and training-only preprocessing. ([Scikit-learn][3])
#
# ### Notebook design
#
# The notebook runs without needing UrbanSound8K installed. It first creates synthetic harmonic, chirp, and noise classes so students can verify the entire
#
# $$\text{waveform}
# \rightarrow
# \text{Mel}
# \rightarrow
# \text{MFCC}
# \rightarrow
# \text{Random Forest}
# \rightarrow
# \text{confusion matrix}$$
#
# pipeline.
#
# Then there is a separate UrbanSound8K section that activates after changing:
#
# ```python
# URBANSOUND_ROOT = Path('/path/to/UrbanSound8K')
# ```
#
# It demonstrates source-aware fold handling rather than randomly shuffling clips.
#
# The two PowerPoints were rendered and checked for slide overflow, and the homework/quiz document was rendered page-by-page and visually verified.
#
# A useful next checkpoint is to build Week 12 around the explicit question raised at the end of Lecture 22: **what information did time-averaged MFCCs destroy that a 2D CNN can exploit?**
#
# [1]: https://librosa.org/doc/latest/generated/librosa.filters.mel.html?utm_source=chatgpt.com "librosa.filters.mel — librosa 0.11.0 documentation"
# [2]: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html?utm_source=chatgpt.com "librosa.feature.mfcc — librosa 0.11.0 documentation"
# [3]: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?utm_source=chatgpt.com "Pipeline — scikit-learn 1.9.0 documentation"
#
# --------------------------------------
#
#

# %% [markdown]
# ## Week 11, Lecture 21, Latex math
#
# Can you convert the math expressions to LaTeX
#
# Gemini Pro:
#
# Explain in more detail what's going on here and what good answers and/or examples are
#
# Chat GPT:
#
# This lecture is doing something conceptually different from the earlier DSP weeks. Up to now, the course has mostly asked:
#
# $$\boxed{\text{How faithfully can we represent and manipulate a signal?}}$$
#
# Lecture 21 changes the question to:
#
# $$\boxed{\text{What information should we keep if the goal is classification rather than reconstruction?}}$$
#
# That shift is explicitly built into the opening slides: classical DSP cares about representing, filtering, or reconstructing the signal, while ML cares about finding a representation that makes the decision boundary easier to learn. The slides also emphasize that feature engineering deliberately discards information, so every feature choice encodes an assumption about what matters to the label.
#
# ## 1. The key philosophical shift: “good representation” now depends on the task
#
# Suppose we have two audio recordings of a dog bark. From a DSP standpoint, the exact waveform matters. If we want perfect reconstruction, we care about phase, sample rate, exact amplitude, and every frequency component.
#
# For classification, however, perhaps we only care about characteristics such as:
#
# $$\text{broad spectral envelope}, \qquad \text{energy distribution}, \qquad \text{time-frequency pattern}.$$
#
# If two waveforms are perceptually and semantically both “dog bark,” it may actually be helpful for our representation to ignore differences such as absolute phase.
#
# That is the meaning of the slide statement:
#
# $$\boxed{\text{Preserve what matters for the label; compress what does not.}}$$
#
# A strong classroom question would be:
#
# > “Give me an example where a lossy representation could be better for classification than the original waveform.”
#
# A good answer is:
#
# > For siren classification, the exact waveform sample values may be irrelevant, while the pattern of frequency modulation and spectral energy is highly informative. Compressing the waveform into a Mel-spectrogram can discard irrelevant phase information and make the useful pattern much easier for the classifier to see.
#
# Another good answer would be keyword spotting: exact waveform reconstruction is unnecessary if the feature preserves the spectral-temporal cues that distinguish “yes” from “no.”
#
# ---
#
# ## 2. Why the raw STFT is not automatically the best ML feature
#
# The STFT is physically meaningful. If we use
#
# $$n_{\text{fft}}=2048,$$
#
# then a real-valued frame produces
#
# $$\frac{2048}{2}+1=1025$$
#
# one-sided frequency bins. The lecture notes use that exact example at 22.05 kHz.
#
# The spacing is approximately
#
# $$\Delta f = \frac{f_s}{N_{\text{FFT}}} = \frac{22050}{2048} \approx 10.77\text{ Hz}.$$
#
# So the STFT distinguishes approximately 1000 Hz from 1010.8 Hz, and equally distinguishes 10,000 Hz from 10,010.8 Hz.
#
# Mathematically that is beautifully uniform.
#
# Perceptually, though, that is not necessarily an efficient allocation of features.
#
# At low frequencies, a difference of 100 Hz can be very significant. At high frequencies, using enormous numbers of narrow linear-Hz bins may provide much more detail than many classification tasks need.
#
# The slide wisely avoids the oversimplification “human hearing is logarithmic” and instead describes Mel as a perceptual warping.
#
# That nuance is worth preserving in class.
#
# A better statement is:
#
# > Human frequency perception is strongly nonlinear, and the Mel scale is one engineered approximation to perceptual frequency spacing.
#
# ---
#
# ## 3. What does the Mel scale actually do?
#
# The slide gives the common HTK-style formula
#
# $$\boxed{m = 2595\log_{10}\left(1+\frac{f}{700}\right).}$$
#
# It also correctly notes that this is one convention and that `librosa` defaults to a Slaney-style convention unless `htk=True`.
#
# The exact constants are less important than the shape of the mapping.
#
# At low frequencies, Mel changes fairly rapidly with $f$.
#
# At higher frequencies, increasingly large changes in Hz correspond to similar changes in Mel.
#
# So the transformation effectively allocates:
#
# $$\boxed{\text{more resolution at low frequencies}}$$
#
# and
#
# $$\boxed{\text{less resolution at high frequencies}.}$$
#
# A good question is:
#
# > “Does converting Hz to Mel somehow change the physical frequencies in the signal?”
#
# No.
#
# A good answer is:
#
# > No. It changes the coordinate system or grouping used to describe the spectral content. The original frequencies do not physically move; we are pooling them into perceptually motivated bands.
#
# ---
#
# ## 4. The most important idea: the Mel spectrogram is weighted frequency pooling
#
# Students sometimes think `librosa.feature.melspectrogram()` is doing something mysterious.
#
# The mathematics is actually quite simple.
#
# Suppose the STFT power is
#
# $$P[k,m] = \vert{}X[m,k]\vert{}^2,$$
#
# where
#
# * $k$ is FFT-bin index,
# * $m$ is time-frame index.
#
# The $b$-th Mel band has weights
#
# $$H_b[k].$$
#
# Then
#
# $$\boxed{S_{\mathrm{mel}}[b,m] = \sum_k H_b[k]P[k,m].}$$
#
# That equation is exactly the conceptual definition in the lecture.
#
# So each Mel output value asks:
#
# > “How much power is present in this perceptually defined frequency region during this time frame?”
#
# That is the correct intuition.
#
# ---
#
# ## 5. What do the triangular filters mean?
#
# Imagine three neighboring Mel filters:
#
# ```text
# weight
#   1            /\        /\        /\
#               /  \      /  \      /  \
#   0 ---------/----\----/----\----/----\------ frequency
#
# ```
#
# One FFT bin may contribute to two adjacent Mel bands.
#
# If a frequency lies exactly at the center of a Mel filter, it receives a relatively large weight.
#
# If it lies near the filter's edge, it receives a smaller weight.
#
# The slides emphasize three useful properties: low-frequency filters are relatively narrow, high-frequency filters cover more Hz, and adjacent filters overlap.
#
# A strong explanation for the overlap is:
#
# > We do not want abrupt boundaries where a tiny shift in frequency suddenly transfers all energy from one feature to another. Overlap makes the representation smoother with respect to small frequency changes.
#
# That “smoothness” can be valuable in classification.
#
# ---
#
# ## 6. The matrix form is especially useful for ECE students
#
# The deck writes the Mel operation as
#
# $$\boxed{S_{\text{mel}}=HP}$$
#
# with dimensions
#
# $$P\in\mathbb{R}^{K\times T},$$
#
# $$H\in\mathbb{R}^{B\times K},$$
#
# and
#
# $$S_{\text{mel}}\in\mathbb{R}^{B\times T}.$$
#
# This is a very nice linear-algebra bridge.
#
# For each time frame, the original $K$-dimensional FFT power vector is projected into a $B$-dimensional Mel-band vector.
#
# If
#
# $$K=1025$$
#
# and
#
# $$B=64,$$
#
# then every time frame is compressed from 1025 frequency values to 64 values.
#
# The time axis is not compressed by the Mel filterbank.
#
# That distinction is very important.
#
# A good question is:
#
# > “Does changing `n_mels` affect the time resolution?”
#
# Correct answer:
#
# > No. The Mel bank operates along frequency. Time resolution is inherited primarily from the STFT window and hop length.
#
# That is explicitly stated in the matrix-view slide.
#
# ---
#
# ## 7. What information did we lose by Mel pooling?
#
# This is worth asking before moving on.
#
# If we combine 1025 STFT bins into 64 Mel bands, we cannot later recover the exact original 1025-bin spectrum.
#
# For example, suppose two nearby high-frequency sinusoids fall largely into the same Mel band.
#
# Their Mel-band energies may look nearly identical.
#
# Therefore:
#
# $$\boxed{\text{Mel pooling deliberately creates frequency invariance.}}$$
#
# That can be beneficial if the classifier should not care about very small frequency shifts.
#
# But it may be harmful if very fine frequency distinctions are actually important.
#
# For example, a machine-condition-monitoring task may depend on narrow spectral lines whose exact locations matter.
#
# That is why the deck says no representation is free: compression trades detail for invariance or efficiency.
#
# ---
#
# ## 8. The `librosa` code mirrors the mathematics
#
# The lecture uses
#
# ```python
# M = librosa.feature.melspectrogram(
#     y=y,
#     sr=22050,
#     n_fft=2048,
#     hop_length=512,
#     n_mels=64,
#     power=2.0
# )
#
# ```
#
# and then
#
# ```python
# M_db = librosa.power_to_db(M, ref=np.max)
#
# ```
#
# The expected first dimension is 64 because `n_mels=64`.
#
# This is a very good place to ask students what every parameter controls.
#
# For `n_fft=2048`, a good answer is:
#
# > It sets the FFT size and therefore contributes to frequency resolution.
#
# For `hop_length=512`:
#
# > It determines the step between successive analysis frames and therefore affects time sampling of the spectrogram.
#
# For `n_mels=64`:
#
# > It determines how many frequency bands remain after perceptual frequency compression.
#
# For `power=2.0`:
#
# > The feature uses squared magnitude, so the Mel values represent power-like quantities rather than magnitude.
#
# ---
#
# ## 9. Why do we take a logarithm?
#
# Raw spectral powers may span many orders of magnitude.
#
# Suppose one band has power 1 and another has 0.000001.
#
# A linear display makes the smaller component almost invisible.
#
# The slides therefore introduce
#
# $$\boxed{S_{\mathrm{dB}} = 10\log_{10}\left(\frac{S}{S_{\mathrm{ref}}}\right)}$$
#
# for power quantities.
#
# The logarithm does several useful things.
#
# First, it compresses dynamic range.
#
# Second, ratios become differences.
#
# For example:
#
# $$\frac{S_1}{S_2}=100$$
#
# corresponds to
#
# $$10\log_{10}(100)=20\text{ dB}.$$
#
# A factor of 10 in power corresponds to 10 dB.
#
# A factor of 100 corresponds to 20 dB.
#
# ---
#
# ## 10. Why $10\log_{10}$ sometimes and $20\log_{10}$ other times?
#
# This is a very good DSP reinforcement point.
#
# For power ratio:
#
# $$\boxed{10\log_{10}\left(\frac{P_2}{P_1}\right)}$$
#
# For amplitude ratio:
#
# $$\boxed{20\log_{10}\left(\frac{A_2}{A_1}\right)}$$
#
# because power is proportional to amplitude squared.
#
# If
#
# $$P\propto A^2,$$
#
# then
#
# $$10\log_{10}(A^2) = 20\log_{10}(A).$$
#
# A good clicker-style question would be:
#
# > If amplitude doubles, what is the change in dB?
#
# $$20\log_{10}(2) \approx 6.02\text{ dB}.$$
#
# If power doubles:
#
# $$10\log_{10}(2) \approx 3.01\text{ dB}.$$
#
# ---
#
# ## 11. Why do we need a floor before the logarithm?
#
# Because
#
# $$\log(0)$$
#
# is undefined.
#
# Numerically, you need something equivalent to
#
# $$\log(S+\epsilon)$$
#
# or a clipping/floor strategy.
#
# The lecture explicitly mentions this.
#
# A good student answer to
#
# > “Why can't we just call `np.log(S)`?”
#
# would be:
#
# > Because zero-valued bins would produce $-\infty$, which is not a usable finite ML feature and can propagate numerical problems.
#
# ---
#
# # 12. Now comes MFCC: what problem are we trying to solve?
#
# After Mel filtering and log compression, we have something like
#
# $$\log S_{\mathrm{mel}}[b,m].$$
#
# Neighboring Mel bands are correlated because the triangular filters overlap.
#
# The MFCC step applies a DCT across the Mel-band dimension:
#
# $$\boxed{C[n,m] = \operatorname{DCT}_b \{ \log S_{\mathrm{mel}}[b,m] \}.}$$
#
# The deck describes MFCCs as a compact spectral-envelope representation, with roughly the first 13–40 coefficients often retained.
#
# The conceptual pipeline is:
#
# $$\boxed{\text{waveform} \rightarrow \text{STFT} \rightarrow \text{Mel} \rightarrow \log \rightarrow \text{DCT} \rightarrow \text{MFCC}.}$$
#
# ---
#
# # 13. What is the DCT doing conceptually?
#
# The DCT is changing basis.
#
# Before the DCT, your coordinates are:
#
# $\text{energy in Mel band 1}, \text{energy in Mel band 2}, \dots$
#
# After the DCT, the coordinates describe patterns of variation across the Mel bands.
#
# The first few cosine basis patterns are slowly varying.
#
# Higher-index DCT basis functions oscillate more rapidly across Mel-band index.
#
# That makes the low-order MFCC coefficients useful for representing broad spectral-envelope shape.
#
# The deck says the DCT gives approximate decorrelation and energy compaction, while carefully warning that “decorrelation” is not a universal guarantee.
#
# That caveat is important.
#
# ---
#
# # 14. A useful visual analogy for MFCC coefficients
#
# Imagine the log-Mel spectrum across frequency is smooth:
#
# ```text
# energy
#   ^
#   |        ______
#   |       /        \
#   |____/          \____
#   +--------------------> Mel band
#
# ```
#
# A few low-order cosine basis functions can describe this broad shape well.
#
# If the spectrum contains very fine alternating structure:
#
# ```text
# energy
#   ^
#   | /\ /\ /\ /\ /\
#   |/  V  V  V  V  \
#   +-----------------> Mel band
#
# ```
#
# you need higher-order DCT coefficients.
#
# So a good intuitive statement is:
#
# $$\boxed{\text{low MFCC index} \approx \text{broad spectral-envelope structure}}$$
#
# while
#
# $$\boxed{\text{higher MFCC index} \approx \text{finer variation across Mel frequency}.}$$
#
# That matches the interpretation slide.
#
# ---
#
# # 15. What does MFCC coefficient 0 mean?
#
# The deck says coefficient 0 largely tracks overall log-energy.
#
# Why?
#
# The first DCT basis is approximately constant across the Mel bands.
#
# So it essentially adds or averages the log-Mel energy across frequency.
#
# Therefore a good answer is:
#
# > $C_0$ mostly represents the broad overall level of the log spectrum rather than fine spectral shape.
#
# This raises a useful design question:
#
# > “Should we keep $C_0$?”
#
# Answer:
#
# > It depends. If overall level contains meaningful class information, yes. If level mostly reflects recording gain, distance to microphone, or nuisance variation, it may be less useful.
#
# That is exactly the type of feature-engineering reasoning you want them to develop.
#
# ---
#
# # 16. Why were MFCCs historically attractive?
#
# Older classical ML models often benefited from:
#
# * fewer dimensions,
# * less feature correlation,
# * compact representation of broad spectral shape.
#
# A Gaussian model with diagonal covariance, for instance, is much happier if features are only weakly correlated.
#
# So a good historical explanation is:
#
# > The DCT helped produce compact coordinates that were easier for classical statistical models to handle.
#
# But I would avoid telling students that MFCCs are mathematically independent.
#
# The slide explicitly says that the decorrelation is approximate and data-dependent.
#
# ---
#
# # 17. What do MFCCs throw away?
#
# Quite a lot.
#
# The lecture states that MFCCs intentionally discard much detailed harmonic structure.
#
# That is sometimes beneficial.
#
# For speech recognition, we often care more about the broad spectral envelope associated with vocal-tract resonances than about the exact harmonic fine structure set by pitch.
#
# But for tasks involving:
#
# * musical pitch,
# * narrow tonal faults,
# * precise harmonics,
# * fine spectro-temporal structure,
#
# MFCC compression may throw away exactly what the classifier needs.
#
# A great question is:
#
# > “Would you use MFCCs to estimate musical pitch?”
#
# A good answer:
#
# > Probably not as the first choice, because MFCCs intentionally suppress much of the detailed harmonic structure that carries pitch information.
#
# ---
#
# # 18. Mel spectrogram versus MFCC matrix versus MFCC summary
#
# The deck gives three very useful representational choices.
#
# A full Mel-spectrogram retains
#
# $$[\text{frequency band} \times \text{time}]$$
#
# geometry.
#
# That is attractive for CNNs.
#
# An MFCC matrix still retains time:
#
# $$[\text{MFCC coefficient} \times \text{time}].$$
#
# It is smaller and more compact.
#
# But if you take mean and standard deviation over time, you may get a fixed vector like
#
# $$[\mu_1, \ldots, \mu_{20}, \sigma_1, \ldots, \sigma_{20}],$$
#
# which has only 40 features.
#
# That is attractive for Random Forests or SVMs.
#
# But now temporal ordering is gone.
#
# ---
#
# # 19. A particularly good example of lost temporal ordering
#
# Imagine two sounds with the same average spectral content:
#
# Sound A:
#
# $$\text{low frequency} \rightarrow \text{high frequency}.$$
#
# Sound B:
#
# $$\text{high frequency} \rightarrow \text{low frequency}.$$
#
# A frequency sweep upward and a frequency sweep downward can have nearly identical **time-averaged** spectra.
#
# If you replace the entire spectrogram by mean Mel energy or mean MFCCs, the two may become almost indistinguishable.
#
# That is an excellent explanation of why time averaging can hurt.
#
# A CNN looking at the 2D spectrogram sees the difference immediately:
#
# ```text
# up-chirp:          down-chirp:
#
# frequency ^        frequency ^
#           /                  \
#          /                    \
#         /                      \
# time -->            time -->
#
# ```
#
# This is probably one of the best examples to use as the bridge to the next week's CNN material.
#
# ---
#
# # 20. Why standardize waveform geometry before extracting features?
#
# The later part of the lecture changes from feature theory to dataset engineering.
#
# The deck requires consistency in:
#
# * channel policy,
# * sample rate,
# * duration,
# * amplitude convention,
# * feature parameters.
#
# This is not just bookkeeping.
#
# ML models expect comparable examples.
#
# If one clip is sampled at 44.1 kHz and another at 16 kHz, FFT bin 100 does not correspond to the same physical frequency.
#
# If one file is 1 second and another is 4 seconds, the resulting feature matrices have different time dimensions.
#
# So preprocessing creates a common coordinate system.
#
# ---
#
# # 21. Stereo to mono is not automatically harmless
#
# The deck uses
#
# $$y_{\text{mono}}[n] = \frac{1}{2}\left(y_L[n]+y_R[n]\right).$$
#
# For UrbanSound-style classification, averaging may be completely reasonable if spatial location is irrelevant.
#
# But consider a task like:
#
# > “Did the sound come from the left or right?”
#
# Then stereo contains label information.
#
# Averaging the channels would destroy the very feature we need.
#
# That's why the slide says the channel decision is part of the measurement model.
#
# A good answer to
#
# > “When is mono conversion inappropriate?”
#
# would be:
#
# > When inter-channel delay, amplitude difference, direction of arrival, or spatial cues are related to the target label.
#
# ---
#
# # 22. Why resample everything to 22.05 kHz?
#
# The lecture uses a course convention of
#
# $$f_s = 22050\text{ Hz}.$$
#
# For 4 seconds:
#
# $$N = 4(22050) = 88200$$
#
# samples.
#
# Why standardize the sample rate?
#
# Because frequency coordinates must mean the same thing for every file.
#
# If
#
# $$N_{\text{FFT}}=2048,$$
#
# then at 22.05 kHz:
#
# $$\Delta f = \frac{22050}{2048} \approx 10.77\text{ Hz}.$$
#
# But at 44.1 kHz:
#
# $$\Delta f \approx 21.53\text{ Hz}.$$
#
# Same FFT bin index, different physical frequency.
#
# That would be problematic for consistent features.
#
# ---
#
# # 23. Why 22.05 kHz may or may not be appropriate
#
# Nyquist becomes
#
# $$f_N=\frac{22050}{2}=11025\text{ Hz}.$$
#
# Therefore everything above approximately 11.025 kHz cannot be represented after resampling.
#
# The lecture correctly says that this is acceptable only if the task does not need higher-frequency information.
#
# A great question is:
#
# > “Would 22.05 kHz be safe for every audio classification problem?”
#
# No.
#
# For example, if distinguishing classes depends on ultrasonic or very high-frequency components, this resampling choice could destroy useful information.
#
# Once again:
#
# $$\boxed{\text{feature preprocessing encodes assumptions}.}$$
#
# ---
#
# # 24. Week 2 comes back: never forget anti-aliasing
#
# The deck explicitly reminds students that resampling must use an anti-aliased method and that naive sample dropping is not legitimate resampling.
#
# This is a great callback.
#
# If a 44.1-kHz recording is naively decimated to 22.05 kHz without low-pass filtering, content above 11.025 kHz can alias into the lower band.
#
# Then the ML model may learn those aliases.
#
# So the earlier DSP theory is not separate from ML preprocessing.
#
# It determines whether the features are physically meaningful.
#
# ---
#
# # 25. Why pad or truncate clips to four seconds?
#
# The code uses a target length of 88,200 samples.
#
# If shorter:
#
# ```python
# y = np.pad(...)
#
# ```
#
# If longer:
#
# ```python
# y = y[:target_len]
#
# ```
#
# The goal is to make every waveform—and therefore every feature matrix—have predictable dimensions.
#
# This is convenient for batching and classical models.
#
# But it also creates new assumptions.
#
# ---
#
# # 26. Padding location matters
#
# Suppose a 1-second dog bark is in a 4-second tensor.
#
# If you right-pad:
#
# ```text
# DOG BARK | silence | silence | silence
#
# ```
#
# If you center it:
#
# ```text
# silence | DOG BARK | silence
#
# ```
#
# A time-sensitive model may learn these positions differently.
#
# So padding is not neutral.
#
# The deck calls the padding location a design choice.
#
# For training a CNN, random temporal placement can sometimes be useful augmentation because it teaches the model that the event can occur anywhere.
#
# ---
#
# # 27. Truncation can be much more dangerous than students expect
#
# Suppose a 6-second recording contains a siren only during seconds 5–6.
#
# Right truncation to 4 seconds gives you:
#
# ```text
# 0 ---- 4 s | discarded
#              SIREN
#
# ```
#
# Your file is still labeled “siren,” but the resulting input may contain no siren at all.
#
# Now you've introduced label noise.
#
# The lecture points out that random or multiple crops may be better for training.
#
# That's an excellent place to connect preprocessing choices to ML dataset quality.
#
# ---
#
# # 28. Shape bookkeeping is more important than it looks
#
# The slide gives approximately
#
# $$[88200]$$
#
# for waveform,
#
# $$[1025, \sim 173]$$
#
# for STFT power,
#
# $$[64, \sim 173]$$
#
# for Mel,
#
# $$[20, \sim 173]$$
#
# for MFCC,
#
# and
#
# $$[40]$$
#
# for MFCC mean+std.
#
# This is a very good debugging exercise.
#
# Students should predict array shapes before running code.
#
# For example:
#
# ```python
# mfcc.shape
#
# ```
#
# should have first dimension 20 if
#
# ```python
# n_mfcc=20
#
# ```
#
# If it doesn't, something is wrong.
#
# ---
#
# # 29. Why is the frame count “about 173” rather than always exactly 173?
#
# Because STFT implementations have conventions involving:
#
# * padding,
# * centering,
# * frame alignment,
# * boundary treatment.
#
# The deck explicitly warns that exact frame count depends on centering/padding conventions.
#
# This is a good place to discourage students from memorizing one output dimension without understanding where it came from.
#
# A rough no-padding estimate is
#
# $$N_{\text{frames}} \approx 1+ \left\lfloor \frac{N-N_{\text{FFT}}}{H} \right\rfloor$$
#
# where $H$ is the hop length.
#
# With library centering, the result changes.
#
# ---
#
# # 30. Flattening versus mean/std: a dramatic dimensionality tradeoff
#
# The slide points out:
#
# $$20 \times 173 = 3460$$
#
# MFCC values if flattened.
#
# Mean plus standard deviation gives only
#
# $$20+20=40$$
#
# features.
#
# This is a huge reduction:
#
# $$3460 \rightarrow 40.$$
#
# The benefit:
#
# * much smaller model input,
# * easier classical ML,
# * less risk of overfitting with small datasets.
#
# The cost:
#
# * temporal order gone,
# * short events can get averaged away,
# * onset/offset structure disappears.
#
# So a good answer to
#
# > “Why not always use mean MFCCs?”
#
# would be:
#
# > Because they are compact but discard temporal structure. They are useful when long-term spectral statistics are enough for the classification task, but poor when event timing or spectral trajectories are important.
#
# ---
#
# # 31. The live code-along is really about invariants
#
# The slide's code is
#
# ```python
# y, sr0 = librosa.load(path, sr=None, mono=True)
# y = librosa.resample(y, orig_sr=sr0, target_sr=22050)
# y = fix_length(y, int(4.0 * 22050))
#
# mel = librosa.feature.melspectrogram(...)
# mel_db = librosa.power_to_db(...)
# mfcc = librosa.feature.mfcc(...)
#
# ```
#
# The instructor cue asks a very strong question after each line:
#
# > “What invariant did we establish, and what might we have discarded?”
#
# Here are good answers.
#
# After mono conversion:
#
# > Invariant: one channel per example. Potentially discarded: stereo/spatial information.
#
# After resampling:
#
# > Invariant: common sampling rate and physical frequency grid. Discarded: information above the new Nyquist frequency.
#
# After fixing duration:
#
# > Invariant: same number of waveform samples and eventually comparable feature sizes. Discarded: anything truncated; padded examples gain artificial silence.
#
# After Mel conversion:
#
# > Invariant: same Mel-band coordinate system. Discarded: detailed linear-frequency resolution.
#
# After MFCC:
#
# > Invariant: compact spectral-envelope coordinate system. Discarded: additional fine spectral detail.
#
# That progression would make an excellent board discussion.
#
# ---
#
# # 32. The concept-check answer and why the others are wrong
#
# The deck asks which statement is correct and gives **C**:
#
# > Mean MFCCs reduce dimensionality but remove temporal ordering.
#
# Why not A?
#
# > Increasing `n_mels` gives more features, but more dimensions do not automatically improve classification. You may add redundant or noisy detail and increase model complexity.
#
# Why not B?
#
# > Stereo-to-mono conversion can destroy spatial information that matters to the task.
#
# Why not D?
#
# > Converting power to dB is a nonlinear compression. It does not recreate frequency detail already lost when many STFT bins were pooled into Mel bands.
#
# That's a very good concept question because all four statements test feature-engineering reasoning rather than formula memorization.
#
# ---
#
# # 33. A useful extra example: why feature engineering can improve generalization
#
# Suppose we want to identify whether an audio clip contains a siren.
#
# Raw waveform samples may vary enormously because of:
#
# * microphone position,
# * phase,
# * exact sample alignment,
# * volume,
# * reflections,
# * background noise.
#
# But the siren's broad spectral and temporal pattern may remain similar.
#
# A Mel representation suppresses some irrelevant variation and retains that pattern.
#
# So feature engineering can make examples belonging to the same class “closer” in feature space.
#
# That makes the ML decision boundary easier to learn.
#
# This connects directly to the opening claim that the ML representation should make classification easier.
#
# ---
#
# # 34. But feature engineering can also make the task impossible
#
# Suppose the label is:
#
# > “Was this tone 10,000 Hz or 10,050 Hz?”
#
# If the Mel representation pools both tones into essentially the same band, the classifier cannot recover the difference.
#
# No amount of machine learning can restore information that the feature extractor removed.
#
# This is the Week 11 version of the Week 2 aliasing lesson:
#
# $$\boxed{\text{information discarded before the model cannot be learned later}.}$$
#
# The mechanism is different, but the principle is the same.
#
# ---
#
# # 35. Good questions I would ask throughout the lecture
#
# A particularly useful set would be:
#
# * “Which axis does a Mel filterbank compress—time or frequency?”
# * “Why can two different STFT spectra map to the same Mel vector?”
# * “Why do we log the Mel powers?”
# * “What does the DCT change: information content or coordinate basis?”
# * “Why are low-order MFCCs often retained?”
# * “What information disappears when we average MFCCs across time?”
# * “Why must sample rate be standardized across examples?”
# * “When would downmixing stereo be unacceptable?”
# * “Why isn't converting to float32 a substitute for signal standardization?”
# * “What assumption are we making when we truncate everything to four seconds?”
#
# Those questions get at the actual thinking you want from senior/graduate students.
#
# ---
#
# # 36. What I would want a strong student to say at the end
#
# A mediocre summary might be:
#
# > “MFCCs are used for audio classification.”
#
# A much better answer would be:
#
# > “We begin with a physically meaningful STFT, pool linear frequency bins through perceptually motivated Mel filters, compress power with a logarithm, optionally rotate the log-Mel coordinates with a DCT to produce compact MFCC features, and then standardize channel count, sample rate, duration, and feature dimensions so every example is comparable to an ML model.”
#
# An excellent answer would add:
#
# > “Each step trades information for invariance or efficiency, so the representation must be chosen according to what information the classification task actually needs.”
#
# That is exactly the deeper message of this lecture.
#
# The whole pipeline can therefore be summarized as
#
# $$\boxed{x[n] \rightarrow \vert{}X[m,k]\vert{}^2 \rightarrow S_{\text{mel}}[b,m] \rightarrow \log S_{\text{mel}} \rightarrow C[n,m] \rightarrow \text{fixed ML feature geometry}.}$$
#
# And the most important conceptual question to keep asking students is:
#
# $$\boxed{\text{At this step, what useful information did we preserve—and what did we just throw away?}}$$
#
# That question connects nearly every slide in Lecture 21 and is an excellent mindset to carry into Lecture 22 and the later CNN material.
#
# -------------------

# %% [markdown]
# ## Week 11, Lecture 22, Latex math
#
# Can you convert the math expressions to LaTeX
#
# Gemini Pro:
#
# Explain in more detail what's going on here and what good answers and/or examples are
#
# Chat GPT:
#
# This lecture is really about **how to do machine learning without fooling yourself**.
#
# Lecture 21 was mostly about representation:
#
# **waveform → Mel/MFCC features**
#
# Lecture 22 asks:
#
# > **Now that we have features, how do we learn from them and evaluate the result honestly?**
#
# The central progression in the slides is
#
# $$\text{features } X + \text{labels } y \rightarrow \text{train} \rightarrow \text{validate} \rightarrow \text{test} \rightarrow \text{inspect errors}$$
#
# with the especially important warning that **data leakage can make a bad model look excellent**. The stated learning goals emphasize supervised learning, distinct data splits, source-level audio leakage, fixed-length feature vectors, and baseline evaluation with scikit-learn.
#
# ---
#
# ## 1. What changes when we move from DSP to supervised ML?
#
# In traditional programming, the engineer writes the rule.
#
# For example:
#
# ```text
# if dominant_frequency > threshold:
#     class = "siren"
#
# ```
#
# You explicitly decide what rule maps the input to the output.
#
# In supervised learning, instead we provide examples:
#
# $$(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)$$
#
# and ask an algorithm to learn parameters $\boldsymbol{\theta}$ so that
#
# $$\boxed{\hat{y} = f(\mathbf{x}; \boldsymbol{\theta})}$$
#
# predicts the label.
#
# The slides describe this as changing **who writes the decision rule**: the engineer provides labeled examples, and the learning algorithm estimates the model parameters.
#
# A very good opening question is:
#
# > “What does the engineer still decide if the machine is supposedly ‘learning the rules’?”
#
# Good answers:
#
# * which data to collect,
# * which labels to use,
# * what features to compute,
# * how to split the data,
# * which model family to use,
# * which hyperparameters to try,
# * what metric counts as success.
#
# That is important because ML is not “remove the engineer.”
#
# It changes the engineering problem.
#
# ---
#
# ## 2. What exactly are $X$ and $y$?
#
# Scikit-learn expects a tabular structure:
#
# $$\boxed{X \in \mathbb{R}^{N \times D}}$$
#
# where:
#
# * $N$ = number of examples,
# * $D$ = number of features per example.
#
# The targets are
#
# $$\boxed{y \in \{1, \dots, C\}^N}$$
#
# for a $C$-class classification problem.
#
# Each row $X[i, :]$ must correspond to exactly one label $y[i]$.
#
# ---
#
# ## 3. Concrete audio example
#
# Suppose you have 1000 clips.
#
# For each clip you compute:
#
# * 20 MFCC means,
# * 20 MFCC standard deviations.
#
# Then $D = 40$. Your feature matrix has shape
#
# $$X.\text{shape} = (1000, 40)$$
#
# and your label vector has shape
#
# $$y.\text{shape} = (1000,)$$
#
# A row might conceptually look like
#
# $$\mathbf{x}_i = [\mu_1, \dots, \mu_{20}, \sigma_1, \dots, \sigma_{20}]$$
#
# Its label could be $y_i =$ "dog bark".
#
# A good question:
#
# > “What would happen if one clip produced 40 features and another produced 38?”
#
# Answer:
#
# > A classical tabular estimator cannot use them directly because every row of $X$ must have the same dimensionality.
#
# ---
#
# ## 4. What counts as a feature?
#
# The deck defines a feature as a coordinate chosen to make prediction easier.
#
# That is a much better definition than “a number you feed into the model.”
#
# For audio:
#
# ### Raw waveform sample
#
# $x[n]$
# Very physical, but highly sensitive to alignment and very high dimensional.
#
# ### Mel-bin energy
#
# Localized spectral energy in a perceptually warped band.
#
# ### MFCC coefficient
#
# A compact combination of log-Mel energies.
#
# ### Mean MFCC
#
# One number summarizing one coefficient over the whole clip.
#
# Every step adds some invariance.
#
# But every step also loses something.
#
# ---
#
# ## 5. What does “invariance” mean here?
#
# Suppose the same siren occurs at:
#
# * 0.5 s into one clip,
# * 2.5 s into another.
#
# If you average MFCCs over time, those two examples may look similar.
#
# That is useful if the class label should not depend on *when* the siren occurred.
#
# So the representation becomes approximately invariant to event timing.
#
# But the cost is:
#
# **The model no longer knows the temporal order.**
#
# The slide explicitly uses this tradeoff: time averaging reduces sensitivity to when a sound happens but hides temporal ordering.
#
# ---
#
# ## 6. The three-way split: training, validation, test
#
# This is probably the most important conceptual section of the lecture.
#
# The deck gives the three sets different jobs.
#
# ### Training set
#
# Used to estimate model parameters.
#
# For example:
#
# * decision-tree split thresholds,
# * logistic-regression coefficients,
# * support-vector locations,
# * neural-network weights.
#
# Also used to fit data-dependent preprocessing.
#
# ---
#
# ### Validation set
#
# Used to make development decisions:
#
# * number of trees,
# * SVM $C$ parameter,
# * kernel choice,
# * number of MFCCs,
# * feature type,
# * regularization strength.
#
# ---
#
# ### Test set
#
# Used to answer:
#
# > “How well does the final chosen procedure generalize to genuinely unseen data?”
#
# It should not participate in model selection.
#
# ---
#
# ## 7. Why isn't the test set just another validation set?
#
# Because once you make decisions based on test performance, the test set is no longer unseen.
#
# Suppose you try:
#
# * Model A → 61% test accuracy
# * Model B → 64%
# * Model C → 67%
# * Model D → 69%
#
# and choose D because it performed best on the test set.
#
# You have effectively **trained your decision-making process on the test set**.
#
# The reported 69% is now optimistically biased.
#
# A good student answer to
#
# > “Why only open the test set after our choices are fixed?”
#
# is:
#
# > Because otherwise information about test performance influences model selection, so the final reported score is no longer an unbiased estimate of generalization.
#
# ---
#
# ## 8. Data leakage: what does it actually mean?
#
# Data leakage means information that should be unavailable during training somehow influences the model or preprocessing.
#
# The lecture focuses on a particularly dangerous audio form:
#
# **Source-level leakage.**
#
# The key problem is that segmented audio clips from the same physical recording are highly correlated.
#
# ---
#
# ## 9. The siren example is excellent
#
# Suppose one physical siren recording is 20 seconds long.
#
# You cut it into:
#
# * clip A1,
# * clip A2,
# * clip A3,
# * clip A4.
#
# If you randomly shuffle all clips, perhaps:
#
# ```text
# Training:
# A1
# A3
#
# Test:
# A2
# A4
#
# ```
#
# That looks like a valid split numerically.
#
# But scientifically, it is poor.
#
# Why?
#
# Because the train and test sets contain:
#
# * the same siren source,
# * the same microphone,
# * same background,
# * same recording gain,
# * same environment,
# * perhaps continuous neighboring time segments.
#
# The classifier can exploit source identity instead of learning the general concept “siren.”
#
# ---
#
# ## 10. Why might the resulting accuracy be very high?
#
# Imagine the model learns some subtle background hum unique to recording A.
#
# Then during testing, A2 contains the same hum.
#
# The classifier effectively says:
#
# > “I've heard this recording before.”
#
# It may score 95% even though it would perform badly on a completely new siren recorded elsewhere.
#
# This is why the slide's rule is:
#
# **Split by source, not merely by clip.**
#
# All segments from one underlying recording should remain in the same fold or split.
#
# ---
#
# ## 11. A great classroom question
#
# Which evaluation is harder?
#
# ### Experiment A
#
# Training contains segment 1 from one siren; testing contains segment 2 from the same siren recording.
#
# ### Experiment B
#
# Training contains several sirens; testing contains a siren recorded in a completely different environment.
#
# Answer:
#
# > B is harder, but also far more scientifically meaningful if deployment requires generalization to new recordings.
#
# This leads to an important ML principle:
#
# **A lower trustworthy score can be better science than a higher leaked score.**
#
# ---
#
# ## 12. UrbanSound8K and folds
#
# The lecture instructs students to honor the dataset's predefined folds rather than blindly running `train_test_split()` on excerpts.
#
# For the course baseline, the deck suggests:
#
# * Folds 1–8 → Train
# * Fold 9 → Validation
# * Fold 10 → Test
#
# The key point isn't that those particular numbers are sacred.
#
# The key idea is:
#
# > Fold identity carries information about how the dataset should be evaluated.
#
# A stronger evaluation can rotate which folds are held out and report cross-validation performance.
#
# ---
#
# ## 13. Why should we keep the same split when comparing features?
#
# Suppose:
#
# * MFCC mean+std gets 63%,
# * Mel summary gets 68%.
#
# That comparison is meaningful only if both methods are evaluated under the same split policy.
#
# If one feature set happens to receive an easier test set, the comparison is confounded.
#
# So:
#
# **Hold the split fixed while changing the representation.**
#
# This is experimental control.
#
# ---
#
# ## 14. Leakage can happen even after the split
#
# This is a major subtlety.
#
# Suppose we standardize each feature:
#
# $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
#
# Where do $\mu_j$ and $\sigma_j$ come from?
#
# They must come from the **training set only**.
#
# The slides write:
#
# $$z_j = \frac{x_j - \mu_{j,\text{train}}}{\sigma_{j,\text{train}}}$$
#
# ---
#
# ## 15. Why is computing the mean from the test set leakage?
#
# Suppose the test set happens to have unusually large values for MFCC coefficient 4.
#
# If you compute the normalization mean using all examples:
#
# $$\mu_4 = \text{mean}(\text{train} + \text{test})$$
#
# you have allowed information about the test distribution to affect the transformation used on the training data.
#
# Even though no labels leaked, test-set information leaked.
#
# That makes evaluation less honest.
#
# ---
#
# ## 16. Pipeline is more than convenience
#
# The deck uses
#
# ```python
# pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('clf', SVC(kernel='rbf'))
# ])
#
# ```
#
# then
#
# ```python
# pipe.fit(X_train, y_train)
#
# ```
#
# A scikit-learn `Pipeline` helps ensure that transformations are fitted within the training workflow.
#
# The conceptual sequence is:
#
# $X_{\text{train}}$ → fit scaler → transform → fit classifier.
#
# The test set only gets transformed using the already learned training statistics.
#
# ---
#
# ## 17. Why doesn't Random Forest usually need standardization?
#
# A tree splits on comparisons such as:
#
# $x_j < 3.7$
#
# If you linearly rescale that feature, the tree can simply learn another threshold.
#
# Distance-based and margin-based methods behave differently.
#
# For an RBF SVM, feature scales matter because distances like
#
# $\vert{}\mathbf{x}_i - \mathbf{x}_j\vert{}^2$
#
# matter directly.
#
# The deck therefore correctly says Random Forest generally does not require standard scaling, while SVM/logistic-type pipelines often benefit from it.
#
# ---
#
# ## 18. The 2D-to-1D problem
#
# From Lecture 21, an MFCC might have shape:
#
# 20 × 173.
#
# But classical scikit-learn models expect one vector per clip.
#
# So the deck presents three choices.
#
# ### Flatten
#
# 20 × 173 → 3460 features.
#
# ### Mean only
#
# 20 × 173 → 20 features.
#
# ### Mean + standard deviation
#
# 20 × 173 → 40 features.
#
# Each option hard-codes different assumptions.
#
# ---
#
# ## 19. What does flattening preserve?
#
# Flattening might take $C[b,t]$ and concatenate all entries into one long vector.
#
# This preserves frame order in the sense that frame 1 occupies one set of coordinates, frame 2 another, and so on.
#
# But there is a serious drawback:
#
# > A pattern occurring at frame 20 and the same pattern occurring at frame 40 become completely different coordinates.
#
# So flattening is sensitive to exact alignment.
#
# Imagine the same bark occurs:
#
# * at 0.3 seconds in one clip,
# * at 1.8 seconds in another.
#
# The feature vectors can be very different even though perceptually they are the same event.
#
# ---
#
# ## 20. What does taking the mean preserve and destroy?
#
# For MFCC coefficient $b$,
#
# $$\boxed{\bar{c}[b] = \frac{1}{T} \sum_{t} C[b,t]}$$
#
# as shown in the slides.
#
# This preserves approximately:
#
# > “How much of this MFCC characteristic occurred on average?”
#
# But loses:
#
# * when it occurred,
# * sequence order,
# * onset structure,
# * duration of local patterns.
#
# ---
#
# ## 21. Rising versus falling siren
#
# This is probably the best example in that section.
#
# Suppose one siren rises:
#
# ```text
# frequency
#    ^
#    |        /
#    |      /
#    |    /
#    |  /
#    +------------> time
#
# ```
#
# and another falls:
#
# ```text
# frequency
#    ^
#    |  \
#    |    \
#    |      \
#    |        \
#    +------------> time
#
# ```
#
# They may contain very similar overall frequency content.
#
# Their time-averaged MFCCs can therefore be similar.
#
# But their temporal trajectories are opposite.
#
# The deck explicitly uses rising and falling sirens to illustrate what averaging destroys.
#
# That becomes a natural motivation for CNNs or sequence models.
#
# ---
#
# ## 22. Why mean + standard deviation is a good baseline
#
# The deck suggests mean + std for each MFCC coefficient.
#
# If there are 20 MFCCs:
#
# 20 means + 20 stds = 40 features.
#
# The mean captures the typical spectral-envelope coordinate.
#
# The standard deviation adds:
#
# > “How much did that coefficient vary over the clip?”
#
# So two sounds with the same average but different levels of temporal variability may become distinguishable.
#
# Still, sequence order is gone.
#
# For example, these two sequences:
#
# $[1, 2, 3, 4]$ and $[4, 3, 2, 1]$
#
# have the same mean and same standard deviation.
#
# So no classifier using only those statistics can know which direction the sequence went.
#
# ---
#
# ## 23. Scikit-learn's API is intentionally boring
#
# That is a compliment.
#
# The slides show:
#
# ```python
# clf = RandomForestClassifier(...)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = clf.score(X_test, y_test)
#
# ```
#
# The pattern is consistent across many models.
#
# ### Instantiate
#
# Choose model family and hyperparameters.
#
# ### Fit
#
# Estimate parameters from training examples.
#
# ### Predict
#
# Generate predictions for unseen $X$.
#
# ### Evaluate
#
# Compare predictions against ground truth.
#
# This allows students to focus more on experimental design and less on API memorization.
#
# ---
#
# ## 24. What exactly does `.fit()` mean?
#
# This is worth asking.
#
# For different model families, `.fit()` means different mathematics.
#
# For Random Forest:
#
# > Build many decision trees from training data and random subsets of samples/features.
#
# For logistic regression:
#
# > Estimate coefficients minimizing a regularized classification loss.
#
# For SVM:
#
# > Find a separating boundary/margin determined by the training data.
#
# So `fit()` is a common interface, not one common algorithm.
#
# ---
#
# ## 25. Why Random Forest first?
#
# The deck is careful not to claim Random Forest is best.
#
# Instead, it treats it as a diagnostic baseline.
#
# Advantages:
#
# * works well with compact tabular features,
# * nonlinear decision boundaries,
# * no scaling usually required,
# * fast relative to deep networks,
# * good sanity check for whether the features contain class information.
#
# The key question it answers is:
#
# **Do these engineered features contain enough information to classify above chance?**
#
# If the Random Forest performs terribly, the problem may be:
#
# * feature extraction,
# * labeling,
# * preprocessing,
# * dataset quality,
#
# rather than lack of a deep network.
#
# ---
#
# ## 26. Why shouldn't you promise “Random Forest gets 60–65%”?
#
# Because performance depends on:
#
# * split,
# * dataset version,
# * features,
# * number of trees,
# * class distribution,
# * preprocessing,
# * random seed.
#
# The deck explicitly says not to promise a fixed accuracy ceiling.
#
# That's a good scientific habit.
#
# A baseline is meaningful because it is reproducible and honest, not because it reaches a preordained number.
#
# ---
#
# ## 27. What does a confusion matrix tell us?
#
# Accuracy compresses everything into one number.
#
# The confusion matrix tells us **which classes fail**.
#
# In the convention used on the slide:
#
# * rows = true classes
# * columns = predicted classes
# * diagonal = correct
# * off-diagonal = mistakes
#
# Suppose:
#
# | True \ Predicted | Dog | Siren | Engine |
# | --- | --- | --- | --- |
# | **Dog** | 80 | 5 | 15 |
# | **Siren** | 3 | 92 | 5 |
# | **Engine** | 20 | 4 | 76 |
#
# Overall accuracy might seem decent.
#
# But we immediately see:
#
# > dog and engine are being confused much more than dog and siren.
#
# That tells us where to investigate.
#
# ---
#
# ## 28. A good confusion-matrix question
#
# Suppose most errors are:
#
# air conditioner → engine idling.
#
# What could cause that?
#
# Good hypotheses:
#
# * both have relatively stationary low-frequency energy,
# * mean MFCCs may not preserve temporal signatures,
# * recordings may share background hum,
# * labels may be acoustically ambiguous.
#
# The correct ML workflow is not:
#
# > “Immediately use a bigger model.”
#
# It is:
#
# > Listen to the errors and inspect the representations.
#
# ---
#
# ## 29. Why accuracy alone can be misleading
#
# The lecture introduces precision, recall, and macro-F1 in addition to accuracy.
#
# Suppose 90% of clips are “engine” and 10% are “siren.”
#
# A terrible classifier predicts “engine” for everything.
#
# Accuracy: 90%.
# But siren recall: 0%.
#
# If your application is emergency-siren detection, that system is useless.
#
# So class-sensitive metrics matter.
#
# ---
#
# ## 30. Precision and recall intuitively
#
# For the class “siren”:
#
# ### Precision
#
# Of everything I predicted as siren, how much really was siren?
#
# $$\text{Precision} = \frac{TP}{TP+FP}$$
#
# ### Recall
#
# Of all actual sirens, how many did I detect?
#
# $$\text{Recall} = \frac{TP}{TP+FN}$$
#
# A nice applied distinction:
#
# For an alarm system, false negatives might be especially costly, so recall may matter greatly.
#
# ---
#
# ## 31. Why macro-F1?
#
# Macro averaging computes a metric independently for each class and then averages them equally.
#
# So a rare class gets the same class-level weight as a common class.
#
# That makes it useful when class imbalance exists and you want to avoid letting dominant classes control the score.
#
# The deck makes exactly this point.
#
# ---
#
# ## 32. Why split the metadata first?
#
# The live-demo slide does:
#
# ```python
# train = meta[meta['fold'].isin(range(1, 9))]
# valid = meta[meta['fold'].eq(9)]
# test  = meta[meta['fold'].eq(10)]
#
# ```
#
# and only then extracts features.
#
# That order is very good practice.
#
# Why?
#
# Because the dataset membership is defined before feature extraction and experimentation.
#
# A good answer to the instructor's question is:
#
# > Splitting metadata first makes the intended train/validation/test membership explicit and reduces the chance of accidentally shuffling correlated excerpts together after feature generation.
#
# It's also easier to audit.
#
# ---
#
# ## 33. Reproducibility is part of the experiment
#
# The deck asks students to freeze and record:
#
# * sample rate,
# * duration,
# * FFT size,
# * hop length,
# * Mel count,
# * MFCC count,
# * fold assignments,
# * random seed.
#
# Why?
#
# Suppose Student A gets 71% and Student B gets 64%.
#
# If they used different:
#
# * sample rates,
# * MFCC counts,
# * random splits,
# * random seeds,
#
# the comparison is nearly meaningless.
#
# A reproducible experiment lets someone answer:
#
# > “What exactly changed?”
#
# ---
#
# ## 34. Why set `random_state`?
#
# Many algorithms contain randomness.
#
# For example Random Forest may randomly:
#
# * bootstrap training examples,
# * choose feature subsets.
#
# If you set:
#
# ```python
# random_state=42
#
# ```
#
# then repeated runs become much more reproducible.
#
# This does **not** mean 42 is magically optimal.
#
# It means:
#
# > The experiment is repeatable.
#
# ---
#
# ## 35. Error analysis: listen to your mistakes
#
# This is one of the best slides in the lecture.
#
# The deck suggests finding high-confidence wrong predictions, listening to them, examining Mel-spectrograms, checking label ambiguity and artifacts, and comparing feature choices.
#
# The proposed engineering loop is:
#
# **Measure → Inspect → Hypothesize → Change one thing → Remeasure**
#
# That is exactly the mindset students should learn.
#
# ---
#
# ## 36. Example of useful error analysis
#
# Suppose the classifier predicts:
#
# **dog bark → children playing**
#
# Listen to the clip.
#
# Maybe you discover:
#
# * children are shouting loudly in the background,
# * the bark is only 100 ms long,
# * the four-second MFCC average is dominated by speech.
#
# Now you have a hypothesis:
#
# > Time averaging suppresses short transient events.
#
# A sensible next experiment might be:
#
# * retain the 2D Mel-spectrogram,
# * use max/percentile statistics,
# * use shorter analysis windows,
# * eventually use a CNN.
#
# That is much better than arbitrarily doubling `n_estimators`.
#
# ---
#
# ## 37. Why classical features may hit a ceiling
#
# The lecture carefully says representation may be the bottleneck.
#
# Mean MFCCs remove:
#
# * trajectories,
# * modulation,
# * temporal shape.
#
# Flattened MFCCs preserve more ordering but create:
#
# * thousands of coordinates,
# * alignment sensitivity.
#
# So the limitation may not be the Random Forest's intelligence.
#
# The model may simply never have received the information needed.
#
# ---
#
# ## 38. Why CNNs become attractive next
#
# A Mel-spectrogram is a 2D field:
#
# frequency × time.
#
# Suppose a dog bark has a characteristic local pattern.
#
# It may occur at:
#
# * 0.5 seconds,
# * 2 seconds,
# * 3 seconds.
#
# A CNN can use a learned local filter and detect approximately the same pattern at multiple positions.
#
# That gives a degree of translation tolerance.
#
# By contrast, flattening says:
#
# > Feature number 312 means one very specific frequency/time location.
#
# The slides therefore end the bridge with:
#
# **Preserve the picture, learn the filters.**
#
# ---
#
# ## 39. The concept-check answer
#
# The slide asks which evaluation is trustworthy.
#
# ### A
#
# Randomly split segmented clips, then standardize using all data.
#
# Bad for two reasons:
#
# 1. source leakage,
# 2. preprocessing leakage.
#
# ### B
#
# Use source-aware folds, fit preprocessing/model on training only, test once at the end.
#
# **Correct.**
#
# ### C
#
# Tune until test accuracy stops improving.
#
# That turns the test set into validation data.
#
# ### D
#
# Report only training accuracy.
#
# Training performance says little about unseen generalization.
#
# ---
#
# ## 40. A strong answer to the final exit ticket
#
# The lecture ends with:
#
# > **Why can a more accurate random clip split be a worse scientific result?**
#
# A weak answer:
#
# > Because random splitting can be bad.
#
# A good answer:
#
# > Because clips from the same recording can appear in both training and test sets, allowing the classifier to exploit source-specific information rather than generalize to new recordings.
#
# A very strong answer:
#
# > The apparent accuracy can increase because the independence assumption between training and test data has been violated. Correlated segments from the same physical recording leak source identity across the split, so the measured accuracy underestimates the difficulty of deployment on truly unseen sources.
#
# That is exactly the kind of answer I would want from a graduate student.
#
# ---
#
# ## 41. Some additional clicker questions I would use
#
# ### Question 1
#
# A `StandardScaler` is fit using train + test data, but no labels are used. Is this leakage?
#
# A. No, because labels were not used.
# B. Yes, because test distribution statistics influenced training preprocessing.
# C. Only for neural networks.
# D. Only if accuracy increases.
#
# **Answer: B**
#
# ---
#
# ### Question 2
#
# Two 4-second clips come from adjacent portions of one 10-second recording. Where should they go?
#
# A. Randomly to different splits
# B. Always into validation
# C. Same split/fold
# D. It doesn't matter if labels match
#
# **Answer: C**
#
# ---
#
# ### Question 3
#
# Which feature representation preserves the most temporal ordering?
#
# A. Mean MFCC
# B. Mean+std MFCC
# C. Flattened MFCC matrix
# D. Class label
#
# **Answer: C**, although it is alignment-sensitive.
#
# ---
#
# ### Question 4
#
# Why might mean+std outperform mean alone?
#
# A. It restores exact temporal ordering.
# B. Standard deviation captures temporal variability.
# C. It eliminates leakage.
# D. It doubles sample rate.
#
# **Answer: B**
#
# ---
#
# ### Question 5
#
# A model gets:
#
# * training accuracy = 99%
# * validation accuracy = 62%
#
# What is the likely concern?
#
# Good answer:
#
# > Overfitting, although one should also inspect whether the train/validation distributions differ or there is some preprocessing issue.
#
# ---
#
# ## 42. A good senior/graduate distinction
#
# For senior students, I would expect them to be comfortable with:
#
# * $X$ and $y$,
# * train/validation/test,
# * leakage,
# * fixed-length features,
# * `fit`, `predict`, `score`,
# * confusion matrices.
#
# For graduate students, I would additionally expect them to:
#
# * explain *why* source leakage biases performance,
# * explain why training-only preprocessing matters,
# * reason about invariances introduced by time averaging,
# * distinguish model limitation from representation limitation,
# * discuss why a lower honest score can be more informative than a higher leaked score,
# * propose controlled follow-up experiments based on error analysis.
#
# ---
#
# ## 43. The deepest message of the lecture
#
# The final slide says a trustworthy baseline is a **scientific instrument**.
#
# That is the right framing.
#
# A baseline is not there just to produce a number.
#
# It lets us ask controlled questions:
#
# * **Do the features contain useful class information?**
# * **Does a different representation improve generalization?**
# * **Which classes remain hard?**
# * **Are the errors caused by the model, the features, or the data?**
#
# And the entire lecture can be compressed into this chain:
#
# $$\text{Audio} \rightarrow \text{features} \rightarrow \underbrace{\text{honest split}}_{\text{critical}} \rightarrow \text{fit model} \rightarrow \text{held-out predictions} \rightarrow \text{metrics} \rightarrow \text{error analysis}$$
#
# The central habit I would want students to carry forward is:
#
# **Before asking "How accurate is my model?", ask "Was my experiment designed so that accuracy means anything?"**
#
# That question is arguably more important than the choice between Random Forest, SVM, and the CNN they will use next.
#
# -------------------------

# %%
