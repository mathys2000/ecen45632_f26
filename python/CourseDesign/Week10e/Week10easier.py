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
# ## Week 10, Lecture 19
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT (Medium):
#
# Below is a more detailed instructor-oriented explanation of Lecture 19, with emphasis on what is happening mathematically, what students should understand conceptually, where misconceptions usually occur, and what strong student answers would sound like.
#
# The main arc of the lecture is:
#
# **DFT as sampled DTFT $\rightarrow$ circular convolution $\rightarrow$ zero-padding for linear convolution $\rightarrow$ FFT-based block filtering $\rightarrow$ overlap-add and overlap-save $\rightarrow$ streaming DSP/ML systems.** 
#
# ## 1. What is the central idea of this lecture?
#
# The core question is:
#
# > How can an operation that is naturally defined for a whole sequence be implemented efficiently on an endless stream?
#
# Linear convolution is conceptually defined over complete sequences:
#
# $$
# y[n] =
# \sum_m x[m]h[n-m].
# $$
#
# That works nicely on paper.
#
# But real systems receive data continuously:
#
# * microphone samples,
# * vibration sensor data,
# * radio signals,
# * biomedical signals,
# * streaming audio,
# * sensor feeds before ML inference.
#
# We cannot wait for an infinite signal to end.
#
# So the lecture develops a way to perform exactly the same FIR filtering using finite blocks and FFTs.
#
# The important conceptual chain is:
#
# $$
# \text{convolution}
# \rightarrow
# \text{DFT multiplication}
# \rightarrow
# \text{circular convolution}
# \rightarrow
# \text{zero-padding}
# \rightarrow
# \text{linear convolution}
# \rightarrow
# \text{block processing}.
# $$
#
# Students should leave understanding not just how overlap-add or overlap-save works, but **why the bookkeeping is necessary**.
#
# ---
#
# # 2. What exactly is the DFT?
#
# For an $N$-sample sequence,
#
# $$
# x[0],x[1],\ldots,x[N-1],
# $$
#
# the $N$-point DFT is
#
# $$
# X[k] =
# \sum_{n=0}^{N-1}
# x[n]e^{-j2\pi kn/N},
# \qquad
# k=0,\ldots,N-1.
# $$
#
# A useful interpretation is that the DFT samples the DTFT of the finite record at
#
# $$
# \omega_k =
# \frac{2\pi k}{N}.
# $$
#
# So if
#
# $$
# X(e^{j\omega}) =
# \sum_{n=0}^{N-1}
# x[n]e^{-j\omega n},
# $$
#
# then
#
# $$
# X[k] =
# X(e^{j\omega})
# \bigg|_{\omega=2\pi k/N}.
# $$
#
# This is precisely the interpretation emphasized in the slides. 
#
# ---
#
# # 3. A useful mental picture: DTFT versus DFT
#
# Students often confuse the DTFT and DFT.
#
# A very useful classroom picture is:
#
# * the **DTFT** is a continuous function of frequency;
# * the **DFT** consists of $N$ equally spaced samples of that function.
#
# So imagine a smooth spectral curve.
#
# The DFT is a set of dots placed on that curve at
#
# $$
# \omega_k=\frac{2\pi k}{N}.
# $$
#
# The slides make this explicit: the smooth curve is the finite record's DTFT and the dots are its DFT samples. 
#
# A strong student answer to
#
# > How are the DFT and DTFT related?
#
# would be:
#
# > For an $N$-sample finite record, the $N$-point DFT is the DTFT evaluated at $N$ uniformly spaced frequencies $\omega_k=2\pi k/N$.
#
# ---
#
# # 4. What is the FFT, then?
#
# This distinction is very important.
#
# The DFT is the mathematical transform:
#
# $$
# X[k] =
# \sum_{n=0}^{N-1}
# x[n]e^{-j2\pi kn/N}.
# $$
#
# The FFT is an **algorithm for computing the DFT efficiently**.
#
# The FFT does not define a different transform.
#
# That means
#
# $$
# \text{FFT result} =
# \text{DFT result}
# $$
#
# apart from numerical roundoff.
#
# The difference is computational complexity.
#
# A direct DFT requires approximately
#
# $$
# O(N^2)
# $$
#
# operations.
#
# An FFT requires approximately
#
# $$
# O(N\log_2 N).
# $$
#
# The slides give the useful example that for $N=4096$,
#
# $$
# N^2 =
# 4096^2
# \approx
# 16.8\times10^6,
# $$
#
# whereas
#
# $$
# N\log_2N =
# 4096(12) =
# 49152.
# $$
#
# Constant factors matter in an actual implementation, but this illustrates why FFT-based convolution becomes attractive for long filters and repeated processing. 
#
# ---
#
# # 5. Good answer: "Why is the FFT faster?"
#
# A weak answer is:
#
# > Because the FFT uses fewer calculations.
#
# A stronger answer is:
#
# > A direct DFT independently evaluates each of $N$ output frequencies using approximately $N$ operations, giving $O(N^2)$ complexity. FFT algorithms exploit symmetry and repeated structure in the complex exponentials so the same DFT can be computed in approximately $O(N\log N)$ operations.
#
# That demonstrates understanding rather than memorizing the complexity classes.
#
# ---
#
# # 6. Zero-padding does not add information
#
# This connects directly with the previous spectral-estimation lectures.
#
# If you have only $N$ measured samples and append zeros before computing a larger FFT, you obtain more closely spaced samples of the same finite-record DTFT.
#
# You do not acquire new measurements.
#
# So:
#
# $$
# \text{larger zero-padded FFT}
# \rightarrow
# \text{denser frequency grid}
# $$
#
# but not
#
# $$
# \text{larger zero-padded FFT}
# \rightarrow
# \text{new signal information}.
# $$
#
# The slides explicitly warn against confusing those two ideas. 
#
# However, zero-padding serves a **different and essential purpose** later in this lecture: preventing circular-convolution wrap-around.
#
# That is an important distinction.
#
# ---
#
# # 7. Linear convolution
#
# For finite sequences $x[n]$ and $h[n]$,
#
# $$
# y[n] =
# x[n]*h[n] =
# \sum_m x[m]h[n-m].
# $$
#
# If
#
# $$
# L_x=\text{length of }x[n]
# $$
#
# and
#
# $$
# L_h=\text{length of }h[n],
# $$
#
# then the linear convolution has length
#
# $$
# L_y =
# L_x+L_h-1.
# $$
#
# That "-1" is worth having students explain rather than memorize.
#
# If $x[n]$ occupies indices
#
# $$
# 0,\ldots,L_x-1
# $$
#
# and $h[n]$ occupies
#
# $$
# 0,\ldots,L_h-1,
# $$
#
# then the last possible nonzero convolution sample occurs at
#
# $$
# (L_x-1)+(L_h-1) =
# L_x+L_h-2.
# $$
#
# Counting from zero gives
#
# $$
# L_x+L_h-1
# $$
#
# samples.
#
# ---
#
# # 8. Circular convolution
#
# The $N$-point circular convolution is
#
# $$
# y_N[n] =
# \sum_{m=0}^{N-1}
# x[m]
# h[(n-m)\bmod N].
# $$
#
# The important word is **modulo**.
#
# Indices that would normally extend beyond $N-1$ wrap back to the beginning.
#
# The slides emphasize that DFT-domain multiplication corresponds to circular convolution, not automatically linear convolution. 
#
# If
#
# $$
# Y[k]=X[k]H[k],
# $$
#
# then
#
# $$
# y[n] =
# \operatorname{IDFT}\{Y[k]\}
# $$
#
# is the $N$-point circular convolution of the corresponding $N$-point sequences.
#
# ---
#
# # 9. Why does the DFT produce circular convolution?
#
# This is one of the deeper conceptual points.
#
# The DFT naturally treats an $N$-sample sequence as one period of an $N$-periodic sequence.
#
# So when convolution causes a result to extend beyond index
#
# $$
# N-1,
# $$
#
# the periodic interpretation brings those values back to the start.
#
# That is why the operation wraps around.
#
# A useful phrase for students is:
#
# > The DFT lives naturally in a periodic world.
#
# Linear convolution does not wrap.
#
# Circular convolution does.
#
# ---
#
# # 10. Small circular-convolution example
#
# Suppose
#
# $$
# x[n]=[1,2,3]
# $$
#
# and
#
# $$
# h[n]=[1,1].
# $$
#
# The linear convolution is
#
# $$
# [1,3,5,3].
# $$
#
# Its length is
#
# $$
# 3+2-1=4.
# $$
#
# Now suppose we try to perform a 3-point DFT convolution.
#
# The linear convolution has four samples, but the circular result has room for only three.
#
# The final sample wraps back onto the beginning:
#
# $$
# [1,3,5,3]
# $$
#
# becomes roughly
#
# $$
# [1+3,3,5] =
# [4,3,5].
# $$
#
# That extra contribution at the beginning is **time aliasing** or circular wrap-around.
#
# This is the error the slides illustrate. 
#
# ---
#
# # 11. Circular convolution is not "wrong"
#
# This is worth emphasizing.
#
# The slides correctly state:
#
# > Circular is not "wrong"—it is a different operation.
#
# That is important because students sometimes learn:
#
# > Circular convolution is the bad version of convolution.
#
# That is misleading.
#
# Circular convolution is the natural convolution operation for periodic sequences and for DFT-domain algebra.
#
# The problem occurs only when we **wanted linear convolution but accidentally computed circular convolution**.
#
# ---
#
# # 12. How zero-padding fixes the problem
#
# Suppose $x[n]$ has length $L_x$ and $h[n]$ has length $L_h$.
#
# The linear convolution requires
#
# $$
# L_x+L_h-1
# $$
#
# samples.
#
# Therefore choose the FFT size so that
#
# $$
# N_{\text{FFT}}
# \geq
# L_x+L_h-1.
# $$
#
# Then:
#
# 1. pad $x[n]$ with zeros to length $N_{\text{FFT}}$;
# 2. pad $h[n]$ with zeros to length $N_{\text{FFT}}$;
# 3. compute both FFTs;
# 4. multiply:
#
# $$
# Y[k]=X[k]H[k];
# $$
#
# 5. compute the IFFT.
#
# Now the circular convolution is long enough that nothing wraps around before the true linear convolution finishes.
#
# Therefore
#
# $$
# y[n] =
# x[n]*h[n]
# $$
#
# for the first
#
# $$
# L_x+L_h-1
# $$
#
# samples. 
#
# ---
#
# # 13. Important distinction: two reasons for zero-padding
#
# This would make a good class discussion because students have just encountered zero-padding in spectral estimation.
#
# ### Zero-padding for spectral display
#
# Produces more frequency samples.
#
# It does not create additional time-domain information.
#
# ### Zero-padding for FFT convolution
#
# Prevents time-domain circular wrap-around.
#
# Here zero-padding is **mathematically necessary** if you want DFT multiplication to reproduce linear convolution.
#
# Same operation, completely different purpose.
#
# That distinction is highly worth emphasizing.
#
# ---
#
# # 14. Concept check from the slides
#
# The lecture asks:
#
# > $x[n]$ has length 80 and $h[n]$ has length 33. What is the minimum FFT size for one-shot FFT-based linear convolution?
#
# The required size is
#
# $$
# N_{\text{FFT}} \geq
# 80+33-1 =
# 112.
# $$
#
# So the minimum mathematical value is
#
# $$
# N_{\text{FFT}}=112.
# $$
#
# An implementation might instead choose
#
# $$
# N_{\text{FFT}}=128
# $$
#
# because some FFT implementations are particularly efficient at convenient factorizations. 
#
# ---
#
# # 15. A strong answer to that concept check
#
# > The linear convolution has $80+33-1=112$ output samples. The FFT length must therefore be at least 112 so that the entire linear convolution fits before circular wrap-around occurs. A larger value such as 128 is also mathematically valid and may be computationally faster.
#
# That last sentence is important because it separates:
#
# * mathematical minimum,
# * engineering choice.
#
# ---
#
# # 16. Why not just FFT the entire input?
#
# Suppose a microphone produces samples continuously.
#
# There is no complete finite sequence called "the microphone signal."
#
# It keeps arriving.
#
# The same is true for:
#
# * radar,
# * radio,
# * vibration monitoring,
# * medical monitoring,
# * live audio.
#
# Therefore we need to divide the input into finite blocks.
#
# The slides emphasize that each block must nevertheless produce exactly the same output as ordinary linear FIR filtering. 
#
# That requirement is crucial:
#
# > Block processing is an implementation strategy, not a different filter.
#
# If properly implemented,
#
# $$
# y_{\text{FFT blocks}}[n]
# \approx
# y_{\text{direct FIR}}[n]
# $$
#
# to floating-point precision.
#
# ---
#
# # 17. Why block filtering is not trivial
#
# Suppose we arbitrarily divide the input into blocks and independently filter each block.
#
# That fails because an FIR filter has memory.
#
# For an FIR filter of length $L_h$,
#
# $$
# y[n] =
# \sum_{m=0}^{L_h-1}
# h[m]x[n-m].
# $$
#
# So output sample $y[n]$ depends on the current input sample and up to
#
# $$
# L_h-1
# $$
#
# previous input samples.
#
# At a block boundary, some required input samples belong to the previous block.
#
# So simply treating every block as independent creates boundary errors.
#
# Overlap-add and overlap-save are two mathematically correct ways of handling this filter memory.
#
# ---
#
# # 18. Overlap-add: the basic idea
#
# In overlap-add, split the incoming signal into **non-overlapping input blocks** of length $B$:
#
# $$
# x_0[n],x_1[n],x_2[n],\ldots
# $$
#
# For each block:
#
# 1. zero-pad the block;
# 2. zero-pad $h[n]$;
# 3. compute FFTs;
# 4. multiply in frequency;
# 5. IFFT back to time.
#
# The FFT size must satisfy
#
# $$
# N_{\text{FFT}}
# \geq
# B+L_h-1.
# $$
#
# Each block therefore produces
#
# $$
# B+L_h-1
# $$
#
# linear-convolution samples. 
#
# ---
#
# # 19. Why outputs overlap in overlap-add
#
# Suppose
#
# $$
# B=4
# $$
#
# and
#
# $$
# L_h=3.
# $$
#
# Each four-sample input block produces
#
# $$
# 4+3-1=6
# $$
#
# output samples.
#
# But the next input block begins only four samples later.
#
# Therefore the first block's output extends two samples into the region occupied by the second block's output.
#
# Those two overlapping values represent legitimate contributions from both input blocks.
#
# They must be **added**.
#
# That is the origin of the name:
#
# > overlap-add.
#
# The input blocks do not overlap.
#
# The **output convolutions** overlap.
#
# ---
#
# # 20. A more mathematical view of overlap-add
#
# Suppose we decompose the signal into shifted blocks:
#
# $$
# x[n] =
# \sum_r x_r[n-rB].
# $$
#
# Because convolution is linear,
#
# $$
# x[n]*h[n] =
# \left(
# \sum_r x_r[n-rB]
# \right)*h[n].
# $$
#
# By linearity,
#
# $$
# y[n] =
# \sum_r
# \left(
# x_r[n-rB]*h[n]
# \right).
# $$
#
# So we may independently convolve every block with the filter and then add all shifted block outputs.
#
# That is exactly what overlap-add is doing.
#
# This is useful because it shows that overlap-add is not a computational trick with mysterious corrections.
#
# It follows directly from linearity.
#
# ---
#
# # 21. Why the overlap length is $L_h-1$
#
# An input block of length $B$ produces an output of length
#
# $$
# B+L_h-1.
# $$
#
# The next block begins $B$ samples later.
#
# Therefore the excess tail is
#
# $$
# (B+L_h-1)-B =
# L_h-1.
# $$
#
# Hence adjacent outputs overlap by exactly
#
# $$
# L_h-1
# $$
#
# samples.
#
# That relationship is very useful for students to derive.
#
# ---
#
# # 22. A good overlap-add student answer
#
# Question:
#
# > Why do we need to add neighboring block outputs?
#
# Good answer:
#
# > Because an FIR filter has memory. A block of $B$ input samples produces $B+L_h-1$ output samples, so its response extends $L_h-1$ samples beyond the end of the input block. The next block also contributes to those output times. By linearity, both contributions must be added.
#
# That is significantly better than:
#
# > Because that's what overlap-add does.
#
# ---
#
# # 23. Why verify overlap-add against direct convolution?
#
# The slides state that the direct FIR output and overlap-add result should agree to numerical precision and suggest treating this as a coding unit test. 
#
# That is excellent engineering practice.
#
# A student implementation should compute
#
# $$
# y_{\text{direct}}[n]
# $$
#
# and
#
# $$
# y_{\text{OLA}}[n]
# $$
#
# and examine something such as
#
# $$
# e[n] =
# y_{\text{direct}}[n] -
# y_{\text{OLA}}[n].
# $$
#
# Then check
#
# $$
# \max_n |e[n]|.
# $$
#
# For a correct floating-point implementation, the error should be very small.
#
# This teaches an important professional habit:
#
# > Fast code should be checked against a simple reference implementation.
#
# ---
#
# # 24. Overlap-save takes the opposite perspective
#
# Overlap-save handles the block boundaries differently.
#
# Instead of independent input blocks whose **outputs overlap**, overlap-save deliberately creates **overlapping input blocks**.
#
# For an FIR of length $L_h$, each block contains
#
# $$
# L_h-1
# $$
#
# samples inherited from the previous block.
#
# Then we perform an $N_{\text{FFT}}$-point circular convolution.
#
# Some output samples are corrupted by wrap-around.
#
# Specifically, discard the first
#
# $$
# L_h-1
# $$
#
# samples.
#
# Keep
#
# $$
# N_{\text{FFT}}-L_h+1
# $$
#
# valid output samples. 
#
# That is overlap-save.
#
# ---
#
# # 25. Why are the first $L_h-1$ OLS samples corrupted?
#
# The circular convolution assumes the current block is periodic.
#
# At the beginning of the block, convolution requires samples from before the block.
#
# Circular convolution supplies those samples by wrapping values from the end of the block.
#
# Those are not the correct historical samples for the linear convolution.
#
# Therefore the beginning is contaminated by time aliasing.
#
# By providing the preceding
#
# $$
# L_h-1
# $$
#
# real samples as overlap and then throwing away the contaminated outputs, the remaining samples are exactly the desired linear-convolution samples.
#
# ---
#
# # 26. Why the name "overlap-save"?
#
# Input blocks **overlap** by
#
# $$
# L_h-1
# $$
#
# samples.
#
# After each circular convolution, we **save** only the uncontaminated part:
#
# $$
# N_{\text{FFT}}-L_h+1
# $$
#
# samples.
#
# Hence:
#
# > overlap-save.
#
# The slides give the clean contrast:
#
# * OLA overlaps and adds outputs.
# * OLS overlaps inputs and discards corrupted outputs. 
#
# That is probably the simplest high-level comparison students should remember.
#
# ---
#
# # 27. A numerical overlap-save example
#
# Suppose
#
# $$
# N_{\text{FFT}}=1024
# $$
#
# and
#
# $$
# L_h=129.
# $$
#
# The input overlap must be
#
# $$
# L_h-1 =
# 128.
# $$
#
# Therefore the number of new samples processed each block is
#
# $$
# 1024-128 =
# 896.
# $$
#
# Equivalently,
#
# $$
# N_{\text{valid}} =
# N_{\text{FFT}}-L_h+1
# $$
#
# so
#
# $$
# N_{\text{valid}} =
# 1024-129+1 =
# 896.
# $$
#
# Thus each FFT block contains:
#
# * 128 old input samples,
# * 896 new input samples.
#
# After the IFFT:
#
# * discard 128 corrupted samples,
# * keep 896 valid output samples.
#
# That arithmetic makes the method much more concrete.
#
# ---
#
# # 28. OLA versus OLS
#
# The slides provide an engineering comparison. 
#
# ### Overlap-add
#
# Input:
#
# * non-overlapping blocks.
#
# Output:
#
# * overlapping block responses.
#
# Action:
#
# * add overlapping output tails.
#
# Requires:
#
# * an output accumulation buffer.
#
# ### Overlap-save
#
# Input:
#
# * overlapping blocks.
#
# Output:
#
# * circular-convolution blocks.
#
# Action:
#
# * discard the first $L_h-1$ samples.
#
# Requires:
#
# * preserving input history.
#
# ### Both
#
# Both ultimately compute the same desired linear FIR output.
#
# Both can reuse
#
# $$
# H[k] =
# \operatorname{FFT}\{h[n]\}
# $$
#
# for every block.
#
# That reuse is important.
#
# The filter does not usually change from one block to the next, so there is no reason to repeatedly FFT the impulse response.
#
# ---
#
# # 29. A good OLA versus OLS exam or clicker question
#
# > Which statement is correct?
#
# A. OLA overlaps input blocks and OLS overlaps output blocks.
# B. OLA adds overlapping output contributions; OLS discards contaminated output samples.
# C. OLA computes linear convolution while OLS computes only circular convolution.
# D. OLS requires no overlap.
#
# Correct answer:
#
# **B.**
#
# A strong explanation would be:
#
# > Both methods use FFT-based circular convolution internally but arrange the blocks so the retained final result equals linear convolution. OLA uses independent input blocks and adds overlapping output tails. OLS overlaps the inputs and discards the time-aliased beginning of each output block.
#
# ---
#
# # 30. Why FFT convolution can outperform direct FIR convolution
#
# For a direct FIR filter of length $L_h$, approximately $L_h$ multiplications are needed per output sample.
#
# For a long signal, the computational cost grows roughly like
#
# $$
# O(NL_h).
# $$
#
# FFT convolution instead processes blocks using operations whose cost behaves roughly like
#
# $$
# O(N_{\text{FFT}}\log N_{\text{FFT}}).
# $$
#
# For a sufficiently long $L_h$, the FFT method can therefore be dramatically more efficient.
#
# However, for a very short FIR filter, direct convolution may still be faster because FFTs have overhead.
#
# So:
#
# > FFT convolution is not automatically faster for every filter.
#
# That is a useful engineering nuance.
#
# ---
#
# # 31. Why not make the FFT enormous?
#
# At first glance, students may reason:
#
# > Large FFTs are computationally efficient, so let's make $N_{\text{FFT}}$ enormous.
#
# But a streaming system has latency.
#
# Suppose the sample rate is
#
# $$
# F_s=48000\text{ Hz}.
# $$
#
# If we wait for
#
# $$
# N=48000
# $$
#
# samples before processing, that represents approximately
#
# $$
# \frac{48000}{48000} =
# 1\text{ second}
# $$
#
# of data.
#
# That could be unacceptable for:
#
# * hearing aids,
# * active noise cancellation,
# * interactive audio,
# * control systems,
# * real-time inference.
#
# The slides emphasize exactly this throughput-versus-latency trade-off. 
#
# ---
#
# # 32. A useful latency calculation
#
# Suppose
#
# $$
# F_s=48000\text{ samples/s}
# $$
#
# and the block contains
#
# $$
# B=480
# $$
#
# new samples.
#
# The time required just to collect that block is
#
# $$
# T_B =
# \frac{480}{48000} =
# 0.01\text{ s} =
# 10\text{ ms}.
# $$
#
# So before computation even begins, the buffering contributes roughly 10 ms of latency.
#
# If instead
#
# $$
# B=4800,
# $$
#
# then
#
# $$
# T_B =
# \frac{4800}{48000} =
# 0.1\text{ s} =
# 100\text{ ms}.
# $$
#
# A larger block may reduce computational cost per sample but severely worsen responsiveness.
#
# ---
#
# # 33. Good answer to the slide's 10-ms latency question
#
# The lecture asks essentially:
#
# > If a classifier requires 10-ms latency, would a huge FFT block still be acceptable just because it is computationally efficient?
#
# Good answer:
#
# > Not necessarily. A large FFT block requires waiting for many samples to accumulate, which creates buffering latency. The best block size must satisfy the end-to-end latency constraint as well as computational efficiency.
#
# An excellent answer would also mention:
#
# > Algorithmic efficiency per sample is only one design objective. Buffering, memory movement, feature extraction, and model inference all contribute to the total latency budget.
#
# ---
#
# # 34. The filter length matters too
#
# Suppose the FIR filter is long.
#
# In OLA the FFT must satisfy
#
# $$
# N_{\text{FFT}}
# \geq
# B+L_h-1.
# $$
#
# Therefore a longer $L_h$ forces either:
#
# * a larger FFT,
# * or a smaller number of new samples per FFT.
#
# Similarly, for overlap-save,
#
# $$
# B_{\text{new}} =
# N_{\text{FFT}}-L_h+1.
# $$
#
# If $L_h$ gets large while $N_{\text{FFT}}$ stays fixed, fewer new samples are processed per block.
#
# So filter length influences computational efficiency directly.
#
# ---
#
# # 35. An overlap-save efficiency example
#
# Suppose
#
# $$
# N_{\text{FFT}}=1024.
# $$
#
# ### Filter A
#
# $$
# L_h=65.
# $$
#
# Then
#
# $$
# B_{\text{new}} =
# 1024-65+1 =
# 960.
# $$
#
# Almost all the FFT block contains new samples.
#
# ### Filter B
#
# $$
# L_h=513.
# $$
#
# Then
#
# $$
# B_{\text{new}} =
# 1024-513+1 =
# 512.
# $$
#
# Now only half the block contains new samples.
#
# The remaining half is overlap.
#
# So the same FFT size is much less efficient for the longer filter.
#
# ---
#
# # 36. Why powers of two are common but not mathematically required
#
# The condition might say
#
# $$
# N_{\text{FFT}}
# \geq112.
# $$
#
# That does **not** mean the FFT size mathematically must be 128.
#
# An FFT can be computed for many different lengths.
#
# The engineering choice of
#
# $$
# N_{\text{FFT}}=128
# $$
#
# may be attractive because FFT libraries often have efficient algorithms for lengths with favorable factorization.
#
# Likewise, modern FFT libraries may perform very well for sizes such as
#
# $$
# 3\times2^k,
# $$
#
# $$
# 5\times2^k,
# $$
#
# or other composite lengths.
#
# So a useful distinction is:
#
# > Required length comes from convolution mathematics.
#
# > Chosen FFT length comes from implementation performance.
#
# The slides make this distinction in the 112-versus-128 example. 
#
# ---
#
# # 37. Why measured performance can beat asymptotic theory
#
# The slides correctly note that block size should be chosen using actual platform performance rather than asymptotic complexity alone. 
#
# Two implementations with the same big-$O$ complexity may behave very differently because of:
#
# * cache sizes,
# * memory bandwidth,
# * SIMD/vector instructions,
# * FFT library implementation,
# * GPU transfer overhead,
# * embedded-device memory constraints.
#
# For example, an FFT size of 2048 might actually run more efficiently than 1500 on one platform.
#
# But on another system, the additional buffering or memory movement might dominate.
#
# This is where pure DSP theory becomes systems engineering.
#
# ---
#
# # 38. Why this matters for machine learning
#
# The lecture connects fast convolution to the preprocessing chain
#
# $$
# \text{stream}
# \rightarrow
# \text{fast FIR/resampling}
# \rightarrow
# \text{framing}
# \rightarrow
# \text{features}
# \rightarrow
# \text{model}.
# $$
#
#
#
# This is a very useful connection.
#
# A neural network or classical ML classifier often does not consume the raw sensor signal directly.
#
# The stream may first undergo:
#
# * anti-alias filtering,
# * denoising,
# * equalization,
# * resampling,
# * bandpass filtering,
# * framing,
# * spectral analysis.
#
# Fast convolution lets those operations run continuously and efficiently.
#
# ---
#
# # 39. Why block boundaries must not leak into ML features
#
# Suppose an FFT filtering implementation is incorrect at block boundaries.
#
# Then every block might contain a small discontinuity or transient.
#
# A downstream spectrogram or feature extractor could interpret those artifacts as actual signal events.
#
# Then the ML model might learn:
#
# > block-processing artifacts
#
# instead of
#
# > meaningful physical characteristics of the signal.
#
# That is why the slides emphasize that the block implementation should not change the underlying filter response or create boundary artifacts. 
#
# This is an excellent connection between DSP correctness and ML reliability.
#
# ---
#
# # 40. A concrete ML example
#
# Suppose you have a vibration-monitoring classifier.
#
# Pipeline:
#
# $$
# \text{accelerometer}
# \rightarrow
# \text{bandpass FIR}
# \rightarrow
# \text{STFT}
# \rightarrow
# \text{features}
# \rightarrow
# \text{fault classifier}.
# $$
#
# If the FIR has
#
# $$
# L_h=1001,
# $$
#
# direct convolution may be computationally expensive.
#
# You therefore use overlap-save FFT filtering.
#
# If implemented correctly, the classifier sees exactly the same filtered signal it would have received from direct convolution.
#
# If implemented incorrectly, every block boundary may introduce a broadband transient.
#
# The classifier might then see false high-frequency features.
#
# That is a very practical reason for insisting on numerical equivalence.
#
# ---
#
# # 41. A good systems-design question
#
# Ask students:
#
# > Two implementations produce almost identical throughput. One uses 4096-sample blocks and one uses 512-sample blocks. Which is better?
#
# The correct answer is:
#
# > There is not enough information.
#
# Students should ask about:
#
# * sampling rate,
# * latency requirement,
# * filter length,
# * processor,
# * memory limits,
# * power consumption,
# * downstream frame size.
#
# That encourages them to stop thinking that there is one universally optimal FFT size.
#
# ---
#
# # 42. What should students understand about direct convolution versus FFT convolution?
#
# A good summary is:
#
# ### Direct convolution
#
# Conceptually simple.
#
# Cost for FIR of length $L_h$:
#
# $$
# O(L_h)
# $$
#
# per output sample.
#
# Attractive for short filters.
#
# ### FFT convolution
#
# Uses
#
# $$
# \operatorname{IFFT}
# \left\{
# \operatorname{FFT}\{x\}
# \operatorname{FFT}\{h\}
# \right\}.
# $$
#
# Potentially much faster for long filters.
#
# Requires:
#
# * zero-padding,
# * block bookkeeping,
# * management of overlap,
# * FFT/IFFT overhead.
#
# So the engineering choice depends strongly on filter length and system constraints.
#
# ---
#
# # 43. The most important misconceptions to watch for
#
# ### Misconception 1
#
# > "The FFT is a different transform than the DFT."
#
# Better:
#
# > The FFT is an efficient algorithm for computing the DFT.
#
# ---
#
# ### Misconception 2
#
# > "Multiplying DFTs gives ordinary linear convolution."
#
# Better:
#
# > Multiplying $N$-point DFTs gives $N$-point circular convolution. Adequate zero-padding makes that circular result equal the desired linear convolution.
#
# ---
#
# ### Misconception 3
#
# > "Circular convolution is an approximation."
#
# Better:
#
# > Circular convolution is an exact mathematical operation. It becomes an error only when linear convolution was intended.
#
# ---
#
# ### Misconception 4
#
# > "Zero-padding always just interpolates the spectrum."
#
# Better:
#
# > For spectral visualization, zero-padding provides denser DTFT sampling. For FFT convolution, sufficient zero-padding prevents time-domain aliasing and is required to recover linear convolution.
#
# ---
#
# ### Misconception 5
#
# > "Overlap-add means input blocks overlap."
#
# Better:
#
# > OLA normally uses non-overlapping input blocks. The **output responses** overlap and are added.
#
# ---
#
# ### Misconception 6
#
# > "Overlap-save adds overlapping outputs."
#
# Better:
#
# > OLS overlaps the **inputs** and discards the corrupted first $L_h-1$ output samples.
#
# ---
#
# ### Misconception 7
#
# > "A larger FFT is always better."
#
# Better:
#
# > Larger FFTs may improve computational efficiency per sample, but they can increase buffering latency and memory requirements.
#
# ---
#
# # 44. Strong explanation of OLA in one paragraph
#
# A good student should eventually be able to say something like:
#
# > In overlap-add, the input stream is divided into non-overlapping blocks of $B$ samples. Each block is zero-padded so that $N_{\text{FFT}}\geq B+L_h-1$, then filtered using FFT multiplication. Each block produces $B+L_h-1$ valid linear-convolution samples. Since adjacent input blocks begin only $B$ samples apart, their output responses overlap by $L_h-1$ samples. Those overlapping contributions are added to produce the same output as direct linear FIR filtering.
#
# That demonstrates real understanding.
#
# ---
#
# # 45. Strong explanation of OLS in one paragraph
#
# Likewise:
#
# > In overlap-save, every FFT input block contains $L_h-1$ samples from the preceding block along with new input samples. The block is circularly convolved with the FIR using FFT multiplication. The first $L_h-1$ output samples contain time-aliasing contamination and are discarded. The remaining $N_{\text{FFT}}-L_h+1$ samples are valid linear-convolution outputs and are concatenated across blocks.
#
# That is an excellent senior/graduate-level answer.
#
# ---
#
# # 46. A useful side-by-side student summary
#
# Students could keep this mental table:
#
# | Property          | Overlap-Add                      | Overlap-Save           |
# | ----------------- | -------------------------------- | ---------------------- |
# | Input blocks      | Non-overlapping                  | Overlap by $L_h-1$     |
# | FFT result        | Linear convolution after padding | Circular convolution   |
# | Invalid output    | None within padded block         | First $L_h-1$ samples  |
# | Block handling    | Add output tails                 | Discard output prefix  |
# | Valid new outputs | $B$ after assembly               | $N_{\text{FFT}}-L_h+1$ |
# | Filter FFT        | Reused                           | Reused                 |
#
# That comparison reinforces the conceptual difference without treating them as unrelated algorithms.
#
# ---
#
# # 47. Good answer to the lecture's core question
#
# The core question is:
#
# > How can an operation that is naturally defined for a whole sequence be implemented efficiently on an endless stream?
#
# A strong answer would be:
#
# > Divide the incoming signal into finite blocks and use FFT-domain multiplication to implement the FIR efficiently. Because DFT multiplication produces circular rather than linear convolution, the blocks must be arranged carefully. Overlap-add zero-pads independent input blocks and adds their overlapping linear-convolution outputs, while overlap-save overlaps the input blocks and discards the time-aliased portion of each circular-convolution output. Both reproduce the same result as continuous linear FIR filtering.
#
# That answer captures essentially the whole lecture.
#
# ---
#
# # 48. Good answer to the final exit ticket
#
# The lecture ends with:
#
# > Why is $N_{\text{FFT}}\geq B+L_h-1$ required in overlap-add, and what fails if $N_{\text{FFT}}$ is smaller? 
#
# ### Minimum acceptable answer
#
# > Because a $B$-sample input block convolved with an $L_h$-sample FIR produces $B+L_h-1$ output samples. The FFT must be at least that long.
#
# ### Good answer
#
# > In overlap-add, each $B$-sample block must undergo a complete linear convolution with the $L_h$-sample filter. That convolution has length $B+L_h-1$. Therefore the FFT must have at least that many points. Otherwise the DFT multiplication produces circular wrap-around.
#
# ### Excellent answer
#
# > DFT multiplication always represents circular convolution over the chosen FFT length. A block of length $B$ convolved linearly with an FIR of length $L_h$ occupies $B+L_h-1$ samples. Choosing $N_{\text{FFT}}\geq B+L_h-1$ ensures that this entire result fits before any circular wrap-around occurs. If $N_{\text{FFT}}$ is smaller, the convolution tail aliases back onto the beginning of the block, corrupting samples that overlap-add expects to be valid linear-convolution results.
#
# That last answer demonstrates understanding of both the **length rule** and the **reason behind it**.
#
# ## The conceptual thread I would emphasize throughout the 75 minutes
#
# I would keep returning to four distinctions:
#
# $$
# \boxed{\text{DFT} \neq \text{FFT}}
# $$
#
# The DFT is the transform; the FFT is the fast algorithm.
#
# $$
# \boxed{\text{DFT multiplication} \rightarrow \text{circular convolution}}
# $$
#
# not automatically linear convolution.
#
# $$
# \boxed{\text{zero-padding} \rightarrow \text{prevent wrap-around}}
# $$
#
# when used for convolution.
#
# And finally,
#
# $$
# \boxed{\text{OLA/OLS} \rightarrow \text{same linear FIR result, different bookkeeping}}
# $$
#
# The deeper engineering lesson is that **fast convolution is not changing the DSP operation—it is changing how we implement it efficiently**.
#
# For your DSP/ML course, I would especially emphasize the last part: the block size becomes a genuine systems-design variable because computational efficiency, memory, throughput, and latency all interact. That makes this lecture a useful bridge from classical transform theory to the real-time preprocessing pipelines the students will encounter in later ML work.
#
# -------------------
#

# %% [markdown]
# ## Week 10, Lecture 20
#
# Explain in more detail what's going on here and what good answers and/or examples are.  For all mathematical equations, use standard single dollar signs $...$ for inline math and double dollar signs $$...$$ for block/display math. Do not use \( ... \) or \[ ... \]. Output the raw text without escaping the dollar signs.
#
# Chat GPT (Medium):
#
# Below is a more detailed instructor-oriented explanation of Lecture 20. The main thread is:
#
# **global spectrum loses timing information $\rightarrow$ STFT restores local time context $\rightarrow$ window length creates the time-frequency resolution trade-off $\rightarrow$ spectrograms expose interpretable structure $\rightarrow$ spectrograms become structured inputs for CNNs.** 
#
# ## 1. Why is a single global spectrum sometimes not enough?
#
# A Fourier transform answers:
#
# > What frequencies are present in the entire record?
#
# But for a non-stationary signal, the more important question may be:
#
# > What frequencies are present at a particular time?
#
# That distinction matters for signals whose spectral content changes:
#
# * speech,
# * chirps,
# * radar returns,
# * rotating machinery,
# * ECG/EEG,
# * transients,
# * impact events.
#
# The slides emphasize that a global spectrum may correctly show that many frequencies occurred while completely losing the fact that those frequencies occurred at different times. 
#
# ### Simple example
#
# Consider two signals.
#
# Signal A:
#
# * 500 Hz for the first half,
# * 1000 Hz for the second half.
#
# Signal B:
#
# * 1000 Hz for the first half,
# * 500 Hz for the second half.
#
# Their global magnitude spectra can be very similar.
#
# Both contain strong components at 500 Hz and 1000 Hz.
#
# But their temporal structures are different.
#
# A global spectrum cannot tell you which tone occurred first.
#
# That is exactly the type of information the STFT is designed to preserve.
#
# ---
#
# ## 2. What does the STFT actually do?
#
# The STFT takes a short section of the signal, computes its local spectrum, moves forward in time, and repeats.
#
# One form is
#
# $$
# X(r,\omega) =
# \sum_m
# x[m]w[m-rR]e^{-j\omega m}.
# $$
#
# Here:
#
# * $r$ is the frame index,
# * $R$ is the hop size,
# * $w[n]$ is the analysis window.
#
# Each value of $r$ gives one local Fourier transform. 
#
# A useful interpretation is:
#
# $$
# \boxed{
# \text{STFT} =
# \text{window} +
# \text{Fourier transform} +
# \text{repeat over time}
# }
# $$
#
# ---
#
# ## 3. A very intuitive way to explain it
#
# Imagine placing a short window over a long waveform.
#
# At the first position:
#
# 1. keep only the local signal;
# 2. calculate its spectrum.
#
# Then shift the window slightly.
#
# Again:
#
# 1. keep the new local section;
# 2. calculate its spectrum.
#
# Repeat this across the entire signal.
#
# The result is not one spectrum.
#
# It is a **sequence of spectra**.
#
# If we stack those spectra next to each other, we obtain a time-frequency representation.
#
# ---
#
# ## 4. From frames to an STFT matrix
#
# Suppose each frame produces $F$ frequency bins and the signal produces $T$ frames.
#
# Then the STFT can be regarded as a matrix roughly shaped like
#
# $$
# F\times T.
# $$
#
# One axis represents frequency.
#
# The other represents time.
#
# The slides describe the processing chain as:
#
# * frame the signal,
# * apply a window,
# * take an FFT,
# * stack the spectra. 
#
# This is important because it leads directly into the ML interpretation.
#
# A waveform is essentially 1-D.
#
# The STFT converts it into structured 2-D data.
#
# ---
#
# ## 5. Why a chirp is such a good demonstration
#
# Suppose a sinusoid has a frequency that increases with time.
#
# For example,
#
# $$
# x(t) =
# \cos\left(2\pi
# \left(
# f_0t+\frac{k}{2}t^2
# \right)
# \right).
# $$
#
# Its instantaneous frequency is
#
# $$
# f(t) =
# f_0+kt.
# $$
#
# A conventional spectrum may show energy across a range of frequencies.
#
# But it does not clearly show the temporal ordering.
#
# The STFT produces a ridge that moves upward with time.
#
# So a spectrogram of an upward chirp contains a sloped bright line.
#
# The slides identify this directly: horizontal position represents time, vertical position represents frequency, and brightness or color represents local magnitude or power. 
#
# ---
#
# # 6. How to read a spectrogram
#
# A very useful classroom rule is:
#
# > Ask what geometric structure in the spectrogram corresponds to what physical behavior in the signal.
#
# The lecture gives several important cases. 
#
# ### Horizontal line
#
# A sustained tone.
#
# For example:
#
# $$
# x(t)=\cos(2\pi1000t).
# $$
#
# The frequency remains constant, so the ridge stays at the same vertical location.
#
# ---
#
# ### Sloped line
#
# A chirp or changing pitch.
#
# Frequency increases or decreases as time progresses.
#
# ---
#
# ### Vertical line
#
# A brief broadband transient.
#
# A very short event is localized in time but spreads across many frequencies.
#
# Examples include:
#
# * an impulse,
# * a click,
# * an impact,
# * a sudden mechanical fault.
#
# ---
#
# ### Parallel horizontal lines
#
# Harmonics.
#
# If a periodic signal has fundamental frequency $f_0$, harmonic structure may appear at
#
# $$
# f_0,\;2f_0,\;3f_0,\ldots
# $$
#
# This is common in speech and rotating machinery.
#
# ---
#
# ### Diffuse energy
#
# Broadband noise.
#
# Instead of one narrow ridge, energy is distributed across many frequencies.
#
# ---
#
# # 7. Good answer to the vertical-stripe question
#
# The slides ask:
#
# > What time-domain event would create a narrow vertical stripe across many frequencies?
#
# A good answer is:
#
# > A short broadband transient, such as an impulse, click, impact, or abrupt discontinuity.
#
# A stronger answer explains why:
#
# > A very short-duration event is highly localized in time, so its Fourier content spreads over a wide range of frequencies. Because it occurs over only a small time interval, the spectrogram shows a narrow structure horizontally but broad structure vertically.
#
# That demonstrates the time-frequency duality rather than just memorizing the pattern.
#
# ---
#
# # 8. The central STFT trade-off
#
# This is the most important conceptual idea in the lecture.
#
# The STFT requires choosing a window length.
#
# The window cannot be simultaneously arbitrarily short in time and arbitrarily narrow in frequency.
#
# The slides summarize:
#
# ### Short window
#
# * good time localization,
# * poorer frequency localization,
# * broader spectral features.
#
# ### Long window
#
# * good frequency localization,
# * poorer time localization,
# * temporal smearing.
#
#
#
# A useful shorthand is:
#
# $$
# \boxed{
# \text{short window}
# \rightarrow
# \text{good time resolution}
# }
# $$
#
# and
#
# $$
# \boxed{
# \text{long window}
# \rightarrow
# \text{good frequency resolution}
# }
# $$
#
# ---
#
# # 9. Why does a short window give poor frequency resolution?
#
# Suppose the sample rate is
#
# $$
# F_s.
# $$
#
# If the frame length is $L$, the approximate Fourier-bin spacing is
#
# $$
# \Delta f =
# \frac{F_s}{L}.
# $$
#
# If $L$ is small, then
#
# $$
# \Delta f
# $$
#
# is relatively large.
#
# There are fewer samples from which to estimate frequency behavior.
#
# The window spectrum also has a relatively broad main lobe.
#
# So nearby tones tend to merge.
#
# ---
#
# # 10. Why does a long window give poor time localization?
#
# Suppose the window duration is
#
# $$
# T_w =
# \frac{L}{F_s}.
# $$
#
# If $T_w$ is long, then every local spectrum represents signal behavior averaged across a relatively long time interval.
#
# If the signal changes halfway through that interval, both behaviors may contribute to the same frame.
#
# So the transition becomes blurred in time.
#
# A useful phrase for students is:
#
# > A long window asks, "What frequencies were present over this long interval?" rather than "What was happening at this exact instant?"
#
# ---
#
# # 11. Numerical example: short versus long window
#
# Suppose
#
# $$
# F_s=8000\text{ Hz}.
# $$
#
# ### 20-ms window
#
# Window duration:
#
# $$
# T_w=0.02\text{ s}.
# $$
#
# Number of samples:
#
# $$
# L =
# 8000(0.02) =
# 160.
# $$
#
# Approximate frequency spacing:
#
# $$
# \Delta f =
# \frac{8000}{160} =
# 50\text{ Hz}.
# $$
#
# This gives reasonably good time localization but relatively coarse frequency discrimination.
#
# ### 200-ms window
#
# Now
#
# $$
# L =
# 8000(0.2) =
# 1600.
# $$
#
# Then
#
# $$
# \Delta f =
# \frac{8000}{1600} =
# 5\text{ Hz}.
# $$
#
# The frequency structure becomes much sharper.
#
# But events within that 200-ms window become temporally blurred.
#
# That is the trade-off in concrete numbers.
#
# ---
#
# # 12. Why window length is not the same thing as FFT length
#
# This is another important misconception.
#
# Suppose the actual frame contains
#
# $$
# L=256
# $$
#
# samples.
#
# You can compute:
#
# $$
# N_{\text{FFT}}=256
# $$
#
# or zero-pad and use
#
# $$
# N_{\text{FFT}}=2048.
# $$
#
# The second calculation creates a denser frequency grid.
#
# But it does not change the fact that only 256 measured samples were observed.
#
# So:
#
# $$
# \text{larger }N_{\text{FFT}}
# \rightarrow
# \text{denser displayed spectrum}
# $$
#
# but
#
# $$
# \text{larger }N_{\text{FFT}}
# \not\rightarrow
# \text{fundamentally better true frequency resolution}.
# $$
#
# The slides explicitly warn that resolution is not the same as FFT display density. 
#
# ---
#
# # 13. A very useful clicker question
#
# > A 20-ms frame is zero-padded from 256 points to 4096 FFT points. Did its true frequency resolution improve?
#
# A. Yes, by a factor of 16
# B. Yes, slightly
# C. No; only the frequency sampling became denser
# D. No; the STFT no longer contains frequency information
#
# Correct answer:
#
# **C.**
#
# Good explanation:
#
# > The amount of measured time-domain data is unchanged. Zero-padding evaluates the same windowed DTFT on a denser frequency grid but does not narrow the window's spectral main lobe.
#
# ---
#
# # 14. Hop size is different from window size
#
# Another important distinction:
#
# Window length determines the amount of signal used for each local spectrum.
#
# Hop size determines how far the window moves between frames.
#
# If the hop is $R$, adjacent frames are separated by
#
# $$
# \frac{R}{F_s}
# $$
#
# seconds.
#
# The slides emphasize:
#
# * small $R$ gives a dense time grid;
# * large $R$ gives fewer frames and lower computational cost;
# * changing overlap does not remove the fundamental time-frequency trade-off. 
#
# ---
#
# # 15. Example: window versus hop
#
# Suppose
#
# $$
# F_s=16000\text{ Hz}
# $$
#
# and the window length is
#
# $$
# L=400.
# $$
#
# Then window duration is
#
# $$
# \frac{400}{16000} =
# 25\text{ ms}.
# $$
#
# If hop size is
#
# $$
# R=160,
# $$
#
# then adjacent frames occur every
#
# $$
# \frac{160}{16000} =
# 10\text{ ms}.
# $$
#
# So the window examines 25 ms of signal at a time, but produces a new spectrum every 10 ms.
#
# The frames overlap.
#
# That distinction is very important:
#
# > **window duration** tells us how much time contributes to each spectrum.
#
# > **hop duration** tells us how frequently spectra are produced.
#
# ---
#
# # 16. Does smaller hop size improve time resolution?
#
# This is subtle.
#
# A smaller hop gives finer **time sampling** of the STFT.
#
# But it does not fundamentally change the temporal smearing imposed by the window length.
#
# For example, if you use a 200-ms window but move it every 1 ms, you get many heavily overlapping spectra.
#
# The plot may look temporally dense.
#
# But each individual spectrum still averages over about 200 ms of signal.
#
# So:
#
# $$
# \text{small hop}
# \neq
# \text{arbitrarily high intrinsic time resolution}.
# $$
#
# This would make an excellent conceptual question.
#
# ---
#
# # 17. Why overlap is common
#
# Suppose a Hann window strongly downweights values near its edges.
#
# If frames did not overlap, some portions of the original signal might repeatedly appear near low-weight regions.
#
# Overlap allows successive windows to observe the same data at different relative positions.
#
# That creates smoother coverage of the signal.
#
# For analysis, 50–75% overlap is common, as the slides note, although there is no universal best value. 
#
# ---
#
# # 18. What exactly is a spectrogram?
#
# The spectrogram is based on the squared magnitude of the STFT:
#
# $$
# P(r,\omega) =
# |X(r,\omega)|^2.
# $$
#
# So the complex STFT
#
# $$
# X(r,\omega)
# $$
#
# contains magnitude and phase.
#
# The spectrogram discards the phase and visualizes local energy or power-like content.
#
# The slides define this explicitly. 
#
# A useful distinction is:
#
# $$
# \text{STFT} =
# \text{complex time-frequency representation}
# $$
#
# while
#
# $$
# \text{spectrogram} =
# |\text{STFT}|^2.
# $$
#
# Those terms are often used casually as though they were identical, but mathematically they are not.
#
# ---
#
# # 19. Why use decibels?
#
# Spectrogram power often spans a very large dynamic range.
#
# A strong component might have power
#
# $$
# P_1=1
# $$
#
# while a weaker component has
#
# $$
# P_2=10^{-4}.
# $$
#
# On a linear color scale, the weak component may be nearly invisible.
#
# Instead, display
#
# $$
# P_{\text{dB}} =
# 10\log_{10}(P+\epsilon).
# $$
#
# Then
#
# $$
# 10\log_{10}(1)=0\text{ dB}
# $$
#
# while
#
# $$
# 10\log_{10}(10^{-4})=-40\text{ dB}.
# $$
#
# The logarithm compresses the dynamic range and makes weaker structures easier to see.
#
# The small $\epsilon$ prevents numerical problems when
#
# $$
# P=0.
# $$
#
# ---
#
# # 20. Magnitude dB versus power dB
#
# This is a useful point for students.
#
# If using **power**,
#
# $$
# P=|X|^2,
# $$
#
# then use
#
# $$
# 10\log_{10}P.
# $$
#
# If using **magnitude**,
#
# $$
# A=|X|,
# $$
#
# then use
#
# $$
# 20\log_{10}A.
# $$
#
# Because
#
# $$
# 10\log_{10}(|X|^2) =
# 20\log_{10}|X|.
# $$
#
# This prevents the common mistake of mixing 10 and 20 arbitrarily.
#
# ---
#
# # 21. Why color scaling matters
#
# The slides correctly emphasize that a spectrogram is not merely a pretty picture. 
#
# Suppose two plots use the same data but different displayed dynamic ranges.
#
# One shows
#
# $$
# -20\text{ dB to }0\text{ dB}
# $$
#
# and another shows
#
# $$
# -80\text{ dB to }0\text{ dB}.
# $$
#
# The second may expose weak harmonics and background noise that are invisible in the first.
#
# Thus two spectrograms can look very different even though the underlying numerical STFT is identical.
#
# Students should always ask:
#
# * What are the axes?
# * What are the units?
# * What is the window?
# * What is the hop?
# * Is the display magnitude or power?
# * Is it linear or logarithmic?
# * What dynamic range is displayed?
#
# ---
#
# # 22. The concept check: 500 Hz then 550 Hz
#
# The lecture asks:
#
# > A signal contains 500 Hz for the first 50 ms and 550 Hz for the next 50 ms. Would you begin with a very short or very long STFT window?
#
# The slides' intended answer is that neither extreme is ideal. 
#
# Why?
#
# The tones differ by
#
# $$
# 550-500 =
# 50\text{ Hz}.
# $$
#
# The window must be long enough to distinguish that frequency separation.
#
# But the transition occurs at 50 ms.
#
# The window must also be short enough that it does not mix substantial portions of both tones into the same frame.
#
# So the window should balance both requirements.
#
# ---
#
# # 23. A stronger quantitative answer to that concept check
#
# A student could reason:
#
# To obtain frequency spacing much smaller than
#
# $$
# 50\text{ Hz},
# $$
#
# we would like roughly
#
# $$
# T_w
# \gg
# \frac{1}{50} =
# 20\text{ ms}.
# $$
#
# But if
#
# $$
# T_w
# $$
#
# becomes comparable to or greater than 50 ms, frames near the transition will contain both tones.
#
# So a reasonable starting range might be something like 20–40 ms, depending on the window and desired separation.
#
# That is not an exact universal answer.
#
# The important thing is the reasoning:
#
# > The application itself determines the acceptable compromise.
#
# ---
#
# # 24. What a very short window might show
#
# Suppose the window is only
#
# $$
# 5\text{ ms}.
# $$
#
# The transition between the two tones would be sharply localized in time.
#
# But
#
# $$
# \frac{1}{0.005} =
# 200\text{ Hz}.
# $$
#
# The frequency resolution would be too poor to cleanly distinguish 500 Hz from 550 Hz.
#
# They could blur together.
#
# ---
#
# # 25. What a very long window might show
#
# Suppose the window is
#
# $$
# 100\text{ ms}.
# $$
#
# Now the frequency resolution could be much better.
#
# But one window could include:
#
# * 50 ms of 500 Hz,
# * 50 ms of 550 Hz.
#
# The result may show both frequencies simultaneously, even though the physical signal contained them sequentially.
#
# That is exactly the type of temporal smearing the STFT trade-off produces.
#
# ---
#
# # 26. The uncertainty-style interpretation
#
# At a deeper level, this trade-off reflects the fact that time and frequency localization cannot both be made arbitrarily precise.
#
# A short window is highly localized in time but has a broad Fourier transform.
#
# A long window is spread in time but has a narrow Fourier transform.
#
# Conceptually:
#
# $$
# \Delta t\,\Delta f
# \gtrsim
# \text{constant}.
# $$
#
# The exact form depends on how the spreads are defined and on the chosen window.
#
# For this course, the important lesson is not necessarily the formal uncertainty bound but the engineering interpretation:
#
# > Better localization in one domain costs localization in the other.
#
# ---
#
# # 27. Speech is an excellent real example
#
# Speech is approximately locally stationary over short intervals but highly non-stationary over an entire utterance.
#
# For example, during a voiced vowel, harmonics may appear at approximately
#
# $$
# f_0,\;2f_0,\;3f_0,\ldots
# $$
#
# and the formant structure may remain approximately stable for a brief period.
#
# As the speaker transitions between phonemes, those patterns change.
#
# A global Fourier transform of an entire sentence loses most of that temporal evolution.
#
# A spectrogram retains it.
#
# This is one reason spectrogram-like representations have historically been so important in speech processing.
#
# ---
#
# # 28. Machinery example
#
# Suppose a motor speed increases.
#
# The fundamental rotation frequency increases, perhaps along with harmonics:
#
# $$
# f_r(t),
# $$
#
# $$
# 2f_r(t),
# $$
#
# $$
# 3f_r(t).
# $$
#
# A spectrogram may display multiple upward-sloping ridges.
#
# If a bearing fault produces impulsive broadband energy, short vertical structures may also appear.
#
# That means geometric structures in the spectrogram can correspond directly to physical machine behavior.
#
# ---
#
# # 29. Radar example
#
# A radar chirp deliberately changes frequency over time.
#
# The transmitted signal might have
#
# $$
# f(t) =
# f_0+kt.
# $$
#
# Its spectrogram contains a clear diagonal ridge.
#
# Thus the STFT can expose modulation structure that is almost invisible in a conventional magnitude spectrum.
#
# This is a useful example because many ECE students immediately connect it to communication and sensing systems.
#
# ---
#
# # 30. Spectrograms as ML inputs
#
# This is the major bridge in the second half of the lecture.
#
# The STFT creates a matrix:
#
# $$
# S[f,t].
# $$
#
# That matrix has local structure in two dimensions:
#
# * nearby frequencies,
# * nearby times.
#
# A 2-D convolutional network can therefore operate on local neighborhoods.
#
# The slides give a typical tensor format
#
# $$
# [\text{batch},\text{channel},\text{frequency},\text{time}].
# $$
#
# For example,
#
# $$
# [32,1,257,101].
# $$
#
#
#
# That could mean:
#
# * 32 examples,
# * 1 spectrogram channel,
# * 257 frequency bins,
# * 101 time frames.
#
# ---
#
# # 31. Why a CNN makes sense here
#
# A CNN uses small learned filters that scan across local neighborhoods.
#
# In a spectrogram, those filters could learn structures such as:
#
# * horizontal ridges,
# * vertical transients,
# * diagonal chirps,
# * harmonic stacks,
# * frequency modulation,
# * onset patterns.
#
# The model does not have to be told explicitly:
#
# > Look for a rising chirp.
#
# A learned 2-D kernel might become sensitive to diagonal local structure.
#
# That is one reason the spectrogram is such a natural bridge between DSP and computer vision-style neural architectures.
#
# ---
#
# # 32. But a spectrogram is not actually an ordinary photograph
#
# This is one of the best points in the lecture.
#
# Time and frequency have very different meanings.
#
# A photographic image usually treats horizontal and vertical spatial directions somewhat symmetrically.
#
# A spectrogram does not.
#
# Horizontal translation means:
#
# > the event occurred later.
#
# Vertical translation means:
#
# > the event occurred at a different frequency.
#
# These are physically different transformations.
#
# The slides explicitly warn against blindly applying generic image augmentations. 
#
# ---
#
# # 33. Why vertical flipping is usually nonsense
#
# Imagine vertically flipping an ordinary photograph.
#
# It may still represent a plausible image.
#
# Now vertically flip a spectrogram.
#
# A component originally at 200 Hz might be mapped near several kHz.
#
# Harmonic relationships may be reversed.
#
# Low-frequency phenomena become high-frequency phenomena.
#
# That usually has no physical interpretation.
#
# So a vertical flip can create training examples that could never occur in the real system.
#
# ---
#
# # 34. Why arbitrary rotation is usually worse
#
# Suppose you rotate a spectrogram by 90 degrees.
#
# Now time becomes frequency and frequency becomes time.
#
# That has no meaningful physical interpretation for most signal-processing problems.
#
# Even small arbitrary rotations alter the slopes and scales of temporal-frequency structures in ways that may not correspond to any plausible signal operation.
#
# So augmentations need to be chosen using domain knowledge.
#
# ---
#
# # 35. Examples of potentially meaningful augmentations
#
# Depending on the application, reasonable augmentations might include:
#
# * small time shifts,
# * additive noise,
# * gain changes,
# * slight time stretching,
# * small physically plausible frequency shifts,
# * masking selected time or frequency regions.
#
# But even these should be justified by the invariances expected in the task.
#
# For example, if absolute frequency is essential for diagnosing a machine fault, arbitrary frequency shifting may be harmful.
#
# ---
#
# # 36. Magnitude versus power before ML
#
# The slides advise using magnitude or power consistently. 
#
# A magnitude feature uses
#
# $$
# |X(r,\omega)|.
# $$
#
# A power feature uses
#
# $$
# |X(r,\omega)|^2.
# $$
#
# Then log compression might use something like
#
# $$
# \log(1+|X|)
# $$
#
# or
#
# $$
# \log(1+|X|^2).
# $$
#
# The exact convention matters less than consistency across training, validation, testing, and deployment.
#
# Changing conventions changes the numerical feature space seen by the model.
#
# ---
#
# # 37. Why log compression is useful for ML
#
# Suppose two spectrogram values are
#
# $$
# 1
# $$
#
# and
#
# $$
# 10^{-6}.
# $$
#
# Their ratio is enormous.
#
# A neural network operating directly on raw powers may be dominated by the strongest components.
#
# A logarithm compresses that dynamic range.
#
# For example,
#
# $$
# \log(1+P).
# $$
#
# Strong values remain strong, but weaker structures become numerically more significant.
#
# This can make learning easier.
#
# ---
#
# # 38. Why normalization must use training statistics
#
# Suppose we standardize a spectrogram value as
#
# $$
# z =
# \frac{x-\mu}{\sigma}.
# $$
#
# The mean $\mu$ and standard deviation $\sigma$ should be estimated from the training data.
#
# They should then be applied unchanged to:
#
# * validation data,
# * test data,
# * deployment data.
#
# If the test set participates in calculating $\mu$ or $\sigma$, information from the test distribution has leaked into preprocessing.
#
# The slides explicitly warn about this. 
#
# ---
#
# # 39. Why fixed STFT parameters matter
#
# Suppose one training sample uses
#
# $$
# L=256
# $$
#
# and another uses
#
# $$
# L=1024.
# $$
#
# They do not have the same frequency resolution or even necessarily the same number of frequency bins.
#
# Likewise, changing the hop changes the time axis.
#
# Therefore the STFT configuration is effectively part of the feature definition.
#
# The slides make an excellent point:
#
# > Changing window length or hop changes the feature space just like changing a network layer. 
#
# That is exactly how students should think about it.
#
# ---
#
# # 40. PyTorch pipeline
#
# The slides show the conceptual pipeline
#
# $$
# \text{waveform}
# \rightarrow
# \text{STFT}
# \rightarrow
# \text{magnitude}
# \rightarrow
# \text{log}
# \rightarrow
# [B,1,F,T]
# \rightarrow
# \text{Conv2d}.
# $$
#
#
#
# The example code uses an STFT and then something equivalent to
#
# $$
# S =
# \log(1+|Z|^2).
# $$
#
# The added channel dimension allows the tensor to be interpreted by `Conv2d`.
#
# So if the original STFT magnitude has shape
#
# $$
# [B,F,T],
# $$
#
# adding the channel dimension gives
#
# $$
# [B,1,F,T].
# $$
#
# ---
#
# # 41. Why Conv2d rather than Conv1d?
#
# This is a useful class question.
#
# A `Conv1d` network operating on the raw waveform learns local time-domain structures.
#
# A `Conv2d` network operating on a spectrogram can learn local patterns jointly across:
#
# * time,
# * frequency.
#
# For example, a diagonal chirp occupies a local 2-D pattern.
#
# A 2-D convolution can directly detect that geometry.
#
# Neither approach is universally better.
#
# They simply expose different representations to the model.
#
# ---
#
# # 42. A useful comparison
#
# ### Raw waveform model
#
# Input:
#
# $$
# x[n].
# $$
#
# Potential advantages:
#
# * retains phase,
# * retains precise temporal waveform,
# * avoids hand-designed spectral transform.
#
# Potential disadvantages:
#
# * model must learn useful frequency-selective representations from the data.
#
# ### Spectrogram model
#
# Input:
#
# $$
# |X(r,\omega)|^2.
# $$
#
# Potential advantages:
#
# * explicitly exposes time-frequency structure,
# * allows CNN filters to detect interpretable patterns.
#
# Potential disadvantages:
#
# * typically discards phase,
# * resolution is fixed by STFT parameters,
# * representation design may discard useful information.
#
# This makes a good discussion question because there is no universal winner.
#
# ---
#
# # 43. A strong answer to "Why not just use the global FFT?"
#
# Suppose the task is to classify a chirp that sweeps upward versus one that sweeps downward.
#
# The global magnitude spectra may be nearly identical because both occupy approximately the same frequency range.
#
# But the spectrogram shows:
#
# Up-chirp:
#
# $$
# f(t)\uparrow
# $$
#
# Down-chirp:
#
# $$
# f(t)\downarrow.
# $$
#
# A CNN operating on the spectrogram can distinguish the slopes.
#
# That is an excellent example of why preserving temporal organization matters.
#
# ---
#
# # 44. A good classroom example involving three signals
#
# You could compare:
#
# ### Signal A
#
# A 500-Hz tone for the full second.
#
# Spectrogram:
#
# > one horizontal ridge.
#
# ### Signal B
#
# A tone sweeping from 200 Hz to 1000 Hz.
#
# Spectrogram:
#
# > diagonal ridge.
#
# ### Signal C
#
# A short impulse halfway through.
#
# Spectrogram:
#
# > vertical broadband stripe.
#
# Then ask students to match waveforms to spectrograms.
#
# This requires genuine interpretation instead of formula recall.
#
# ---
#
# # 45. A good reverse question
#
# Show a spectrogram with:
#
# * horizontal harmonic bands,
# * interrupted briefly by a broadband vertical stripe.
#
# Ask:
#
# > Describe a plausible time-domain event.
#
# A strong answer:
#
# > The signal probably contains a sustained periodic or voiced component, producing the harmonic ridges, along with a brief transient or impact at the time corresponding to the vertical broadband stripe.
#
# That tests whether students can reason from time-frequency geometry back to physical behavior.
#
# ---
#
# # 46. The exit-ticket question
#
# The lecture asks:
#
# > A transient lasts 8 ms. Why might a 200-ms STFT window be a poor choice even if it gives excellent frequency resolution? 
#
# ### Minimum acceptable answer
#
# > Because the window is much longer than the transient, so the event will be smeared in time.
#
# ### Good answer
#
# > A 200-ms window averages signal content over a duration far longer than the 8-ms transient. Although the long window provides narrow spectral features and good frequency localization, it gives poor time localization. The transient's energy will therefore appear spread across many neighboring time frames rather than being localized near its true occurrence time.
#
# ### Excellent answer
#
# > The STFT window determines the fundamental time-frequency resolution trade-off. Since the event duration is only 8 ms, a 200-ms window is about 25 times longer than the event. Each local spectrum therefore mixes the transient with a large amount of surrounding signal. This improves frequency localization but destroys much of the temporal precision needed to identify when the transient occurred. A shorter window would generally be more appropriate if transient timing is important.
#
# ---
#
# # 47. Another strong exit-ticket question
#
# I would also consider asking:
#
# > You reduce the STFT hop from 10 ms to 1 ms but keep the window length fixed at 200 ms. Have you solved the poor time-resolution problem? Explain.
#
# Good answer:
#
# > No. The smaller hop gives a denser time grid and more overlapping frames, but each spectrum still represents roughly 200 ms of data. The fundamental temporal smearing is set mainly by the window duration, so reducing the hop does not undo the time-frequency resolution trade-off.
#
# This is an excellent diagnostic question because many students confuse frame spacing with true temporal resolution.
#
# ---
#
# # 48. Another useful numerical question
#
# Suppose
#
# $$
# F_s=16000\text{ Hz},
# $$
#
# window length
#
# $$
# L=512,
# $$
#
# and hop size
#
# $$
# R=128.
# $$
#
# Ask students to compute:
#
# ### Window duration
#
# $$
# T_w =
# \frac{512}{16000} =
# 0.032\text{ s} =
# 32\text{ ms}.
# $$
#
# ### Hop duration
#
# $$
# T_h =
# \frac{128}{16000} =
# 0.008\text{ s} =
# 8\text{ ms}.
# $$
#
# ### Overlap
#
# $$
# 512-128 =
# 384\text{ samples}.
# $$
#
# Percentage overlap:
#
# $$
# \frac{384}{512}\times100\% =
# 75\%.
# $$
#
# This is a good way to make STFT parameter settings physically meaningful rather than leaving them as arbitrary software arguments.
#
# ---
#
# # 49. What students should understand about `n_fft`
#
# A common PyTorch or librosa call contains:
#
# * `n_fft`,
# * `hop_length`,
# * `win_length`.
#
# Students frequently treat these as interchangeable.
#
# They are not.
#
# Conceptually:
#
# **`win_length`**
#
# Controls how many measured samples the local analysis window uses.
#
# This strongly controls the time-frequency resolution trade-off.
#
# **`hop_length`**
#
# Controls how far the window moves.
#
# This controls the density of the time grid and computational load.
#
# **`n_fft`**
#
# Controls FFT evaluation size.
#
# If larger than the actual window length, zero-padding gives a denser frequency sampling grid but does not create additional information.
#
# That distinction is extremely useful once students begin writing PyTorch code.
#
# ---
#
# # 50. The most important misconceptions to watch for
#
# **"The STFT gives perfect time and frequency information."**
#
# Better:
#
# > The STFT trades time localization against frequency localization through the window.
#
# **"A smaller hop gives better intrinsic time resolution."**
#
# Better:
#
# > It gives denser time sampling, but window duration still controls the fundamental temporal smearing.
#
# **"A larger FFT always improves frequency resolution."**
#
# Better:
#
# > If achieved only by zero-padding, it improves frequency-grid density but not the underlying resolution determined by the observed window.
#
# **"A spectrogram and STFT are exactly the same thing."**
#
# Better:
#
# > The STFT is complex. The spectrogram is typically $|X(r,\omega)|^2$.
#
# **"A spectrogram is basically just a photograph."**
#
# Better:
#
# > It is a physically structured matrix whose two axes have different meanings.
#
# **"Any image augmentation should work."**
#
# Better:
#
# > Spectrogram transformations must correspond to physically meaningful signal transformations.
#
# ---
#
# # 51. The conceptual thread I would emphasize throughout the lecture
#
# I would repeatedly return to these distinctions:
#
# $$
# \boxed{
# \text{global spectrum}
# \rightarrow
# \text{what frequencies occurred}
# }
# $$
#
# versus
#
# $$
# \boxed{
# \text{STFT}
# \rightarrow
# \text{what frequencies occurred when}
# }
# $$
#
# Then:
#
# $$
# \boxed{
# \text{short window}
# \rightarrow
# \text{better time localization}
# }
# $$
#
# versus
#
# $$
# \boxed{
# \text{long window}
# \rightarrow
# \text{better frequency localization}
# }
# $$
#
# And separately:
#
# $$
# \boxed{
# \text{window length}
# \neq
# \text{hop size}
# \neq
# \text{FFT display density}
# }
# $$
#
# Finally:
#
# $$
# \boxed{
# \text{spectrogram}
# \rightarrow
# \text{structured numerical feature map, not merely an image}
# }
# $$
#
# The deeper lesson connecting this lecture to the ML part of your course is that **representation choice determines what information is made easy for the model to use and what information is discarded**. The STFT makes local time-frequency patterns explicit, which is extremely useful for CNNs, but it also imposes a particular time-frequency resolution and typically discards phase when converted to a spectrogram. That is exactly the kind of DSP-versus-ML design decision students should learn to recognize.
#
# --------------------
#

# %% [markdown]
# ## Week 10, Lectures 19 and 20, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# Gemini Pro:
#
# ### Lecture 19: The DFT and Fast Convolution
#
# **Question 1: Overlap-Add Condition** (Directly from your slide wrap-up)
#
# * **Question:** Why is $N_{\text{FFT}} \ge B + L_h - 1$ required in overlap-add fast convolution, and what happens if $N_{\text{FFT}}$ is smaller than this lower bound?
#
#
# * **Answer:** The exact length of the linear convolution between an input block of length $B$ and an FIR filter of length $L_h$ is $B + L_h - 1$. Setting $N_{\text{FFT}} \ge B + L_h - 1$ ensures sufficient zero-padding so that the circular convolution computed via DFT multiplication matches the linear convolution without time-domain aliasing. If $N_{\text{FFT}}$ is smaller, the tail of the linear convolution wraps around modulo $N_{\text{FFT}}$ and corrupts the beginning of the output block.
#
#
#
# **Question 2: Overlap-Add vs. Overlap-Save**
#
# * **Question:** How do Overlap-Add (OLA) and Overlap-Save (OLS) differ in how they frame input blocks and assemble the continuous output stream?
#
#
# * **Answer:** Overlap-Add uses non-overlapping input blocks of length $B$, zero-pads them, and adds the overlapping output tails of length $L_h - 1$ from adjacent blocks to reconstruct the stream. Overlap-Save overlaps adjacent input blocks by $L_h - 1$ samples, performs circular convolution on each block, discards the first $L_h - 1$ time-aliased output samples from each block, and concatenates the remaining valid output samples without an output addition step.
#
#
#
# **Question 3: Linear vs. Circular Convolution**
#
# * **Question:** $x[n]$ has length 80 and $h[n]$ has length 33. What is the minimum FFT size required for one-shot FFT-based linear convolution, and why can't we simply multiply their 80-point DFTs?
#
#
# * **Answer:** The minimum FFT size is $N_{\text{FFT}} = 80 + 33 - 1 = 112$ points. Multiplying 80-point DFTs computes an 80-point circular convolution, which causes the final 32 samples of the 112-sample linear output to wrap around and sum into the first 32 samples, distorting the signal.
#
#
#
# ---
#
# ### Lecture 20: The STFT and Spectrograms
#
# **Question 4: Time–Frequency Resolution Trade-off** (Directly from your slide wrap-up)
#
# * **Question:** A transient event in an audio signal lasts 8 ms. Why might a 200-ms STFT window be a poor choice for analyzing this signal, even though it provides excellent frequency resolution?
#
#
# * **Answer:** A 200-ms analysis window averages energy over a broad temporal interval, severely smearing the short 8-ms transient across the entire frame in time. While a longer window yields a narrower spectral main lobe (sharper frequency resolution), it sacrifices time localization, making it impossible to precisely pinpoint when the transient occurred or separate it from closely preceding or following events.
#
#
#
# **Question 5: Global Spectrum Flaw**
#
# * **Question:** Why is a standard global Fourier transform (DTFT/DFT) inadequate for analyzing speech, ECG signals, or linear frequency modulated chirps?
#
#
# * **Answer:** A global transform integrates across the entire duration of the signal, showing *what* frequencies are present overall, but completely losing information regarding *when* those frequencies occurred. Non-stationary signals have time-varying spectral properties (e.g., sweeping pitch or transient bursts) that require short-time local windowing to capture both time and frequency dynamics simultaneously.
#
#
#
# **Question 6: Spectrograms as CNN Inputs**
#
# * **Question:** When formatting a spectrogram as a 2D tensor for a PyTorch `Conv2d` layer, why are standard computer vision augmentations like arbitrary spatial rotation or vertical flipping physically invalid?
#
#
# * **Answer:** Unlike spatial images where horizontal and vertical pixels share the same physical meaning, the axes of a spectrogram represent fundamental and non-interchangeable physical quantities: vertical for frequency and horizontal for time. Vertical flipping reverses low and high frequency bands, and rotation mixes time and frequency, creating physically nonsensical signals that violate the underlying signal dynamics the model needs to learn.
#
# ---------------------
#

# %% [markdown]
# ## Week 9, Lectures 17 and 18, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# ChatGPT:
#
# For these two Week 10 lectures, I would make the exit tickets do more than check definitions. The strongest ones ask students to **predict consequences of design choices**: FFT size, block size, zero-padding, STFT window length, and hop size.
#
# Lecture 19 focuses on DFT/FFT convolution and block processing; Lecture 20 focuses on STFT interpretation and the time-frequency resolution trade-off.  
#
# ## Lecture 19: DFT and Fast Convolution
#
# ### Exit Ticket 1 — Best overall conceptual question
#
# **Question:**
# Why does multiplying two $N$-point DFTs not automatically give the linear convolution of the original sequences?
#
# **Good answer:**
# Multiplication of $N$-point DFTs corresponds to $N$-point circular convolution. If $N$ is too small, samples from the tail of the linear convolution wrap around and contaminate the beginning. To obtain linear convolution, the FFT length must be large enough that the entire linear convolution fits without wrap-around. 
#
# A strong mathematical statement is
#
# $$
# N_{\text{FFT}}\geq L_x+L_h-1.
# $$
#
# **What you are testing:**
# Whether students understand the reason for zero-padding, not just the rule.
#
# ---
#
# ### Exit Ticket 2 — Padding calculation
#
# **Question:**
# A data block has length
#
# $$
# B=256
# $$
#
# and an FIR filter has length
#
# $$
# L_h=65.
# $$
#
# What is the minimum FFT size for overlap-add processing?
#
# **Answer:**
#
# $$
# N_{\text{FFT}}
# \geq B+L_h-1
# $$
#
# so
#
# $$
# N_{\text{FFT}}
# \geq256+65-1=320.
# $$
#
# Thus the mathematical minimum is
#
# $$
# N_{\text{FFT}}=320.
# $$
#
# An implementation might choose a larger convenient size such as 512 for computational reasons. 
#
# **What you are testing:**
# Whether students distinguish the **mathematical minimum** from an **engineering implementation choice**.
#
# ---
#
# ### Exit Ticket 3 — What fails when padding is insufficient?
#
# **Question:**
# Suppose overlap-add uses
#
# $$
# N_{\text{FFT}}<B+L_h-1.
# $$
#
# What specifically goes wrong?
#
# **Good answer:**
# The linear convolution result is longer than the circular-convolution buffer. Therefore the tail wraps back onto the beginning of the block. This time aliasing corrupts samples that overlap-add assumes are valid linear-convolution outputs.
#
# This directly targets the lecture's final question. 
#
# ---
#
# ### Exit Ticket 4 — FFT versus DFT
#
# **Question:**
# What is the difference between the DFT and FFT?
#
# **Good answer:**
# The DFT is the mathematical transform
#
# $$
# X[k] =
# \sum_{n=0}^{N-1}
# x[n]e^{-j2\pi kn/N}.
# $$
#
# The FFT is an efficient algorithm for computing exactly those DFT values. It changes the computational cost, not the transform itself. 
#
# **Weak answer to watch for:**
# "The FFT is a higher-resolution DFT."
#
# That is incorrect.
#
# ---
#
# ### Exit Ticket 5 — Overlap-add versus overlap-save
#
# **Question:**
# Complete the statements:
#
# * Overlap-add overlaps __________.
# * Overlap-save overlaps __________.
#
# Then explain what each method does with that overlap.
#
# **Answer:**
#
# * Overlap-add overlaps **output block responses** and adds them.
# * Overlap-save overlaps **input blocks** by $L_h-1$ samples and discards the first $L_h-1$ corrupted output samples. 
#
# **Excellent answer:**
# Both methods ultimately produce the same linear FIR result as direct convolution; they differ only in how they organize the block boundaries.
#
# ---
#
# ### Exit Ticket 6 — OLS arithmetic
#
# **Question:**
# An overlap-save implementation uses
#
# $$
# N_{\text{FFT}}=1024
# $$
#
# and an FIR filter of length
#
# $$
# L_h=129.
# $$
#
# How many old samples are included in each block, and how many new valid output samples are produced?
#
# **Answer:**
#
# Old samples:
#
# $$
# L_h-1=128.
# $$
#
# Valid outputs:
#
# $$
# N_{\text{FFT}}-L_h+1 =
# 1024-129+1 =
# 896.
# $$
#
# So each block contains 128 old samples and 896 new samples, and produces 896 retained output samples. 
#
# ---
#
# ### Exit Ticket 7 — Latency versus efficiency
#
# **Question:**
# Why might a very large FFT block be a poor choice even if it minimizes arithmetic cost per sample?
#
# **Good answer:**
# A large block requires more samples to accumulate before processing, increasing buffering latency. In a real-time system, latency constraints may dominate arithmetic efficiency. Larger blocks can also increase memory use and affect cache behavior. 
#
# **Excellent answer:**
# The optimal block size must balance throughput, latency, FFT efficiency, memory movement, and the downstream inference timing budget.
#
# ---
#
# ### Exit Ticket 8 — ML connection
#
# **Question:**
# Why must FFT block filtering produce the same output as direct FIR filtering before the signal is passed to an ML feature extractor?
#
# **Good answer:**
# Incorrect block handling can introduce artificial discontinuities or transients at block boundaries. A downstream feature extractor or classifier could mistake those artifacts for real signal structure. Fast convolution should change computational efficiency, not the underlying filter response. 
#
# ---
#
# ## Lecture 20: STFT and Spectrograms
#
# ### Exit Ticket 1 — Best overall conceptual question
#
# **Question:**
# A transient lasts only 8 ms. Why might a 200-ms STFT window be a poor choice even if it gives excellent frequency resolution?
#
# **Good answer:**
# The 200-ms window is much longer than the transient, so each local spectrum averages over a large amount of signal surrounding the event. The transient becomes smeared in time. A shorter window would provide better temporal localization, although frequency resolution would become worse. 
#
# This is already an excellent exit ticket in the lecture.
#
# ---
#
# ### Exit Ticket 2 — Window-length prediction
#
# **Question:**
# What happens if you double the STFT window length while leaving the sample rate unchanged?
#
# **Good answer:**
# Frequency localization improves because the window contains more samples and its spectral main lobe becomes narrower. Time localization worsens because each STFT frame averages signal behavior over a longer time interval. 
#
# In shorthand:
#
# $$
# L\uparrow
# \Rightarrow
# \text{frequency resolution improves}
# $$
#
# but
#
# $$
# L\uparrow
# \Rightarrow
# \text{time resolution worsens}.
# $$
#
# ---
#
# ### Exit Ticket 3 — Hop versus resolution
#
# **Question:**
# You keep a 200-ms STFT window but reduce the hop from 20 ms to 2 ms. Have you fixed the poor time resolution?
#
# **Good answer:**
# No. The smaller hop gives a denser time grid and more overlapping frames, but every frame still uses 200 ms of signal. The fundamental temporal smearing is still controlled mainly by the window duration. 
#
# **What you are testing:**
# Whether students distinguish **time sampling density** from **intrinsic time resolution**.
#
# ---
#
# ### Exit Ticket 4 — FFT size versus frequency resolution
#
# **Question:**
# A 256-sample STFT frame is zero-padded and transformed with a 4096-point FFT. Did the true frequency resolution improve?
#
# **Good answer:**
# No. The larger FFT gives a denser set of frequency samples and produces a smoother-looking spectrum, but only 256 measured samples were used. The true frequency localization remains limited by the window duration and window shape. 
#
# This is especially valuable because it connects directly back to the same zero-padding misconception from Lecture 19.
#
# ---
#
# ### Exit Ticket 5 — Interpret a spectrogram feature
#
# **Question:**
# What time-domain behavior would most likely produce each of these spectrogram patterns?
#
# 1. horizontal ridge
# 2. vertical stripe
# 3. sloped ridge
# 4. stack of equally spaced ridges
#
# **Answer:**
#
# 1. sustained tone
# 2. short broadband transient
# 3. chirp or changing pitch
# 4. periodic signal with harmonics
#
# These interpretations are directly emphasized in the lecture. 
#
# ---
#
# ### Exit Ticket 6 — Two switching tones
#
# **Question:**
# A signal contains 500 Hz for the first 50 ms and 550 Hz for the next 50 ms. Why is neither an extremely short nor an extremely long STFT window ideal?
#
# **Good answer:**
# A very short window has good time localization but may not provide enough frequency resolution to distinguish frequencies only 50 Hz apart. A very long window can distinguish the frequencies more clearly but may smear the 50-ms transition and show both tones simultaneously around the switch. 
#
# **Excellent answer:**
# The appropriate window is determined by both the minimum frequency separation that must be resolved and the fastest temporal change that must remain localized.
#
# ---
#
# ### Exit Ticket 7 — STFT versus spectrogram
#
# **Question:**
# What information is discarded when an STFT is converted to a spectrogram?
#
# **Good answer:**
# The STFT is complex and contains magnitude and phase. A spectrogram is typically
#
# $$
# |X(r,\omega)|^2,
# $$
#
# so the phase is discarded. The spectrogram retains local magnitude or power information versus time and frequency. 
#
# ---
#
# ### Exit Ticket 8 — Numerical parameter interpretation
#
# **Question:**
# A signal is sampled at
#
# $$
# F_s=16000\text{ Hz}.
# $$
#
# The STFT uses a 512-sample window and a 128-sample hop. Find:
#
# * the window duration,
# * the hop duration,
# * the percentage overlap.
#
# **Answer:**
#
# Window duration:
#
# $$
# T_w =
# \frac{512}{16000} =
# 0.032\text{ s} =
# 32\text{ ms}.
# $$
#
# Hop duration:
#
# $$
# T_h =
# \frac{128}{16000} =
# 0.008\text{ s} =
# 8\text{ ms}.
# $$
#
# Overlap:
#
# $$
# 512-128=384
# $$
#
# samples.
#
# Percentage overlap:
#
# $$
# \frac{384}{512}\times100\% =
# 75\%.
# $$
#
# This is a good calculation because it forces students to translate software parameters into physical time.
#
# ---
#
# ### Exit Ticket 9 — Spectrograms for CNNs
#
# **Question:**
# Why can a spectrogram be a useful input to a 2-D CNN?
#
# **Good answer:**
# A spectrogram is a structured time-frequency matrix. Local 2-D CNN kernels can learn patterns across neighboring times and frequencies, such as horizontal tones, vertical transients, diagonal chirps, harmonic stacks, and modulation patterns. 
#
# ---
#
# ### Exit Ticket 10 — Why not treat it like an ordinary photograph?
#
# **Question:**
# Why is vertical flipping usually a poor data augmentation for a spectrogram?
#
# **Good answer:**
# The vertical axis represents frequency, so flipping it maps low frequencies to high frequencies and reverses physically meaningful spectral relationships. Unlike an ordinary image, the two spectrogram axes are not interchangeable spatial dimensions. 
#
# ---
#
# ## My recommended actual exit tickets
#
# If you want **one strong question per lecture**, I would use these.
#
# ### Lecture 19
#
# > A block has length $B$ and the FIR has length $L_h$. Explain why overlap-add requires
# >
# > $$
# > N_{\text{FFT}}\geq B+L_h-1
# > $$
# >
# > and describe exactly what happens if the FFT is shorter.
#
# A strong answer should say that the block's linear convolution has length
#
# $$
# B+L_h-1,
# $$
#
# that DFT multiplication computes circular convolution, and that insufficient FFT length causes the convolution tail to wrap onto the beginning and corrupt the result. 
#
# This checks whether students understand the entire chain:
#
# $$
# \text{DFT multiplication}
# \rightarrow
# \text{circular convolution}
# \rightarrow
# \text{zero-padding}
# \rightarrow
# \text{linear convolution}.
# $$
#
# ### Lecture 20
#
# > You need to detect when a short transient occurs and also estimate its dominant frequency accurately. Explain what happens as you make the STFT window shorter or longer.
#
# A strong answer should say:
#
# > A shorter window improves time localization but broadens spectral features and worsens frequency localization. A longer window improves frequency localization but smears the transient over time. There is no window that simultaneously gives arbitrarily good resolution in both domains. 
#
# For your senior/first-year graduate DSP/ML class, I would favor these kinds of **predict–explain–justify** exit tickets. They reveal much more than questions such as "define circular convolution" or "define STFT," and they directly test whether students can make engineering design decisions from the concepts.
#
# ------------------------
#
