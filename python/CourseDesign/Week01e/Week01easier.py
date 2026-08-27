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
# # Week 1 easier
#
# **Week 1: Fundamentals.** Refresher of Discrete-Time Signals, Systems, and the DTFT. LTI properties.
#

# %% [markdown]
# Week 1 easier
#
# Can you map out two 75 min lectures for week 1:
#
# Gemini Pro:
#
# Here is a detailed breakdown for the two 75-minute lectures in Week 1.
#
# The first week is crucial for setting the tone of the course. It must firmly establish the mathematical rigor required by the Oppenheim & Schafer text while constantly reminding students *why* this math is necessary for the machine learning applications they will build later.
#

# %% [markdown]
# ## Lecture 1: Signals, LTI Systems, and Convolution
#
# **Objective:** Define the fundamental discrete-time signals, establish the properties of Linear Time-Invariant (LTI) systems, and master the convolution sum.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 15 min** | **Course Intro & The "Why"** | Introduce the dual nature of the course: classical DSP theory meets modern ML. Explain that audio features for CNNs are just manipulated discrete-time signals. |
# | **15 - 35 min** | **Fundamental Sequences** | Define the unit sample $\delta[n]$, unit step $u[n]$, and complex exponentials $x[n] = A\alpha^n e^{j\omega_0 n}$. Discuss how any discrete signal can be represented as a sum of scaled, shifted impulses. |
# | **35 - 55 min** | **System Properties** | Define Linearity (superposition) and Time-Invariance mathematically. Introduce Stability (BIBO) and Causality. *Audio example:* A simple delay (echo) is LTI; a dynamic range compressor is not strictly linear. |
# | **55 - 75 min** | **The Convolution Sum** | Derive convolution: $y[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]$. Walk through a graphical example on the board (flip, shift, multiply, add). |
#
# ### Key Equations to Board
#
# * **Superposition:** $T\{a x_1[n] + b x_2[n]\} = a T\{x_1[n]\} + b T\{x_2[n]\}$
# * **Time-Invariance:** If $y[n] = T\{x[n]\}$, then $y[n-n_0] = T\{x[n-n_0]\}$
# * **Convolution:** $y[n] = x[n] * h[n]$
#
# ---
#

# %% [markdown]
# ## Lecture 2: Eigenfunctions and the DTFT
#
# **Objective:** Prove that complex exponentials are eigenfunctions of LTI systems, derive the Discrete-Time Fourier Transform (DTFT), and explore its core properties.
#
# | Time Block | Topic & Mathematical Focus | Teaching Strategy & Audio Connection |
# | --- | --- | --- |
# | **00 - 10 min** | **Recap & The Eigenfunction** | Show that passing $x[n] = e^{j\omega n}$ through an LTI system yields $y[n] = H(e^{j\omega}) e^{j\omega n}$. Explain that $H(e^{j\omega})$ is the system's frequency response. |
# | **10 - 35 min** | **Deriving the DTFT** | Introduce the forward DTFT and the Inverse DTFT (IDTFT). Emphasize that the frequency variable $\omega$ is continuous and the spectrum is always periodic with period $2\pi$. |
# | **35 - 55 min** | **DTFT Properties** | Prove the Convolution Property: $x[n]*h[n] \longleftrightarrow X(e^{j\omega})H(e^{j\omega})$. Discuss Time-Shifting and Parseval's Theorem (energy conservation). |
# | **55 - 75 min** | **Think-Pair-Share Activity** | Give students a simple difference equation (e.g., $y[n] = 0.5y[n-1] + x[n]$). Have them pair up to find the frequency response $H(e^{j\omega})$ and sketch its magnitude. |
#
# ### Key Equations to Board
#
# * **The DTFT:**
#
# $$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}$$
#
#
# * **The Inverse DTFT:**
#
# $$x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n} d\omega$$
#
#
# * **Difference to Frequency:**
#
# $$\sum_{k=0}^{N} a_k y[n-k] = \sum_{m=0}^{M} b_m x[n-m] \implies H(e^{j\omega}) = \frac{\sum_{m=0}^{M} b_m e^{-j\omega m}}{\sum_{k=0}^{N} a_k e^{-j\omega k}}$$
#
#
#
# > **Instructor Tip:** When introducing the $2\pi$ periodicity of the DTFT, relate it back to physical audio. Explain that $\omega = \pi$ represents the absolute highest frequency a digital system can represent (the Nyquist frequency), setting the stage for Week 2's deep dive into sampling and aliasing.
#
# ---
#

# %% [markdown]
# Prompt: Can you create slides and python code for these two 75 minute classes of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. Add some homework problems with solutions and a weekly quiz with solutions as well.
#
# Chat GPT:
#
# Done. I built the complete Week 1 package around the two 75-minute classes.
#
# ### Complete package
#
# [Download the complete Week 1 DSP + ML package](sandbox:/mnt/data/week1_dsp_ml/Week1_DSP_ML_Complete_Package.zip)
#
# ### Individual files
#
# * [Lecture 1 — Signals, LTI Systems, and Convolution](sandbox:/mnt/data/week1_dsp_ml/Week1_Lecture1_Signals_LTI_Convolution.pptx) — 22 slides with timing and instructor notes.
# * [Lecture 2 — Eigenfunctions and the DTFT](sandbox:/mnt/data/week1_dsp_ml/Week1_Lecture2_Eigenfunctions_DTFT.pptx) — 22 slides with instructor notes, derivations, Think-Pair-Share activity, and solution.
# * [Jupyter Notebook — Week 1 DSP + ML Code](sandbox:/mnt/data/week1_dsp_ml/Week1_DSP_ML_Code.ipynb) — NumPy, SciPy, Matplotlib, and PyTorch examples.
# * [Python Script Version](sandbox:/mnt/data/week1_dsp_ml/Week1_DSP_ML_Code.py)
# * [Homework + Weekly Quiz + Full Solutions](sandbox:/mnt/data/week1_dsp_ml/Week1_Homework_Quiz_Solutions.docx)
#
# The notebook was executed end-to-end successfully. The slides were also rendered and checked for layout/overflow problems.
#
# A few things I deliberately built in for this senior/first-year-graduate mix:
#
# **Lecture 1** develops convolution from impulse decomposition rather than presenting it as a formula, includes system-property classification, a complete finite convolution example, and makes the important distinction between DSP convolution and PyTorch's `conv1d` cross-correlation convention.
#
# **Lecture 2** actually derives the complex-exponential eigenfunction result, treats DTFT periodicity and normalized versus physical frequency carefully, derives frequency response from difference equations, and uses the (y[n]=0.5y[n-1]+x[n]) system for the Think-Pair-Share exercise.
#
# The homework has **8 core problems plus a graduate-stretch proof problem** on BIBO stability. The weekly quiz is designed for about 15 minutes and emphasizes conceptual understanding along with one short frequency-response derivation.
#
# ----------------------------------------
#

# %% [markdown]
# # Week 1, Lecture 1, LaTeX math
#
# Gemini Pro
#
# Here is the lecture content converted with properly formatted LaTeX for all the mathematical expressions, organized for clear readability:
#
# Yes. This lecture is doing more than introducing a collection of definitions. The real intellectual arc is:
#
# $$\boxed{\text{Represent signals with impulses}}\rightarrow\boxed{\text{Understand LTI properties}}\rightarrow\boxed{\text{Define }h[n]}\rightarrow\boxed{\text{derive convolution}}\rightarrow\boxed{\text{connect convolution to filtering and CNNs}}$$
#
# That chain is the part I would emphasize most strongly. The slides themselves explicitly say the goal is for students to "reason—not just compute," including understanding why the impulse response characterizes an LTI system and connecting convolution to neural-network operations.
#
# Below is how I would teach the major pieces, including good questions, expected answers, and examples.
#
# ---
#
# ## 1. Why are we starting with LTI systems?
#
# The first conceptual hurdle is that students may think, "Why spend so much time on this old DSP material if this is partly an ML course?"
#
# The answer is that **LTI systems give us the cleanest mathematical model of a signal-processing operation**. They let us predict what happens without running the algorithm on every possible input.
#
# Your slide already sets up the sequence:
#
# $$x[n]\rightarrow\text{DSP}\rightarrow\text{features}\rightarrow\text{ML}.$$
#
# and lists filtering, meaningful local structure, efficient operators, and interpretability as the connection between classical DSP and later ML.
#
# A good way to explain it verbally is:
#
# > "Before asking a neural network to learn operations on signals, we should understand the simplest class of signal-processing operators extremely well. LTI systems are that class."
#
# ### Good Opening Question
#
# **"What are some systems that take audio in and produce audio out?"**
#
# **Good student answers:**
#
# * Amplifier
# * Equalizer
# * Low-pass filter
# * Echo
# * Reverb
# * Noise reduction
# * Compressor
# * Limiter
#
# Then ask: **"Are all of these LTI?"**
#
# **Good answer:**
#
# > No. A fixed filter or fixed echo can be LTI. A compressor or limiter is nonlinear. An adaptive system may be time varying.
#
# That distinction is already anticipated in the deck.
#
# ---
#
# ## 2. What does $n$ actually mean?
#
# This seems elementary, but it is worth slowing down here.
#
# For:
#
# $$x[n],$$
#
# $n$ is an **integer sample index**, not physical time.
#
# If:
#
# $$f_s=48,000\text{ samples/s},$$
#
# then sample $n$ occurs at:
#
# $$t_n=\frac{n}{f_s}.$$
#
# So a delay of 2400 samples corresponds to:
#
# $$\frac{2400}{48000}=0.05\text{ s}=50\text{ ms}.$$
#
# That is the example in the slides.
#
# ### Good Question
#
# **"Is $x[n-10]$ always a 10-ms delay?"**
#
# **Good answer:**
#
# > No. It is a 10-sample delay. The physical delay depends on $f_s$.
#
# For example:
#
# * At **1 kHz** sampling: **10 / 1000 = 10 ms**
# * At **48 kHz** sampling: **10 / 48000 ≈ 0.208 ms**.
#
# This will become especially important when you get to sampling and frequency in Week 2.
#
# ---
#
# ## 3. The unit impulse is much more important than it looks
#
# Students often see:
#
# $$\delta[n]=\begin{cases}1,&n=0\\0,&n\neq0\end{cases}$$
#
# and think it is merely another signal they need to memorize.
#
# The important idea is:
#
# $$\boxed{\delta[n]\text{ behaves like a basis vector}}$$
#
# For a finite-dimensional vector:
#
# $$\mathbf{x}=x_0\begin{bmatrix}1\\0\\0\end{bmatrix}+x_1\begin{bmatrix}0\\1\\0\end{bmatrix}+x_2\begin{bmatrix}0\\0\\1\end{bmatrix}.$$
#
# The sequence equivalent is:
#
# $$\boxed{x[n]=\sum_{k=-\infty}^{\infty}x[k]\delta[n-k]}$$
#
# which is exactly the conceptual role described on the impulse slide.
#
# ### Example
#
# Suppose:
#
# $$x[n]=\begin{cases}2,&n=0\\-1,&n=1\\3,&n=3\\0,&\text{otherwise}.\end{cases}$$
#
# Then:
#
# $$x[n]=2\delta[n]-\delta[n-1]+3\delta[n-3].$$
#
# Ask: **"Why isn't there a $\delta[n-2]$ term?"**
#
# **Good answer:**
#
# > There could be, but its coefficient would be zero.
#
# This is important because convolution will emerge directly from this representation.
#
# ---
#
# ## 4. Unit-step shifting
#
# The deck asks:
#
# $$u[n+2]$$
#
# where does it turn on? The correct answer is $n=-2$.
#
# Students routinely struggle with the apparent "backward" sign.
#
# A useful rule:
#
# * $x[n-n_0]$ means **delay/right shift** by $n_0$
# * $x[n+n_0]$ means **advance/left shift**.
#
# ### Good Follow-up
#
# Where does:
#
# $$u[n-4]$$
#
# turn on?
# **Answer:** $n=4.$
#
# Where does:
#
# $$u[-n+3]$$
#
# turn on?
# Now the problem is more interesting because there is a reversal as well:
#
# $$-n+3\ge0$$
#
# so:
#
# $$n\le3.$$
#
# This helps prepare them for the "flip" operation in convolution.
#
# ---
#
# ## 5. Complex exponentials: don't over-teach them yet
#
# The slide presents signals of the form:
#
# $$x[n]=A\alpha^n e^{j\omega_0n}$$
#
# and says complex exponentials are the natural modes of LTI systems, with the formal eigenfunction argument deferred to Lecture 2.
#
# That is the right level for this lecture.
#
# The important pieces are:
#
# $$e^{j\omega_0 n}=\cos(\omega_0n)+j\sin(\omega_0n)$$
#
# and therefore sinusoids can be represented using complex exponentials.
#
# If $\vert{}\alpha\vert{}<1$,
#
# $$\alpha^n$$
#
# provides decay.
#
# For example,
#
# $$x[n]=(0.9)^n\cos(0.2\pi n)u[n]$$
#
# is a decaying sinusoid.
#
# ### Good Question
#
# **"What happens if $\alpha=1$?"**
#
# **Answer:**
#
# > There is no exponential growth or decay. We have a pure sinusoidal/complex-exponential oscillation.
#
# Then tease Lecture 2:
#
# > "Something remarkable happens when we put one of these through an LTI system: the frequency doesn't change. Only its amplitude and phase change."
#
# That sets up the eigenfunction property nicely.
#
# ---
#
# ## 6. What exactly is a system?
#
# This definition deserves to be simple:
#
# $$\boxed{T:\ x[n]\mapsto y[n]}$$
#
# A system is just a mapping from one sequence to another.
#
# Your examples are well chosen:
#
# $$y[n]=2x[n]$$
#
# $$y[n]=x[n-3]$$
#
# $$y[n]=\frac{1}{2}(x[n]+x[n-1])$$
#
# versus squaring, compressors, or explicitly time-varying gains.
#
# The deeper question is:
#
# > "What properties allow us to say something about the output without testing every possible input?"
#
# That motivates linearity, time invariance, causality, and stability.
#
# ---
#
# ## 7. Linearity: students should learn a test, not just a definition
#
# The formal condition is:
#
# $$T\{ax_1[n]+bx_2[n]\}=aT\{x_1[n]\}+bT\{x_2[n]\}.$$
#
# A very useful mental decomposition is:
#
# ### Scaling
#
# If $x[n]\rightarrow y[n]$, then $ax[n]\rightarrow ay[n]$.
#
# ### Addition
#
# If $x_1[n]\rightarrow y_1[n], \qquad x_2[n]\rightarrow y_2[n]$,
# then $x_1[n]+x_2[n]\rightarrow y_1[n]+y_2[n]$.
#
# Your deck gives several linear and nonlinear examples and points out the useful shortcut that every linear system must map zero input to zero output.
#
# ### Example: $y[n]=x[n]+2$
#
# Set $x[n]=0.$ Then $y[n]=2.$
# Therefore it cannot be linear.
#
# This is a **great fast rejection test**, although students should understand:
#
# > Passing the zero-input test does **not prove** linearity.
#
# For example, $y[n]=x^2[n]$ does map zero to zero but is still nonlinear.
#
# ### Excellent Clicker Question
#
# Which is linear?
# A. $y[n]=3x[n]+1$
# B. $y[n]=3x[n-2]$
# C. $y[n]=\vert{}x[n]\vert{}$
# D. $y[n]=x^2[n]$
#
# **Answer: B**.
#
# ---
#
# ## 8. Time invariance: use the two-path test
#
# Students tend to understand the phrase "time invariant," but often cannot prove it.
#
# The deck describes it correctly as shifting the input and obtaining an identical shift in the output.
#
# For a system $y[n]=T\{x[n]\}$, do two experiments:
#
# ### Path 1
#
# Compute the output and then shift:
#
#
# $$y[n-n_0].$$
#
# ### Path 2
#
# Shift the input:
#
#
# $$x[n-n_0]$$
#
#
# and run that through the system.
#
# If the two agree for arbitrary $x[n]$ and $n_0$, the system is time invariant.
#
# ### Example: $y[n]=nx[n]$
#
# **Original output:**
#
#
# $$y[n]=nx[n].$$
#
# **Shift the output:**
#
#
# $$y[n-n_0]=(n-n_0)x[n-n_0].$$
#
# **Now shift the input first:**
#
#
# $$T\{x[n-n_0]\}=n x[n-n_0].$$
#
# These are not equal:
#
#
# $$nx[n-n_0]\neq(n-n_0)x[n-n_0].$$
#
# Therefore the system is time varying.
# This is worth doing fully on the board.
#
# ---
#
# ## 9. Causality
#
# The useful physical definition is:
#
# > The output **now** cannot depend on input values from the **future**.
#
# Mathematically, $y[n_0]$ can depend only on:
#
# $$x[n],\qquad n\le n_0.$$
#
# For an LTI system,
#
# $$\boxed{h[n]=0,\qquad n<0}$$
#
# is the causality condition stated in the slides.
#
# ### Examples
#
# * $y[n]=x[n-3]$ -> causal.
# * $y[n]=x[n]+x[n-2]$ -> causal.
# * But $y[n]=x[n+1]$ is noncausal because today's output requires tomorrow's input.
#
# A nice practical distinction:
#
# > Noncausal systems are impossible for live real-time processing but perfectly possible for offline processing.
#
# For example, an audio editor can use future samples because the entire file already exists.
#
# ---
#
# ## 10. BIBO stability
#
# BIBO means:
#
# $$\boxed{\text{Bounded Input}\Rightarrow\text{Bounded Output}}$$
#
# For an LTI system:
#
# $$\boxed{\sum_{n=-\infty}^{\infty}\vert{}h[n]\vert{}<\infty}$$
#
# guarantees BIBO stability.
#
# The graduate students should see why. If $\vert{}x[n]\vert{}\le B_x,$ then:
#
# $$y[n]=\sum_k x[k]h[n-k].$$
#
# Therefore:
#
# $$\vert{}y[n]\vert{}\le\sum_k\vert{}x[k]\vert{}\vert{}h[n-k]\vert{}.$$
#
# Since $\vert{}x[k]\vert{}\le B_x,$
#
# $$\vert{}y[n]\vert{}\le B_x\sum_k\vert{}h[n-k]\vert{}.$$
#
# A shift does not change the absolute sum, so:
#
# $$\vert{}y[n]\vert{}\le B_x\sum_k\vert{}h[k]\vert{}.$$
#
# If that sum is finite, $y[n]$ is bounded.
# This is an excellent senior/graduate differentiation point: seniors should use the test; graduate students should understand the argument.
#
# ---
#
# ## 11. The classification exercise is particularly good
#
# Your slide gives these four systems:
#
# ### A. $y[n]=x[n-4800]$
#
# At **48 kHz** this is a **100 ms** delay.
# **Answer:**
#
# * Linear: yes
# * Time invariant: yes
# * Causal: yes
# * Stable: yes
# *(A bounded input remains bounded; we are only delaying it.)*
#
# ### B. $y[n]=0.8x[n]+0.2x[n-1]$
#
# **Answer:**
#
# * Linear: yes
# * Time invariant: yes
# * Causal: yes
# * Stable: yes
#
# Its impulse response is $h[n]=0.8\delta[n]+0.2\delta[n-1].$
# Therefore $\sum_n\vert{}h[n]\vert{}=1.$
#
# ### C. $y[n]=\vert{}x[n]\vert{}$
#
# **Answer:**
#
# * Linear: **no**
# * Time invariant: yes
# * Causal: yes
# * Stable: yes
#
# The interesting part is that **nonlinear does not mean bad, unstable, or time varying**. That misconception is worth explicitly addressing.
#
# ### D. $y[n]=nx[n]$
#
# **Answer:**
#
# * Linear: yes
# * Time invariant: **no**
# * Causal: yes
# * BIBO stable over all $n$: **no**
#
# For example, let $x[n]=1.$
# The input is bounded: $\vert{}x[n]\vert{}\le1.$
# But $y[n]=n,$ which is unbounded.
# This is probably the best example on the slide because it separates all four properties cleanly.
#
# ---
#
# ## 12. The most important slide: impulse response
#
# This is the conceptual hinge of the lecture, exactly as your instructor note says.
#
# Feed an impulse into an LTI system:
#
# $$\delta[n]\rightarrow h[n].$$
#
# That defines the impulse response. Now consider a shifted impulse:
#
# $$\delta[n-k].$$
#
# Because of **time invariance**:
#
# $$\delta[n-k]\rightarrow h[n-k].$$
#
# Now scale the impulse:
#
# $$x[k]\delta[n-k].$$
#
# Because of **linearity**:
#
# $$x[k]\delta[n-k]\rightarrow x[k]h[n-k].$$
#
# But every signal can be written:
#
# $$x[n]=\sum_kx[k]\delta[n-k].$$
#
# Therefore the output must be:
#
# $$y[n]=\sum_k x[k]h[n-k].$$
#
# And there is convolution.
#
# $$\boxed{y[n]=x[n]*h[n]}$$
#
# That derivation is exactly what the following slide is trying to accomplish rather than presenting convolution as an unexplained formula.
#
# A very good question is:
# **"Why do I need both linearity and time invariance?"**
#
# **Good answer:**
#
# > Time invariance tells me the response to a shifted impulse. Linearity tells me how to scale and add all those shifted impulse responses.
#
# That is one of the best possible exam conceptual questions for this material.
#
# ---
#
# ## 13. What does convolution physically mean?
#
# The formula:
#
# $$y[n]=\sum_kx[k]h[n-k]$$
#
# can look intimidating. The verbal interpretation is much easier:
#
# > Every input sample $x[k]$ creates a scaled copy of $h[n]$, shifted so that it begins at $k$. Add all those copies together.
#
# For example, if:
#
# $$x[n]=2\delta[n]+3\delta[n-2],$$
#
# then immediately:
#
# $$y[n]=2h[n]+3h[n-2].$$
#
# No flip-slide-multiply machinery is necessary. That example is extremely good for establishing the **meaning** before teaching the mechanics.
#
# ---
#
# ## 14. Flip–shift–multiply–sum
#
# Your slide gives the standard four-step graphical procedure.
#
# Starting from:
#
# $$y[n]=\sum_kx[k]h[n-k],$$
#
# treat $n$ as fixed. Then:
#
# 1. Start with $h[k]$
# 2. Reverse it: $h[-k]$
# 3. Shift it: $h[n-k]$
# 4. Multiply by $x[k]$
# 5. Sum over $k$
#
# A phrase I strongly recommend saying repeatedly is:
#
# > "$n$ tells us which output sample we are computing. $k$ is the dummy summation variable."
#
# That prevents a large fraction of convolution confusion.
#
# ---
#
# ## 15. Worked convolution
#
# The deck uses:
#
# $$x[n]=[1,2,1]$$
#
# $$h[n]=[1,-1],$$
#
# both beginning at $n=0$.
#
# The convolution is:
#
# $$y[0]=1$$
#
# $$y[1]=-1+2=1$$
#
# $$y[2]=-2+1=-1$$
#
# $$y[3]=-1$$
#
# so:
#
# $$\boxed{y[n]=[1,1,-1,-1]}$$
#
# ### Two excellent sanity checks
#
# The slides already include both.
#
# **1. Output length**
#
#
# $$L_y=L_x+L_h-1$$
#
#
# so **3 + 2 - 1 = 4**.
#
# **2. Sum property**
#
#
# $$\sum_ny[n]=\left(\sum_nx[n]\right)\left(\sum_nh[n]\right).$$
#
# Here, $\sum x=4, \qquad \sum h=0,$
# therefore $\sum y=0,$
# and indeed **1 + 1 - 1 - 1 = 0**.
# That is a very useful debugging habit.
#
# ---
#
# ## 16. A particularly good interpretation of this filter
#
# There is another insight you can add.
#
# Because $h[n]=[1,-1],$ the system computes:
#
# $$y[n]=x[n]-x[n-1].$$
#
# That is a **first difference**. So it responds strongly where the signal changes.
#
# If the input is constant, $x[n]=C,$ then:
#
# $$y[n]=C-C=0.$$
#
# This gives students a preview of high-pass behavior without requiring the DTFT yet.
#
# For image students, you can say:
#
# > This is closely related to an edge detector: differences respond to changes.
#
# That is a very natural DSP-to-ML connection.
#
# ---
#
# ## 17. Algebraic properties of convolution
#
# The slides introduce commutativity, associativity, and distributivity and connect associativity to cascaded systems.
#
# The most useful one conceptually is associativity:
#
# $$x*h_1*h_2=x*(h_1*h_2).$$
#
# Suppose:
#
# * $h_1$ models a room equalizer
# * $h_2$ models a loudspeaker
#
# Then:
#
# $$h_{\text{equiv}}=h_1*h_2$$
#
# describes the complete cascade.
#
# This becomes a good ML question:
# **"What happens if we stack five purely linear convolution layers and never insert an activation?"**
#
# **Answer:**
#
# > The complete stack is still just one linear system. Ignoring boundary/cropping details, the filters can effectively be combined into one equivalent convolution.
#
# That is why nonlinear activations such as ReLU matter.
#
# ---
#
# ## 18. NumPy convolution: what should students understand?
#
# The slide uses:
#
# ```python
# x = np.array([1., 2., 1.])
# h = np.array([1., -1.])
# y = np.convolve(x, h, mode='full')
#
# ```
#
# giving:
#
# ```text
# [1, 1, -1, -1]
#
# ```
#
# and correctly cautions against letting a library call replace mathematical understanding.
#
# One excellent question is:
# **"What does `mode='same'` mean mathematically?"**
#
# **Best answer:**
#
# > It isn't a different definition of convolution. NumPy first performs convolution conceptually and then returns a selected/cropped portion with a prescribed length.
#
# That distinction will matter later in CNNs because dimensions, padding, stride, and boundary conventions become very important.
#
# ---
#
# ## 19. PyTorch's `conv1d`: an important subtlety
#
# This slide is particularly valuable.
#
# PyTorch calls the operation "convolution," but it normally computes **cross-correlation**, meaning it does not reverse the kernel the way mathematical convolution does.
#
# Mathematical convolution:
#
#
# $$y[n]=\sum_kx[k]h[n-k].$$
#
# Cross-correlation-style local operation:
#
#
# $$y[n]\sim\sum_k x[n+k]w[k].$$
#
# ### Should students worry?
#
# For DSP: **yes**, because the distinction matters when $h[n]$ has a physical interpretation.
#
# For neural networks: usually **much less**, because the weights are learned.
# If a network needs $[1,-1]$ instead of $[-1,1],$ training can simply learn the appropriate orientation.
#
# So the important connection is:
#
# $$\boxed{\text{CNN layer = learned local weighted sum}}$$
#
# rather than:
#
# $$\boxed{\text{CNN literally implements DSP convolution exactly}}$$
#
# That nuance is worth emphasizing.
#
# ---
#
# ## 20. Concept-check answer
#
# The slide asks:
#
# > For an LTI system, which statement is always true?
#
# A. Bounded $x[n]$ implies bounded $y[n]$
# B. The response to $\delta[n-k]$ is $h[n-k]$
# C. The system must be causal
# D. The impulse response must have finite length
#
# The correct answer is **B**, as the deck indicates.
#
# Why? Because time invariance guarantees:
#
# $$\delta[n]\rightarrow h[n]$$
#
# implies:
#
# $$\delta[n-k]\rightarrow h[n-k].$$
#
# ### Why are the others wrong?
#
# **A:** Only guaranteed if the system is also BIBO stable.
# **C:** LTI systems may be noncausal. For example, $y[n]=x[n+1]$ is linear and time invariant but noncausal.
# **D:** An LTI system can have an infinite-duration impulse response.
# For example, $h[n]=(0.5)^nu[n].$
# It has infinite length yet is stable because:
#
# $$\sum_{n=0}^{\infty}(0.5)^n=2.$$
#
# That last example is especially useful because it foreshadows IIR filters.
#
# ---
#
# ## 21. A strong exit-ticket answer
#
# Your final slide asks:
#
# > "In one sentence, why does $h[n]$ characterize an LTI system?"
#
# A mediocre answer would be:
#
# > Because an LTI system is defined by its impulse response.
#
# That mostly restates the claim.
#
# A **good answer** is:
#
# > Because every input can be decomposed into scaled and shifted impulses, and linearity and time invariance tell us that the output is the corresponding sum of scaled and shifted copies of $h[n]$.
#
# An excellent shorter answer is:
#
# $$\boxed{x[n]=\sum_kx[k]\delta[n-k]\quad\Longrightarrow\quad y[n]=\sum_kx[k]h[n-k].}$$
#
# That demonstrates actual understanding.
#
# ---
#
# ## What I would want students to leave class knowing
#
# I would reduce the entire 75-minute lecture to **five questions they should be able to answer without notes**:
#
# 1. **Why is $\delta[n]$ special?**
# Because shifted impulses form a basis for arbitrary discrete-time signals.
# 2. **What makes a system linear?**
# It obeys superposition.
# 3. **What makes a system time invariant?**
# Shifting the input produces exactly the same shift in the output.
# 4. **Why does $h[n]$ completely characterize an LTI system?**
# Because every input is a weighted sum of shifted impulses.
# 5. **What does convolution actually mean?**
# Each input sample launches a scaled, shifted copy of the impulse response, and all those responses add together.
#
# If students truly understand those five ideas, then Lecture 2's jump to:
#
# $$e^{j\omega n}\rightarrow H(e^{j\omega})e^{j\omega n}$$
#
# and:
#
# $$x*h\longleftrightarrow XH$$
#
# will feel like a logical continuation rather than a completely new topic.

# %% [markdown]
# # Week 1, Lecture 2, LaTex math
#
# Gemini Pro:
#
# Here is the lecture content converted with properly formatted LaTeX for all the mathematical expressions, organized for clear readability:
#
# This lecture is making the transition from **time-domain DSP** to **frequency-domain DSP**. The central idea is that convolution, which can look mechanically complicated in time, becomes extremely simple when the input is a complex exponential.
#
# The intellectual chain is
#
# $$\boxed{\text{LTI system}} \rightarrow \boxed{\text{complex exponential input}} \rightarrow \boxed{\text{same exponential at output}} \rightarrow \boxed{H(e^{j\omega})} \rightarrow \boxed{\text{DTFT}} \rightarrow \boxed{Y(e^{j\omega})=X(e^{j\omega})H(e^{j\omega})}.$$
#
# That is exactly the direction of the lecture: prove the eigenfunction result, introduce the DTFT, develop its properties, and then use those ideas to analyze a one-pole filter.
#
# ---
#
# ## 1. The key question: what happens to a sinusoid in an LTI system?
#
# The opening question in the deck is excellent:
#
# > If a pure tone enters an LTI filter, what can change—frequency, amplitude, phase?
#
# For an LTI system, the correct answer is:
#
# * frequency stays the same,
# * amplitude may change,
# * phase may change.
#
# For example, suppose
#
# $$x[n]=\cos(0.2\pi n).$$
#
# If it passes through a low-pass LTI filter, the output might be
#
# $$y[n]=0.8\cos(0.2\pi n-0.3).$$
#
# The amplitude changed from $1$ to $0.8$, and a phase shift of $-0.3$ radians appeared, but the frequency is still
#
# $$0.2\pi\text{ rad/sample}.$$
#
# An LTI system **cannot take one sinusoid and turn it into another frequency**. That is already a remarkable constraint.
#
# Contrast this with a nonlinear system. If
#
# $$y[n]=x^2[n]$$
#
# and
#
# $$x[n]=\cos(\omega_0n),$$
#
# then
#
# $$y[n] = \cos^2(\omega_0n) = \frac{1}{2}+\frac{1}{2}\cos(2\omega_0n).$$
#
# Now we have created DC and a frequency at $2\omega_0$. So this is a very nice way to reinforce:
#
# $$\boxed{\text{new frequencies are evidence of nonlinearity}}$$
#
# in this simple setting.
#
# ---
#
# ## 2. Why complex exponentials are special
#
# From Lecture 1,
#
# $$y[n] = \sum_k h[k]x[n-k].$$
#
# Now choose
#
# $$x[n]=e^{j\omega n}.$$
#
# Then
#
# $$x[n-k] = e^{j\omega(n-k)}.$$
#
# Therefore
#
# $$y[n] = \sum_kh[k]e^{j\omega(n-k)}.$$
#
# Separate the terms involving $n$ and $k$:
#
# $$e^{j\omega(n-k)} = e^{j\omega n}e^{-j\omega k}.$$
#
# Thus
#
# $$y[n] = e^{j\omega n} \sum_kh[k]e^{-j\omega k}.$$
#
# The important point is that $\sum_kh[k]e^{-j\omega k}$ does not depend on $n$.
#
# So define
#
# $$\boxed{ H(e^{j\omega}) = \sum_kh[k]e^{-j\omega k} }$$
#
# and we get
#
# $$\boxed{ y[n]=H(e^{j\omega})e^{j\omega n}. }$$
#
# That is the proof the deck is referring to when it tells you to emphasize what depends on $n$ and what does not.
#
# ---
#
# ## 3. Why call it an eigenfunction?
#
# Students who have had linear algebra should recognize the analogy. For a matrix,
#
# $$A\mathbf{v}=\lambda\mathbf{v}.$$
#
# An eigenvector goes into a linear transformation and comes back pointing in the same direction; only a scalar multiplier appears. For an LTI system,
#
# $$T\{e^{j\omega n}\} = H(e^{j\omega})e^{j\omega n}.$$
#
# So:
#
# $$\boxed{ e^{j\omega n}\quad\text{is the eigenfunction} }$$
#
# and
#
# $$\boxed{ H(e^{j\omega})\quad\text{is the eigenvalue}. }$$
#
# That is exactly how the deck frames the result.
#
# A particularly good classroom question is:
# **"What is special about the output waveform?"**
#
# **Good answer:**
#
# > It has exactly the same $e^{j\omega n}$ dependence as the input. The system changes only the complex scalar multiplying it.
#
# **An even better answer:**
#
# > The system does not mix this frequency with other frequencies.
#
# ---
#
# ## 4. What does the complex number $H(e^{j\omega})$ mean?
#
# Write it in polar form:
#
# $$H(e^{j\omega}) = \vert{}H(e^{j\omega})\vert{}e^{j\angle H(e^{j\omega})}.$$
#
# Then
#
# $$y[n] = \vert{}H(e^{j\omega})\vert{} e^{j(\omega n+\angle H(e^{j\omega}))}.$$
#
# Therefore:
#
# * $\vert{}H(e^{j\omega})\vert{}$ tells us the amplitude scaling, and
# * $\angle H(e^{j\omega})$ tells us the phase shift.
#
# That's why the deck describes the frequency response as a **gain-and-phase lookup table indexed by frequency**.
#
# For example, suppose at $\omega=0.4\pi$,
#
# $$H(e^{j0.4\pi}) = 0.25e^{-j\pi/3}.$$
#
# Then a sinusoid at $0.4\pi$ comes out:
#
# * one quarter the amplitude,
# * shifted by $-\pi/3$,
# * at exactly the same frequency.
#
# ---
#
# ## 5. Good audio analogy: equalization
#
# An audio equalizer is approximately shaping $\vert{}H(e^{j\omega})\vert{}$.
#
# Suppose the frequency response has
#
# $$\vert{}H(e^{j\omega})\vert{}\approx 2$$
#
# for low frequencies and
#
# $$\vert{}H(e^{j\omega})\vert{}\approx 0.5$$
#
# for high frequencies.
#
# Then low-frequency sinusoidal components are amplified while high-frequency components are attenuated.
#
# Students often think of an EQ as somehow "recognizing bass." A more precise DSP description is:
#
# > The signal contains many frequency components. The filter multiplies each component by a different complex gain.
#
# That prepares them for the DTFT naturally.
#
# ---
#
# ## 6. Why introduce the DTFT now?
#
# The eigenfunction result handles **one complex exponential**. But ordinary signals are made of many frequencies.
#
# The DTFT asks:
#
# > How much of each complex exponential frequency is present in $x[n]$?
#
# The deck introduces
#
# $$\boxed{ X(e^{j\omega}) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\omega n} }$$
#
# and the inverse
#
# $$\boxed{ x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega})e^{j\omega n}\,d\omega. }$$
#
# The important conceptual point is that **time is discrete, but DTFT frequency is continuous**.
#
# So:
#
# * $x[n]$ is defined for integer $n$,
# * but $X(e^{j\omega})$ is defined for continuously varying $\omega$.
#
# ---
#
# ## 7. A useful way to explain the forward DTFT
#
# I like to describe
#
# $$X(e^{j\omega}) = \sum_nx[n]e^{-j\omega n}$$
#
# as a **similarity test**.
#
# For each candidate frequency $\omega$, multiply the signal by $e^{-j\omega n}$ and sum.
#
# If the signal contains a strong component at that frequency, the rotating phasors align coherently and produce a relatively large result. If not, the contributions tend to cancel.
#
# This becomes a very useful intuition later for the DFT and FFT.
#
# ---
#
# ## 8. Why does the inverse DTFT work?
#
# The inverse is essentially saying:
#
# > If you know how much of every frequency is present, synthesize the time-domain sequence by adding all those complex exponentials back together.
#
# So:
#
# * $X(e^{j\omega})$ is an analysis representation, while
# * $x[n]$ is recovered by synthesis.
#
# That is conceptually analogous to representing a vector by basis coefficients.
#
# ---
#
# ## 9. Why is the DTFT $2\pi$-periodic?
#
# This deserves a full derivation because students often memorize the fact but do not understand it.
#
# Take
#
# $$e^{j(\omega+2\pi)n}.$$
#
# Since $n$ is integer,
#
# $$e^{j(\omega+2\pi)n} = e^{j\omega n}e^{j2\pi n}.$$
#
# But $e^{j2\pi n}=1$ for every integer $n$. Therefore
#
# $$e^{j(\omega+2\pi)n} = e^{j\omega n}.$$
#
# So discrete time simply cannot distinguish $\omega$ from $\omega+2\pi k$. Hence
#
# $$\boxed{ X(e^{j(\omega+2\pi k)}) = X(e^{j\omega}) }$$
#
# for any integer $k$.
#
# That is exactly the reasoning behind the periodicity slide.
#
# A very good question is:
# **Are $\omega=0.2\pi$ and $\omega=2.2\pi$ different discrete-time frequencies?**
#
# **Answer:**
#
# > No. They generate exactly the same discrete-time complex exponential.
#
# ---
#
# ## 10. Physical frequency versus normalized frequency
#
# The deck gives the relationship through the examples at $f_s=48$ kHz.
#
# The general relation is
#
# $$\boxed{ \omega = 2\pi\frac{f}{f_s} }$$
#
# or
#
# $$\boxed{ f = \frac{\omega}{2\pi}f_s. }$$
#
# At $f_s=48\text{ kHz}$, if $\omega=\frac{\pi}{2}$, then
#
# $$f = \frac{\pi/2}{2\pi}(48000) = 12000\text{ Hz}.$$
#
# Likewise, $\omega=\pi$ corresponds to $24\text{ kHz}$. That's the Nyquist frequency for a 48-kHz sampling rate.
#
# ---
#
# ## 11. A good conceptual trap
#
# Ask students:
# **"What physical frequency corresponds to $\omega=\pi/2$?"**
#
# The correct answer is:
#
# > I cannot tell unless you give me $f_s$.
#
# This is a very useful check.
#
# * At 48 kHz it is 12 kHz.
# * At 8 kHz it is 2 kHz.
#
# The normalized frequency is the same; the physical frequency is not.
#
# ---
#
# ## 12. The one-sided exponential DTFT pair
#
# The deck highlights
#
# $$x[n]=a^nu[n], \qquad \vert{}a\vert{}<1.$$
#
# Its DTFT is
#
# $$X(e^{j\omega}) = \sum_{n=0}^{\infty}a^ne^{-j\omega n}.$$
#
# Combine the terms:
#
# $$\sum_{n=0}^{\infty} (ae^{-j\omega})^n.$$
#
# This is a geometric series, so
#
# $$\boxed{ X(e^{j\omega}) = \frac{1}{1-ae^{-j\omega}} }$$
#
# provided $\vert{}a\vert{}<1$.
#
# The deck emphasizes the convergence condition for precisely this reason.
#
# This pair becomes especially important because it is also the impulse response and frequency response of the one-pole system later in the lecture.
#
# ---
#
# ## 13. Time shifting becomes phase
#
# If
#
# $$x[n] \longleftrightarrow X(e^{j\omega}),$$
#
# then
#
# $$x[n-n_0] \longleftrightarrow e^{-j\omega n_0}X(e^{j\omega}).$$
#
# The multiplier $e^{-j\omega n_0}$ has magnitude $1$. Therefore:
#
# $$\boxed{\text{delay does not change spectral magnitude}}$$
#
# but its phase is $-\omega n_0$.
#
# That is why the slide says that delay appears as **linear phase versus frequency**.
#
# ### Very good example
#
# Suppose
#
# $$y[n]=x[n-5].$$
#
# Then
#
# $$Y(e^{j\omega}) = e^{-j5\omega}X(e^{j\omega}).$$
#
# Thus
#
# $$\vert{}Y\vert{}=\vert{}X\vert{}$$
#
# but
#
# $$\angle Y = \angle X-5\omega.$$
#
# This is a clean demonstration that two signals can have identical magnitude spectra while being different signals.
#
# ---
#
# ## 14. Convolution theorem: probably the biggest payoff
#
# Lecture 1 gave
#
# $$y[n]=x[n]*h[n].$$
#
# The DTFT property gives
#
# $$\boxed{ Y(e^{j\omega}) = X(e^{j\omega})H(e^{j\omega}). }$$
#
# The deck correctly presents this as one of the most important DSP simplifications.
#
# In the time domain: $\sum_k x[k]h[n-k].$
# Lots of shifting, multiplying, and summing.
#
# In the frequency domain: $X(e^{j\omega})H(e^{j\omega}).$
# Just pointwise multiplication.
#
# ---
#
# ## 15. How I would derive the convolution property
#
# Starting with
#
# $$y[n] = \sum_kx[k]h[n-k],$$
#
# take the DTFT:
#
# $$Y(e^{j\omega}) = \sum_n \left[ \sum_kx[k]h[n-k] \right]e^{-j\omega n}.$$
#
# Swap the sums:
#
# $$\sum_kx[k] \sum_nh[n-k]e^{-j\omega n}.$$
#
# Let $m=n-k$. Then $n=m+k$. So
#
# $$Y(e^{j\omega}) = \sum_kx[k] \sum_mh[m]e^{-j\omega(m+k)}.$$
#
# Separate:
#
# $$\left[ \sum_kx[k]e^{-j\omega k} \right] \left[ \sum_mh[m]e^{-j\omega m} \right].$$
#
# Therefore
#
# $$\boxed{ Y(e^{j\omega}) = X(e^{j\omega})H(e^{j\omega}). }$$
#
# For the combined senior/graduate audience, I would show the derivation but emphasize the conceptual result more than the bookkeeping.
#
# ---
#
# ## 16. Why filtering makes intuitive sense in frequency
#
# Suppose $x[n]$ contains two components:
#
# $$x[n] = \cos(0.1\pi n) + 0.5\cos(0.8\pi n).$$
#
# Suppose the filter has approximately $\vert{}H(e^{j0.1\pi})\vert{}=1$ and $\vert{}H(e^{j0.8\pi})\vert{}=0.1$.
#
# Then the output will roughly be
#
# $$y[n] \approx \cos(0.1\pi n) + 0.05\cos(0.8\pi n),$$
#
# ignoring phase for the moment.
#
# That is low-pass filtering in a nutshell. The filter doesn't need to "recognize" a low-frequency waveform in the time domain. It simply applies a large gain to some frequencies and a small gain to others.
#
# ---
#
# ## 17. Parseval's theorem: what should they understand?
#
# The deck introduces Parseval as energy conservation between representations and connects it to later spectral features.
#
# For the DTFT,
#
# $$\boxed{ \sum_{n=-\infty}^{\infty}\vert{}x[n]\vert{}^2 = \frac{1}{2\pi} \int_{-\pi}^{\pi} \vert{}X(e^{j\omega})\vert{}^2\,d\omega. }$$
#
# The left side says:
#
# > total signal energy measured across time.
#
# The right side says:
#
# > the same energy measured across frequency.
#
# A useful interpretation is: $\vert{}X(e^{j\omega})\vert{}^2$ tells us how energy is distributed by frequency.
#
# This makes the connection to audio ML particularly natural because many useful features involve energy in frequency bands.
#
# The deck appropriately warns that later operations such as mel scaling and logarithms are nonlinear and are not themselves covered by Parseval's identity.
#
# ---
#
# ## 18. Difference equations and frequency response
#
# This is another major conceptual bridge. Suppose
#
# $$\sum_{k=0}^{N}a_ky[n-k] = \sum_{m=0}^{M}b_mx[n-m].$$
#
# Because complex exponentials are eigenfunctions, assume $x[n]=e^{j\omega n}$ and $y[n] = H(e^{j\omega})e^{j\omega n}.$
#
# Substitute. For example,
#
# $$y[n]-0.5y[n-1]=x[n].$$
#
# Then
#
# $$He^{j\omega n} - 0.5He^{j\omega(n-1)} = e^{j\omega n}.$$
#
# Factor:
#
# $$He^{j\omega n} \left(1-0.5e^{-j\omega}\right) = e^{j\omega n}.$$
#
# Cancel the nonzero exponential:
#
# $$H(e^{j\omega}) \left(1-0.5e^{-j\omega}\right) = 1.$$
#
# Therefore
#
# $$\boxed{ H(e^{j\omega}) = \frac{1}{1-0.5e^{-j\omega}}. }$$
#
# The deck correctly emphasizes that this is **the same eigenfunction principle**, not a completely separate trick.
#
# ---
#
# ## 19. The one-pole system
#
# The example is
#
# $$y[n] = 0.5y[n-1]+x[n].$$
#
# The deck describes it as a recursive smoother with a preference for low frequencies.
#
# We found
#
# $$H(e^{j\omega}) = \frac{1}{1-0.5e^{-j\omega}}.$$
#
# To understand it, you do not need a full plot immediately. Evaluate a few frequencies.
#
# ---
#
# ## 20. At DC: $\omega=0$
#
# Since $e^{-j0}=1,$
#
# $$H(e^{j0}) = \frac{1}{1-0.5} = 2.$$
#
# Therefore DC gets amplified by 2.
#
# ---
#
# ## 21. At Nyquist: $\omega=\pi$
#
# Since $e^{-j\pi}=-1,$
#
# $$H(e^{j\pi}) = \frac{1}{1+0.5} = \frac{2}{3}.$$
#
# So the highest unique discrete-time frequency is attenuated.
#
# Already we know:
# $\vert{}H(0)\vert{}=2$ and $\vert{}H(\pi)\vert{}=\frac{2}{3}.$
# That strongly suggests low-pass behavior.
#
# ---
#
# ## 22. At $\omega=\pi/2$
#
# $$e^{-j\pi/2}=-j.$$
#
# Therefore
#
# $$H = \frac{1}{1+0.5j}.$$
#
# The magnitude is
#
# $$\vert{}H\vert{} = \frac{1}{\sqrt{1^2+0.5^2}} = \frac{1}{\sqrt{1.25}} \approx 0.894.$$
#
# Those are exactly the three checkpoints used in the Think-Pair-Share solution:
#
# $$2,\qquad 0.894,\qquad 0.667.$$
#
# ---
#
# ## 23. A strong Think-Pair-Share answer
#
# The slide asks students to interpret
#
# $$y[n]=0.5y[n-1]+x[n].$$
#
# A strong student response would be:
#
# > The filter is recursive because the previous output contributes to the current output. Its gain is largest at low frequency, so it has low-pass behavior. The coefficient $0.5$ causes previous samples' effects to decay geometrically, so the system has memory but remains stable.
#
# An even stronger graduate-level answer:
#
# > The corresponding transfer function has a pole at $z=0.5$. Because that pole is inside the unit circle, the causal system is BIBO stable. Moving the pole toward $z=1$ increases the response near DC and creates a narrower, stronger low-frequency peak.
#
# That directly addresses the graduate extension in the deck.
#
# ---
#
# ## 24. What does "memory" mean here?
#
# The slide asks:
#
# > What feature of the recursion causes "memory" of prior samples?
#
# A good answer is:
#
# > $y[n]$ contains $y[n-1]$, so the current output depends on a previous output, which itself depended on earlier inputs.
#
# Expanding recursively makes this even clearer:
#
# $$y[n] = x[n]+0.5y[n-1].$$
#
# Then
#
# $$y[n] = x[n]+0.5x[n-1]+0.25y[n-2].$$
#
# Continuing,
#
# $$y[n] = x[n]+0.5x[n-1]+0.25x[n-2]+\cdots$$
#
# so
#
# $$h[n] = (0.5)^nu[n].$$
#
# This creates a beautiful connection among:
#
# * the difference equation,
# * the impulse response,
# * the geometric DTFT pair,
# * the frequency response.
#
# I would definitely show that connection on the board.
#
# ---
#
# ## 25. Why does increasing 0.5 to 0.9 change the response?
#
# The notebook exercise asks students to change the recursion coefficient $0.5\to0.9\to0.99$ and predict before plotting.
#
# Generalize the system:
#
# $$y[n]=ay[n-1]+x[n].$$
#
# Then
#
# $$H(e^{j\omega}) = \frac{1}{1-ae^{-j\omega}}.$$
#
# At DC,
#
# $$\vert{}H(0)\vert{} = \frac{1}{1-a}.$$
#
# So:
#
# * For $a=0.5$, $\vert{}H(0)\vert{}=2$.
# * For $a=0.9$, $\vert{}H(0)\vert{}=10$.
# * For $a=0.99$, $\vert{}H(0)\vert{}=100$.
#
# Thus the response becomes much more strongly concentrated near low frequencies.
#
# A good question is:
# **"What happens as $a\to1^{-}$?"**
#
# **Good answer:**
#
# > The system retains memory for longer and the DC gain becomes very large. At $a=1$, the causal impulse response is no longer absolutely summable, so BIBO stability is lost.
#
# ---
#
# ## 26. The concept-check with 0.9
#
# The deck asks:
#
# $$H(e^{j\omega}) = \frac{1}{1-0.9e^{-j\omega}}.$$
#
# Where is $\vert{}H\vert{}$ largest?
#
# **Answer:** near $\omega=0$.
#
# The best reasoning is not simply:
#
# > "It's a low-pass filter."
#
# Instead look at the denominator:
#
# $$\vert{}1-0.9e^{-j\omega}\vert{}.$$
#
# At $\omega=0$, $e^{-j0}=1$ so
#
# $$\vert{}1-0.9\vert{}=0.1.$$
#
# Small denominator means large response:
#
# $$\vert{}H(0)\vert{}=10.$$
#
# At $\omega=\pi$, $e^{-j\pi}=-1$ so
#
# $$\vert{}1+0.9\vert{}=1.9$$
#
# and
#
# $$\vert{}H(\pi)\vert{} = \frac{1}{1.9} \approx 0.526.$$
#
# That is much stronger reasoning.
#
# ---
#
# ## 27. DTFT versus DFT versus FFT
#
# This is an extremely important vocabulary slide. The deck distinguishes them carefully.
#
# ### DTFT
#
# Input: $x[n]$
# Frequency: $\omega$ continuous.
# Output: $X(e^{j\omega})$ which is $2\pi$-periodic.
# It is a mathematical transform for a discrete-time sequence.
#
# ### DFT
#
# Take $N$ samples.
# The DFT gives exactly $N$ frequency samples:
#
# $$X[k],\qquad k=0,\ldots,N-1.$$
#
# Frequency is now discrete.
#
# ### FFT
#
# The FFT is **not another transform**.
# It is an efficient algorithm for computing the DFT.
#
# A very good conceptual question:
# **"Can I call `np.fft.fft()` and say I computed the DTFT?"**
#
# **Best answer:**
#
# > Not exactly. You computed a DFT of a finite set of samples. The DFT can be interpreted as frequency samples related to the DTFT of that finite-length sequence, but the DTFT itself is continuous in frequency.
#
# ---
#
# ## 28. Why this matters for spectrograms
#
# The ML bridge at the end is very important.
#
# The deck gives the chain
#
# $$x[n] \rightarrow \text{window} \rightarrow \text{local spectrum} \rightarrow \text{stack over time} \rightarrow \text{spectrogram} \rightarrow \text{CNN}.$$
#
# That means a spectrogram is not merely a pretty picture fed into a CNN.
#
# Its structure comes directly from DSP choices:
#
# * sampling rate,
# * window length,
# * overlap,
# * FFT size,
# * window function,
# * magnitude or power,
# * frequency scaling.
#
# Those choices determine what patterns the ML model is even able to see.
#
# That is a point I would keep repeating throughout the course:
#
# $$\boxed{ \text{ML performance cannot be separated from signal representation.} }$$
#
# ---
#
# ## 29. A very good ML example
#
# Suppose you are classifying two sounds:
#
# * a sustained vowel,
# * a fricative such as "s".
#
# The waveform alone may look complicated. In a spectral representation:
#
# * the vowel tends to have strong harmonic/formant structure,
# * the fricative has relatively strong high-frequency broadband energy.
#
# A CNN operating on a spectrogram can exploit those structures. But the CNN does not somehow eliminate the need to understand the DSP. The DSP determines how those structures appear.
#
# ---
#
# ## 30. A subtle question: does an LTI system really preserve a cosine frequency?
#
# Yes, but for real-valued systems there is a slight subtlety.
#
# A cosine is
#
# $$\cos(\omega_0n) = \frac{1}{2}e^{j\omega_0n} + \frac{1}{2}e^{-j\omega_0n}.$$
#
# Each exponential is an eigenfunction. For a real-valued LTI system,
#
# $$H(e^{-j\omega}) = H^*(e^{j\omega}).$$
#
# The two components recombine into another real sinusoid with the same $\omega_0$, but modified amplitude and phase.
#
# This is a nice graduate-level aside if students ask why we prove everything using complex exponentials even though audio is real.
#
# ---
#
# ## 31. Good answers to the exit ticket
#
# The deck ends with:
#
# > "Why can an LTI system change the amplitude and phase of a sinusoid but not its frequency?"
#
# A weak answer:
#
# > Because LTI systems preserve frequency.
#
# True, but circular.
#
# A good answer:
#
# > Because a complex exponential is an eigenfunction of every LTI system, so the system output is the same exponential multiplied by $H(e^{j\omega})$.
#
# A better answer:
#
# > For $x[n]=e^{j\omega n}$, convolution factors into
# > 
# > $$y[n]=e^{j\omega n}\sum_k h[k]e^{-j\omega k},$$
# > 
# > 
# > 
# > and the summation is only a complex scalar. Therefore it can change magnitude and phase but cannot alter the $e^{j\omega n}$ frequency term.
#
# That demonstrates actual understanding.
#
# ---
#
# ## 32. Questions I would ask students during this lecture
#
# These would make excellent clicker or board questions.
#
# ### Question 1
#
# If
#
# $$x[n]=e^{j0.4\pi n}$$
#
# passes through an LTI system, which output is possible?
# A. 
#
# $$2e^{j0.4\pi n}$$
#
#
# B. 
#
# $$e^{j0.8\pi n}$$
#
#
# C. 
#
# $$\cos(0.4\pi n)+\cos(0.8\pi n)$$
#
#
# D. 
#
# $$e^{j0.4\pi n}+1$$
#
# **Answer: A.**
#
# ---
#
# ### Question 2
#
# If
#
# $$H(e^{j0.2\pi}) = 3e^{-j\pi/4},$$
#
# what happens to a component at $0.2\pi$?
#
# **Answer:**
# Amplitude multiplied by 3 and phase shifted by $-\pi/4$; frequency unchanged.
#
# ---
#
# ### Question 3
#
# Are
#
# $$e^{j0.3\pi n}$$
#
# and
#
# $$e^{j2.3\pi n}$$
#
# different discrete-time signals?
#
# **Answer: No.**
# They differ by $2\pi$.
#
# ---
#
# ### Question 4
#
# At $f_s=16$ kHz, what physical frequency corresponds to
#
# $$\omega=\frac{\pi}{4}?$$
#
# $$f = \frac{\pi/4}{2\pi}(16000) = 2000\text{ Hz}.$$
#
# ---
#
# ### Question 5
#
# If
#
# $$x[n]\leftrightarrow X(e^{j\omega}),$$
#
# what happens to $\vert{}X\vert{}$ when $x[n]$ is delayed five samples?
#
# **Answer: Nothing.**
# Only phase changes:
#
# $$X(e^{j\omega})e^{-j5\omega}.$$
#
# ---
#
# ### Question 6
#
# For
#
# $$y[n]=0.8y[n-1]+x[n],$$
#
# predict whether the system favors low or high frequencies before doing any plotting.
#
# **Answer: low frequencies.**
# At DC:
#
# $$\vert{}H(0)\vert{}=\frac{1}{1-0.8}=5.$$
#
# At $\pi$:
#
# $$\vert{}H(\pi)\vert{}=\frac{1}{1+0.8}\approx 0.556.$$
#
# ---
#
# ## 33. What I would most want them to retain
#
# If students leave this lecture remembering only six things, I would want these:
#
# 1. **Complex exponentials are eigenfunctions of LTI systems.**
#
# $$e^{j\omega n} \rightarrow H(e^{j\omega})e^{j\omega n}.$$
#
#
# 2. **$H(e^{j\omega})$ says what the filter does to each frequency.**
# Magnitude changes amplitude; angle changes phase.
# 3. **The DTFT represents a discrete-time sequence over continuous normalized frequency.**
# 4. **The DTFT is $2\pi$-periodic because discrete-time exponentials repeat every $2\pi$.**
# 5. **Convolution becomes multiplication.**
#
# $$x*h \longleftrightarrow XH.$$
#
#
# 6. **Frequency-domain representations used later in ML are not separate from DSP—they grow directly out of these ideas.**
#
# The most important conceptual progression across your first two lectures is therefore:
#
# $$\boxed{ \delta[n] \rightarrow h[n] \rightarrow x*h \rightarrow e^{j\omega n} \rightarrow H(e^{j\omega}) \rightarrow X(e^{j\omega}) \rightarrow Y=XH }$$
#
# That progression is very strong pedagogically because each new idea solves a limitation of the previous viewpoint rather than appearing as an isolated mathematical topic.

# %% [markdown]
# ## Week 1, Lectures 1 and 2, Exit Tickets
#
# What are good exit ticket questions with answers for these lectures?
#
# ChatGPT (Medium):
#
# Yes. For these two Week 1 lectures, the best exit tickets should check whether students understood the **conceptual hinge** of the lecture, not whether they can grind through a long calculation in the last three minutes.
#
# Lecture 1 emphasizes LTI properties, the impulse-response argument, and convolution; Lecture 2 moves that same LTI structure into the frequency domain through complex exponential eigenfunctions and the DTFT.  
#
# ## Lecture 1 — Signals, LTI Systems, and Convolution
#
# ### 1. Why does the impulse response completely characterize an LTI system?
#
# **Question**
#
# > In one or two sentences, explain why knowing (h[n]), the response of an LTI system to (\delta[n]), is enough to determine the response to any input (x[n]).
#
# **Good answer**
#
# Any discrete-time signal can be written as
#
#
# $$x[n] =
# \sum_{k=-\infty}^{\infty}
# x[k]\delta[n-k].$$
#
#
# By linearity, the response to a scaled impulse is scaled, and by time invariance the response to $\delta[n-k]$ is $h[n-k]$. Therefore,
#
#
# $$y[n] =
# \sum_{k=-\infty}^{\infty}
# x[k]h[n-k] =
# x[n]*h[n].$$
#
#
# **What you are looking for**
#
# A strong answer mentions **both linearity and time invariance**, not simply “because convolution uses $h[n]$.”
#
# This is the conceptual hinge of Lecture 1 and is explicitly emphasized in the slides. 
#
# ---
#
# ### 2. Which system property is being used?
#
# **Question**
#
# For an LTI system,
#
#
# $$T{\delta[n-k]}=h[n-k].$$
#
#
# Which property gives this result?
#
# A. Linearity
# B. Time invariance
# C. Causality
# D. Stability
#
# **Answer**
#
#
# $$\boxed{\text{B. Time invariance}}$$
#
#
# Because
#
#
# $$T{\delta[n]}=h[n],$$
#
#
# shifting the input by (k) shifts the output by the same amount.
#
# **Extension**
#
# Ask:
#
# > Where does linearity enter the convolution derivation?
#
# Good answer:
#
# It lets us scale each shifted impulse by $x[k]$ and add all the responses.
#
# This matches the lecture's existing concept check. 
#
# ---
#
# ### 3. Predict before computing
#
# **Question**
#
# Let
#
#
# $$x[n]=[1,2,1]$$
#
#
# and
#
#
# $$h[n]=[1,-1],$$
#
#
# with both starting at (n=0).
#
# Without doing the full convolution, what is the output length?
#
# **Answer**
#
#
# $$L_y=L_x+L_h-1$$
#
#
# so
#
#
# $$L_y=3+2-1=4.$$
#
#
# Thus,
#
#
# $$\boxed{L_y=4}.$$
#
#
# **Why this is a good exit ticket**
#
# It checks whether students understand support and indexing rather than just arithmetic. The worked example in the lecture explicitly uses the same sanity check. 
#
# ---
#
# ### 4. Classify a system and justify one property
#
# **Question**
#
# Consider
#
#
# $$y[n]=|x[n]|.$$
#
#
# Is the system:
#
# * linear?
# * time invariant?
# * causal?
# * BIBO stable?
#
# Briefly justify one answer.
#
# **Answer**
#
# * Linear: **No**
# * Time invariant: **Yes**
# * Causal: **Yes**
# * BIBO stable: **Yes**
#
# A good justification for nonlinearity is
#
#
# $$|ax[n]|\neq a|x[n]|$$
#
#
# for general negative or complex (a).
#
# A good stability explanation is that if
#
#
# $$|x[n]|\leq B,$$
#
#
# then
#
#
# $$|y[n]|=|x[n]|\leq B.$$
#
#
# This classification is consistent with the system-property exercise in the lecture. 
#
# ---
#
# ### 5. Causality and the impulse response
#
# **Question**
#
# For an LTI system, what condition on $h[n]$ guarantees causality?
#
# **Answer**
#
#
# $$\boxed{
# h[n]=0,\qquad n<0.
# }$$
#
#
# **Good explanation**
#
# A causal output at time $n_0$ cannot depend on future input values. If $h[n]$ were nonzero for negative $n$, the convolution sum could involve input samples with indices greater than the current output index.
#
# The lecture states this condition explicitly. 
#
# ---
#
# ### 6. BIBO stability
#
# **Question**
#
# For an LTI system, what condition on $h[n]$ guarantees BIBO stability?
#
# **Answer**
#
#
# $$\boxed{
# \sum_{n=-\infty}^{\infty}|h[n]|<\infty.
# }$$
#
#
# **Good explanation**
#
# If
#
#
# $$|x[n]|\leq B_x,$$
#
#
# then
#
#
# $$|y[n]| =
# \left|
# \sum_k x[k]h[n-k]
# \right|
# \leq
# B_x\sum_k|h[k]|.$$
#
#
# Therefore the output remains bounded if $h[n]$ is absolutely summable. 
#
# ---
#
# ### 7. DSP-to-ML bridge
#
# **Question**
#
# What is the main similarity between an FIR filter and a 1-D convolution layer in a neural network?
#
# **Good answer**
#
# Both compute local weighted sums of neighboring samples. In classical DSP, the weights are the filter coefficients
#
#
# $$h[k],$$
#
#
# while in a neural network the kernel weights are typically learned from data.
#
# An even stronger answer mentions that deep-learning libraries commonly implement **cross-correlation** rather than textbook convolution, so the kernel orientation differs. 
#
# ---
#
# ### 8. Why do stacked linear convolution layers collapse?
#
# **Question**
#
# Suppose two convolution layers are cascaded with no nonlinear activation between them. What is the equivalent system?
#
# **Answer**
#
# Another single convolution whose impulse response is
#
#
# $$\boxed{
# h_{\text{eq}}[n] =
# h_1[n]*h_2[n].
# }$$
#
#
# Because convolution is associative,
#
#
# $$x*h_1*h_2 =
# x*(h_1*h_2).$$
#
#
# **ML connection**
#
# This previews why nonlinear activations are necessary in neural networks. The lecture makes exactly this connection. 
#
# ---
#
# ## Lecture 2 — Eigenfunctions and the DTFT
#
# ### 1. Why does a sinusoid keep its frequency?
#
# **Question**
#
# Why can an LTI system change the amplitude and phase of a sinusoid but not its frequency?
#
# **Good answer**
#
# Complex exponentials are eigenfunctions of LTI systems:
#
#
# $$x[n] =
# e^{j\omega_0 n}$$
#
#
# produces
#
#
# $$y[n] =
# H(e^{j\omega_0})
# e^{j\omega_0 n}.$$
#
#
# The multiplier
#
#
# $$H(e^{j\omega_0})$$
#
#
# can change magnitude and phase, but the factor
#
#
# $$e^{j\omega_0 n}$$
#
#
# remains at the same frequency.
#
# This is also the exit-ticket prompt already suggested at the end of the deck.  
#
# ---
#
# ### 2. What is the eigenvalue?
#
# **Question**
#
# For input
#
#
# $$x[n]=e^{j\omega n},$$
#
#
# what is the eigenvalue of an LTI system?
#
# **Answer**
#
#
# $$\boxed{
# H(e^{j\omega}) =
# \sum_{k=-\infty}^{\infty}
# h[k]e^{-j\omega k}.}$$
#
#
# This is the DTFT of the impulse response.
#
# A strong answer also says:
#
# > $H(e^{j\omega})$ is the system's frequency response at frequency $\omega$.
#
# ---
#
# ### 3. Why is the DTFT $2\pi$-periodic?
#
# **Question**
#
# Explain why
#
#
# $$X(e^{j(\omega+2\pi)}) =
# X(e^{j\omega}).$$
#
#
# **Good answer**
#
# Because for integer $n$,
#
#
# $$e^{-j(\omega+2\pi)n} =
# e^{-j\omega n}e^{-j2\pi n}=
# e^{-j\omega n},$$
#
#
# since
#
#
# $$e^{-j2\pi n}=1.$$
#
#
# Therefore discrete-time frequencies separated by $2\pi$ are indistinguishable. The lecture explicitly uses this as the reason for DTFT periodicity. 
#
# ---
#
# ### 4. Convert normalized frequency to physical frequency
#
# **Question**
#
# At
#
#
# $$f_s=48\text{ kHz},$$
#
#
# what physical frequency corresponds to
#
#
# $$\omega=\frac{\pi}{2}?$$
#
#
# Use
#
#
# $$\omega =
# 2\pi\frac{f}{f_s}.$$
#
#
# **Answer**
#
#
# $$f = \frac{\omega}{2\pi}f_s =
# \frac{\pi/2}{2\pi}(48,000)$$
#
#
# so
#
#
# $$\boxed{
# f=12\text{ kHz}.}$$
#
#
# The lecture uses this exact correspondence. 
#
# ---
#
# ### 5. What does a delay do in frequency?
#
# **Question**
#
# If
#
#
# $$y[n]=x[n-n_0],$$
#
#
# what happens to the DTFT?
#
# **Answer**
#
#
# $$\boxed{
# Y(e^{j\omega}) =
# e^{-j\omega n_0}
# X(e^{j\omega}).}$$
#
#
# Therefore,
#
#
# $$|Y(e^{j\omega})| =
# |X(e^{j\omega})|$$
#
#
# and the phase changes by
#
#
# $$-\omega n_0.$$
#
#
# **Good interpretation**
#
# A pure delay does not change spectral magnitude; it adds linear phase. 
#
# ---
#
# ### 6. Convolution theorem
#
# **Question**
#
# Complete the statement:
#
# If
#
#
# $$y[n]=x[n]*h[n],$$
#
#
# then in the frequency domain
#
#
# $$\boxed{?}$$
#
#
# **Answer**
#
#
# $$\boxed{
# Y(e^{j\omega}) =
# X(e^{j\omega})H(e^{j\omega}).}$$
#
#
# **Follow-up**
#
# Why is this useful?
#
# Good answer:
#
# > Convolution, which is a sum over shifted products in time, becomes pointwise multiplication in frequency.
#
# This is one of the central simplifications emphasized in the lecture. 
#
# ---
#
# ### 7. Predict low-pass behavior without plotting
#
# **Question**
#
# Consider
#
#
# $$H(e^{j\omega}) =
# \frac{1}{1-0.9e^{-j\omega}}.$$
#
#
# Where is
#
#
# $$|H(e^{j\omega})|$$
#
#
# largest?
#
# A. Near $\omega=0$
# B. Near $\omega=\pi/2$
# C. Near $\omega=\pi$
# D. Constant for all $\omega$
#
# **Answer**
#
#
# $$\boxed{\text{A. Near }\omega=0}$$
#
#
# At
#
#
# $$\omega=0,$$
#
#
# the denominator is
#
#
# $$1-0.9=0.1,$$
#
#
# which is very small, producing a large magnitude.
#
# At
#
#
# $$\omega=\pi,$$
#
#
# the denominator is
#
#
# $$1+0.9=1.9.$$
#
#
# So the response has strong low-frequency emphasis. This is the same reasoning used in the deck's concept check. 
#
# ---
#
# ### 8. Difference equation to frequency response
#
# **Question**
#
# Given
#
#
# $$y[n] =
# 0.5y[n-1]+x[n],$$
#
#
# derive
#
#
# $$H(e^{j\omega}).$$
#
#
# **Answer**
#
# Using
#
#
# $$y[n]
# \leftrightarrow
# Y(e^{j\omega})$$
#
#
# and
#
#
# $$y[n-1]
# \leftrightarrow
# e^{-j\omega}Y(e^{j\omega}),$$
#
#
# we obtain
#
#
# $$Y(e^{j\omega}) =
# 0.5e^{-j\omega}Y(e^{j\omega})+
# X(e^{j\omega}).$$
#
#
# Thus,
#
#
# $$Y(e^{j\omega})
# \left(
# 1-0.5e^{-j\omega}
# \right) =
# X(e^{j\omega}),$$
#
#
# so
#
#
# $$\boxed{
# H(e^{j\omega}) =
# \frac{Y(e^{j\omega})}{X(e^{j\omega})} =
# \frac{1}
# {1-0.5e^{-j\omega}}.
# }$$
#
#
# That is the one-pole example used throughout the second half of the lecture. 
#
# ---
#
# ### 9. DTFT versus DFT versus FFT
#
# **Question**
#
# Give one sentence distinguishing the DTFT, DFT, and FFT.
#
# **Good answer**
#
# > The DTFT is a continuous-frequency, $2\pi$-periodic transform of a discrete-time sequence; the DFT samples a finite sequence at a finite set of frequency bins; the FFT is an efficient algorithm for computing the DFT.
#
# The deck explicitly warns against collapsing these three terms into one concept. 
#
# ---
#
# ### 10. DSP-to-ML bridge
#
# **Question**
#
# What is the connection between the DTFT and a spectrogram used as input to a machine-learning model?
#
# **Good answer**
#
# A spectrogram applies a frequency transform to short, windowed sections of a signal and stacks the resulting local spectra over time. Thus,
#
#
# $$x[n]
# \rightarrow
# \text{windowed segment}
# \rightarrow
# \text{local spectrum}
# \rightarrow
# \text{spectrogram}.$$
#
#
# The ML model then learns patterns in this structured time-frequency representation.
#
# A stronger answer adds:
#
# > The spectrogram's structure is determined by DSP choices such as window length, sampling rate, and frequency resolution before the ML model ever sees the data.
#
# That is the exact bridge highlighted near the end of Lecture 2. 
#
# ---
#
# ## My preferred exit-ticket set
#
# If you want only **three questions per lecture**, I would use these.
#
# For **Lecture 1**:
#
# 1. Why does (h[n]) completely characterize an LTI system?
# 2. For a convolution of sequences of lengths (5) and (4), what is the full-output length, and why?
# 3. What is the connection between an FIR filter and a neural-network convolution kernel?
#
# For **Lecture 2**:
#
# 1. Why can an LTI system change the amplitude and phase of a sinusoid but not its frequency?
# 2. Why is the DTFT (2\pi)-periodic?
# 3. For
#
#
# $$H(e^{j\omega}) =
# \frac{1}{1-0.9e^{-j\omega}},$$
#
#
# would you expect low-pass or high-pass behavior? Explain without plotting.
#
# Those six questions are short enough to answer in roughly **3–5 minutes**, but they reveal whether students understood the central reasoning of the two lectures rather than merely remembering formulas.
#
# ----------------
#

# %%
