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
# # Week 13 easier
#
# **Week 13: Introduction to PyTorch.** Tensors, automatic differentiation, and building basic Multilayer Perceptrons (MLPs) for signal classification (Raschka).
#

# %% [markdown]
# Can you map out two 75 min lectures for week 13. The class is a senior/graduate level DSP and ML course. The textbooks are Oppenheim/Schafer "Discrete-Time Signal Processing" and Raschka "Machine Learning with PyTorch and Scikit-Learn"
#
# Gemini Pro:
#
# Here is a structured plan for Week 13, bridging the deterministic signal processing of Oppenheim/Schafer with the data-driven framework of Raschka.
#
# ### Lecture 25: Tensors and Automatic Differentiation (75 Min)
#
# This session transitions students from traditional DSP arrays to the computational graphs that power modern machine learning.
#
# * **From Vectors to Tensors (20 min):** Map 1D discrete-time signals $x[n]$ and 2D spectrograms to PyTorch tensors. Contrast standard NumPy arrays with PyTorch tensors, emphasizing GPU acceleration and gradient tracking.
# * **Tensor Operations for DSP (20 min):** Demonstrate matrix multiplication, broadcasting, and dimensionality manipulation. Explain how a batch of framed audio is typically represented as a 3D tensor: $(N, C, L)$ for Batch, Channel, and Length.
# * **The Chain Rule & Autograd (35 min):** Introduce `torch.autograd`. Explain how PyTorch dynamically builds a directed acyclic graph during the forward pass to automatically compute gradients $\nabla_{\theta} L$ via backpropagation, eliminating the need to derive complex partial derivatives by hand.
#
# ### Lecture 26: MLPs for Signal Classification (75 Min)
#
# This session focuses on designing, assembling, and training a neural network to classify extracted DSP features (e.g., MFCCs or spectral flux).
#
# * **Architecture of an MLP (20 min):** Define the linear algebraic forward pass $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$. Discuss hidden layers and the critical necessity of non-linear activations (like ReLU) to draw complex decision boundaries between signal classes.
# * **Loss Functions & Optimizers (25 min):** Connect the network's predictions to a measurable cost. Introduce Cross-Entropy Loss for multi-class signal classification, and explain how optimizers like Adam update the network's weights: $\theta_{t+1} = \theta_t - \eta \nabla L$.
# * **The PyTorch Training Loop (30 min):** Walk through the standard 5-step training idiom required in PyTorch:
# 1. Pass the signal features forward.
# 2. Compute the loss.
# 3. Clear old gradients with `optimizer.zero_grad()`.
# 4. Backpropagate the error with `loss.backward()`.
# 5. Update weights with `optimizer.step()`.
#
# ----------
#
#

# %% [markdown]
# ### Week 13, Lecture 25: Tensors and Automatic Differentiation (75 Min)
#
# This session transitions students from traditional DSP arrays to the computational graphs that power modern machine learning.
#
# * **From Vectors to Tensors (20 min):** Map 1D discrete-time signals $x[n]$ and 2D spectrograms to PyTorch tensors. Contrast standard NumPy arrays with PyTorch tensors, emphasizing GPU acceleration and gradient tracking.
# * **Tensor Operations for DSP (20 min):** Demonstrate matrix multiplication, broadcasting, and dimensionality manipulation. Explain how a batch of framed audio is typically represented as a 3D tensor: $(N, C, L)$ for Batch, Channel, and Length.
# * **The Chain Rule & Autograd (35 min):** Introduce `torch.autograd`. Explain how PyTorch dynamically builds a directed acyclic graph during the forward pass to automatically compute gradients $\nabla_{\theta} L$ via backpropagation, eliminating the need to derive complex partial derivatives by hand.
#
# Prompt: Can you create slides and python code for this 75 minute class of a DSP and ML course for a combined senior and first year Electrical and Computer Engineering class? The DSP textbook is Discrete-Time Signal Processing by Oppenheim and Schafer. The ML textbook is Machine Learning with PyTorch and Scikit-Learn by Raschka. Add some homework problems with solutions and a weekly quiz with solutions as well. Please render math equations in LaTeX. 
#
# ChatGPT:
#
# I’ll make this lecture explicitly bridge three things the students already know: x[n] as a discrete-time sequence, framed/STFT audio as multidimensional arrays, and the chain rule from calculus. The central DSP example will be a small learnable FIR system so autograd feels like a generalization of familiar convolution—not a black-box neural-network trick.
#
# ChatGPT:  (Medium)
#
# Created. The package is built around the DSP-to-ML transition you described, with tensor shapes tied directly to signals, spectrograms, filterbanks, and learnable FIR filters.
#
# ### Files
#
# * [Complete Week 13 Lecture 25 instructor package](sandbox:/mnt/data/week13_lecture25/Week13_Lecture25_Instructor_Package.zip)
# * [36-slide PowerPoint](sandbox:/mnt/data/week13_lecture25/Week13_Lecture25_Tensors_Autograd.pptx)
# * [Jupyter notebook](sandbox:/mnt/data/week13_lecture25/notebooks/Lecture25_Tensors_Autograd.ipynb)
# * [Executed/verified notebook](sandbox:/mnt/data/week13_lecture25/notebooks/Lecture25_Tensors_Autograd_executed.ipynb)
# * [Standalone Python code](sandbox:/mnt/data/week13_lecture25/lecture25_tensors_autograd.py)
# * [Homework, quiz, and worked solutions](sandbox:/mnt/data/week13_lecture25/Lecture25_Homework_Quiz_Solutions.md)
# * [README / teaching flow](sandbox:/mnt/data/week13_lecture25/README.txt)
#
# The notebook was executed successfully with **no errors**, and the PowerPoint contains **36 slides**.
#
# ## Lecture structure
#
# The deck follows the requested 75-minute flow:
#
# **0–20 min — From vectors to tensors**
#
# Students move from familiar DSP notation,
#
# [
# x[n]
# ]
#
# to tensors such as
#
# [
# \mathbf{x}\in\mathbb{R}^{L}
# ]
#
# for a waveform and
#
# [
# \mathbf{X}\in\mathbb{C}^{F\times T}
# ]
#
# for an STFT. The slides compare NumPy and PyTorch and emphasize that tensors add device placement and optional gradient tracking rather than changing the underlying DSP mathematics.
#
# **20–40 min — Tensor operations for DSP**
#
# The lecture develops:
#
# [
# (N,C,L)
# ]
#
# for batch, channel, and signal length, along with reshaping, `unsqueeze`, broadcasting, and matrix multiplication.
#
# One DSP-specific example treats a filterbank as
#
# [
# \mathbf{E}
# ==========
#
# \mathbf{W}\mathbf{P},
# ]
#
# where
#
# [
# \mathbf{P}\in\mathbb{R}^{F\times T},
# \qquad
# \mathbf{W}\in\mathbb{R}^{M\times F},
# ]
#
# so that
#
# [
# \mathbf{E}\in\mathbb{R}^{M\times T}.
# ]
#
# This helps students see that tensor operations are not foreign ML constructs—they are familiar DSP linear algebra with more explicit dimensions.
#
# **40–75 min — Chain rule and autograd**
#
# The deck starts with a scalar chain-rule calculation by hand and then maps it to a computational graph.
#
# For example,
#
# [
# u=2\theta-3,
# ]
#
# [
# L=(u+1)^2,
# ]
#
# gives
#
# [
# \frac{dL}{d\theta}
# ==================
#
# \frac{dL}{du}
# \frac{du}{d\theta}.
# ]
#
# Then PyTorch performs the same calculation with
#
# ```python
# theta = torch.tensor(4.0, requires_grad=True)
#
# u = 2 * theta - 3
# L = (u + 1)**2
#
# L.backward()
#
# print(theta.grad)
# ```
#
# The main DSP example treats FIR coefficients as learnable parameters:
#
# [
# \hat y[n]
# =========
#
# \sum_{k=0}^{M-1} h[k]x[n-k].
# ]
#
# Using the mean-square loss
#
# [
# L
# =
#
# \frac{1}{N}
# \sum_n
# \left(
# \hat y[n]-y[n]
# \right)^2,
# ]
#
# the analytical gradient is
#
# [
# \frac{\partial L}{\partial h[r]}
# ================================
#
# \frac{2}{N}
# \sum_n
# \left(
# \hat y[n]-y[n]
# \right)x[n-r].
# ]
#
# The lecture then shows that `autograd` produces this gradient automatically and uses gradient descent,
#
# [
# h_r^{(m+1)}
# ===========
#
# ## h_r^{(m)}
#
# \eta
# \frac{\partial L}{\partial h[r]},
# ]
#
# to recover an unknown FIR filter.
#
# The demonstration converges from zero taps toward approximately
#
# [
# \boxed{
# [0.15,;0.50,;0.25,;-0.10]
# }
# ]
#
# so students see “training” as something directly connected to DSP system identification rather than immediately jumping into a neural-network black box.
#
# ## Homework and quiz
#
# The package includes **7 homework problems** covering tensor shapes, broadcasting, matrix multiplication, the chain rule, gradient accumulation, learnable FIR gradients, and `detach()` versus `no_grad()`.
#
# There is also a **10-minute weekly quiz** with six questions and a separate instructor solution section in the PowerPoint. All mathematical expressions in the instructional material are presented in LaTeX-style notation.
#
# ---------------------
#
# Week 13, Lecture 25
#
# Explain in more detail what's going on here and what good answers and/or examples are. Please use LaTeX for all math expressions
#
#

# %% [markdown]
# # Week 13, Lecture 25, LaTeX math
#
# Please convert all math expressions to LaTeX. Don't change anything else.
#
# Gemini Flash-Lite:
#
# Yes. This lecture is really making one conceptual transition:
#
#
# $$\boxed{ \text{DSP computation} \longrightarrow \text{differentiable DSP computation} }$$
#
#
# The important message is that tensors are not “neural-network objects.” They are familiar arrays plus metadata and, optionally, a record of how they were computed. The deck states this explicitly: a tensor can represent the same waveform, spectrum, filterbank, or feature matrix students already know from DSP.
# Below is how I would explain the lecture in class, including what strong answers and useful examples look like.
# Slides 2–4: What actually changes when a DSP array becomes a tensor?
# A discrete-time signal
#
#
# $$x[n]$$
#
#
# is usually introduced mathematically as a sequence.
# If we observe $L$ samples, we might write
#
# $$\mathbf{x} = \begin{bmatrix} x[0] & x[1] & \cdots & x[L-1] \end{bmatrix}^{T}.$$
#
#
# In NumPy, that becomes an array with shape
#
#
# $$(L,).$$
#
#
# In PyTorch, the numerical values may be identical and the shape can still be
#
#
# $$(L,),$$
#
#
# but the tensor also carries several practical properties:
#
#
# $$\boxed{ \text{shape}, \; \text{dtype}, \; \text{device}, \; \text{requires\_grad} }$$
#
#
# which the deck highlights.
# A useful way to phrase this is:
#
# A mathematical sequence tells us the values. A PyTorch tensor also tells the software how those values are stored and whether their computational history should be tracked.
# Good answer to the slide question
# The slide asks:
#
# “What extra information does the tensor carry that the mathematical sequence $x[n]$ does not?”
# A good student answer is:
#
# “It carries the tensor shape, numerical data type, the device where the data live, and whether PyTorch should track operations involving the tensor for gradient computation.”
# An even better answer is:
#
# “Those properties affect both correctness and performance. Shape determines axis meaning, dtype affects precision, device determines where computation happens, and requires_grad determines whether the tensor participates in automatic differentiation.”
# Slide 5: Spectrograms are already tensors
# Students sometimes hear “tensor” and imagine something exotic.
# But a spectrogram is already naturally a rank-2 object.
# If the STFT is
#
#
# $$X[k,m],$$
#
#
# where
#
#
# $$k$$
#
#
# indexes frequency and
#
#
# $$m$$
#
#
# indexes time frames, then we can represent it as
#
#
# $$\mathbf{X} \in \mathbb{C}^{F\times T}.$$
#
#
# For example, with
#
#
# $$N_{\text{FFT}}=256,$$
#
#
# a one-sided real-input STFT has
#
# $$\frac{256}{2}+1 = 129$$
#
#
# frequency bins.
# So a possible tensor shape is
#
#
# $$(129,32).$$
#
#
# The deck emphasizes that PyTorch tensors can also have complex dtype, so STFT-domain quantities can remain inside a differentiable graph.
# That is an important point.
# You do not necessarily have to reduce everything to real magnitude data before using PyTorch.
# Rank versus shape
# This is worth explicitly distinguishing.
# A tensor with shape
#
#
# $$(129,32)$$
#
#
# has rank 2 because it has two axes.
# A tensor with shape
#
#
# $$(32,2,16000)$$
#
#
# has rank 3.
# The numbers in the shape describe axis lengths.
# The number of entries in the shape describes the rank.
# Students often confuse these.
# A good check is:
#
# “What is the rank of a stereo batch with shape $(32,2,16000)$?”
# Correct answer:
#
#
# $$\boxed{3}$$
#
#
# not $16000$, and not $32$.
# Slide 6: Tensor anatomy is really engineering bookkeeping
# The deck says many PyTorch problems are not “machine-learning problems” at all—they are shape, dtype, or device problems.
# That is absolutely worth stressing to ECE students.
# I would compare tensor shape to units in circuit analysis.
# If a student writes
#
#
# $$5\text{ V} + 3\text{ A},$$
#
#
# the arithmetic operation is defined syntactically, but the physical meaning is wrong.
# Similarly, a tensor operation might run numerically while the axes are semantically wrong.
# For example, suppose
#
#
# $$\mathbf{x}$$
#
#
# has shape
#
#
# $$(N,C,L).$$
#
#
# Then
#
#
# $$x[7,1,250]$$
#
#
# means:
#
#
# $$\text{batch item } 7,$$
#
# $$\text{channel } 1,$$
#
# $$\text{sample } 250.$$
#
#
# If you accidentally interpret the shape as
#
#
# $$(N,L,C),$$
#
#
# the values may still exist, but the computation is conceptually wrong.
# That is why “print the shape” is not beginner trivia. It is a genuine model-design diagnostic.
# Slide 7: NumPy versus PyTorch
# The deck makes a subtle but useful point: the main distinction is not really
#
#
# $$\text{array} \quad\text{versus}\quad \text{tensor}$$
#
#
# but more like
#
#
# $$\text{numerical array} \quad\text{versus}\quad \text{numerical array participating in differentiable computation}.$$
#
#
# Both NumPy and PyTorch can perform vectorized operations.
# For example:
#
# ```python
# y = x**2 + 3*x
#
# ```
#
# looks nearly the same in both libraries.
# But if
#
# ```python
# x.requires_grad = True
#
# ```
#
# then PyTorch can remember enough about the operation
#
#
# $$y=x^2+3x$$
#
#
# to later compute
#
# $$\frac{dy}{dx} = 2x+3.$$
#
#
# That is the key additional capability.
# torch.from_numpy versus torch.tensor
# The deck briefly mentions a useful subtlety.
# If
#
# ```python
# t = torch.from_numpy(a)
#
# ```
#
# then on CPU, $t$ may share memory with the NumPy array $a$.
# So changing one can affect the other.
# For example:
#
# ```python
# a = np.array([1., 2., 3.], dtype=np.float32)
# t = torch.from_numpy(a)
#
# a[0] = 99
#
# ```
#
# can cause
#
# ```python
# t[0]
#
# ```
#
# to also become $99$.
# By contrast,
#
# ```python
# t = torch.tensor(a)
#
# ```
#
# normally creates a copy.
# This is a good example of why tensor behavior is partly about storage semantics, not just equations.
# Slide 8: GPU acceleration is not automatically faster
# The deck correctly says that GPU acceleration is a placement decision, not magic.
# Suppose you have a vector of only
#
#
# $$L=100$$
#
#
# samples and perform one multiplication.
# Moving those $100$ values to a GPU may cost more time than the computation itself.
# The benefit comes when the operation has lots of parallel work, such as
#
#
# $$\text{large matrix multiplication},$$
#
# $$\text{large batched convolution},$$
#
#
# or
#
#
# $$\text{many spectrograms processed together}.$$
#
#
# A good answer to
#
# “Why might a GPU be slower for a tiny operation?”
# is:
#
# “Because transferring data and launching GPU kernels has overhead. The arithmetic must be large enough to amortize that cost.”
# Slides 9–11: Shape conventions
# The concept check asks students to map familiar audio objects to shapes.
# These are worth explaining carefully.
#
# One mono clip
# Suppose one clip has
#
#
# $$L=16000$$
#
#
# samples.
# A simple representation is
#
#
# $$\boxed{(16000,)}$$
#
#
# which is rank 1.
# Batch of stereo clips
# Suppose we have
#
#
# $$N=32$$
#
#
# clips,
#
#
# $$C=2$$
#
#
# channels,
# and
#
#
# $$L=16000$$
#
#
# samples per clip.
# Then the common Conv1d convention is
#
#
# $$\boxed{(32,2,16000)}.$$
#
#
# The deck later formalizes this as
#
#
# $$(N,C,L),$$
#
#
# where $N$ is batch, $C$ is channel, and $L$ is length.
# Why include a channel dimension for mono?
# Suppose you have $32$ mono clips.
# You might initially store them as
#
#
# $$(32,16000).$$
#
#
# But a 1-D convolution layer usually expects
#
#
# $$(N,C,L).$$
#
#
# So for mono,
#
#
# $$C=1,$$
#
#
# and you insert the singleton channel dimension:
#
#
# $$\boxed{ (32,1,16000) }$$
#
#
# using
#
# ```python
# x = x.unsqueeze(1)
#
# ```
#
# The $1$ is not meaningless. It explicitly says:
#
# There is one physical signal channel.
# Slide 10: unsqueeze, permute, and reshape
# These operations are mathematically simple but semantically important.
#
# unsqueeze
# If
#
#
# $$x$$
#
#
# has shape
#
#
# $$(N,L),$$
#
#
# then
#
# ```python
# x = x.unsqueeze(1)
#
# ```
#
# produces
#
#
# $$(N,1,L).$$
#
#
# It inserts a new axis of length $1$.
# No samples are added.
# permute
# Suppose
#
#
# $$A$$
#
#
# has shape
#
#
# $$(8,20,30).$$
#
#
# Then
#
# ```python
# B = A.permute(0,2,1)
#
# ```
#
# produces shape
#
#
# $$(8,30,20).$$
#
#
# This changes which semantic quantity belongs to each axis.
# That is different from simply regrouping the same entries.
# reshape
# This is where students can get into trouble.
# Suppose
#
#
# $$A\in\mathbb{R}^{8\times20\times30}.$$
#
#
# The total number of elements is
#
#
# $$8\cdot20\cdot30=4800.$$
#
#
# You could reshape to
#
#
# $$(8,600),$$
#
#
# because
#
#
# $$8\cdot600=4800.$$
#
#
# But whether that is meaningful depends on what the original axes represented.
# A good teaching statement is:
#
# A shape-compatible reshape is not necessarily a physically meaningful reshape.
# Slide 12: Broadcasting
# The deck uses a very practical DSP example: one gain per channel.
# Suppose
#
# $$\text{audio.shape} = (N,C,L).$$
#
#
# For stereo,
#
#
# $$C=2.$$
#
#
# Suppose we want gains
#
#
# $$g_0=0.5, \qquad g_1=1.25.$$
#
#
# A natural gain tensor is
#
# $$\mathbf{g} = \begin{bmatrix} 0.5 & 1.25 \end{bmatrix}.$$
#
#
# But shape
#
#
# $$(2,)$$
#
#
# does not say which axis should correspond to channel.
# We want
#
#
# $$\boxed{ (1,C,1) }$$
#
#
# because:
# $1$ in the batch dimension means “reuse this gain for every batch item,”
# $C$ matches the channel dimension,
# and the final $1$ means “reuse this gain for every time sample.”
# So
#
# $$\mathbf{y} = \mathbf{x}\odot\mathbf{g}$$
#
#
# broadcasts correctly.
# Slide 14: Why shape $(2,\)$ fails
# The deck's concept check asks about
#
# $$\text{audio.shape} = (8,2,1000)$$
#
#
# and
#
# $$\text{gain.shape} = (2,).$$
#
#
# PyTorch broadcasting aligns shapes from the rightmost dimension.
# So conceptually it tries to compare
#
#
# $$(8,2,1000)$$
#
#
# with
#
#
# $$(\phantom{8,2,}2).$$
#
#
# The last dimensions are
#
#
# $$1000$$
#
#
# and
#
#
# $$2,$$
#
#
# which do not match and neither is $1$.
# Therefore the operation fails.
# The correct gain shape is
#
#
# $$\boxed{(1,2,1)}.$$
#
#
# Then the alignment is
#
#
# $$(8,2,1000)$$
#
#
# with
#
#
# $$(1,2,1),$$
#
#
# and every dimension is either equal or $1$.
# Slide 13: Matrix multiplication as DSP
# The filterbank example is especially useful because it shows that PyTorch operations are just familiar linear algebra.
# Suppose the power spectrogram is
#
#
# $$\mathbf{P} \in \mathbb{R}^{257\times100}.$$
#
#
# Interpretation:
# $257$ frequency bins,
# $100$ time frames.
# Suppose the filterbank matrix is
#
#
# $$\mathbf{W} \in \mathbb{R}^{40\times257}.$$
#
#
# Then
#
# $$\boxed{ \mathbf{E} = \mathbf{W}\mathbf{P} }$$
#
#
# has shape
#
#
# $$(40,100).$$
#
#
# Why?
#
#
# $$(40\times257)(257\times100) \rightarrow (40\times100).$$
#
#
# Each output entry is
#
# $$E_{m,t} = \sum_{k=0}^{256} W_{m,k}P_{k,t}.$$
#
#
# That is simply the energy in filterbank band $m$ at frame $t$.
# A strong answer to “What does matrix multiplication mean physically here?” is:
#
# “Each row of $\mathbf{W}$ defines one spectral weighting function. Multiplying $\mathbf{W}$ by each spectrogram column combines the original frequency-bin powers into band energies.”
# Slide 15: The conceptual turning point
# The lecture then asks:
#
# What if the DSP coefficients are unknown?
# That is where traditional DSP computation becomes a learnable model.
# Suppose we have a system
#
#
# $$\hat{y} = f(x;\theta),$$
#
#
# where
#
#
# $$\theta$$
#
#
# contains unknown parameters.
# We compare the predicted output
#
#
# $$\hat{y}$$
#
#
# against a desired output
#
#
# $$y$$
#
#
# using a loss
#
#
# $$L(\theta).$$
#
#
# Now we ask:
#
#
# $$\boxed{ \text{How should } \theta \text{ change so that } L \text{ decreases?} }$$
#
#
# The mathematical answer is the gradient:
#
#
# $$\boxed{ \nabla_{\theta}L }$$
#
#
# The direction
#
#
# $$-\nabla_{\theta}L$$
#
#
# is locally the direction of steepest decrease.
# This is the bridge from signal processing to learning.
# Slide 16: Chain rule by hand
# The deck gives a scalar example and evaluates the gradient at $\theta=2$.
# Suppose
#
#
# $$u=3\theta+1$$
#
#
# and
#
#
# $$L=u^2.$$
#
#
# Then
#
# $$\frac{dL}{d\theta} = \frac{dL}{du}\frac{du}{d\theta}.$$
#
#
# We have
#
# $$\frac{dL}{du} = 2u$$
#
#
# and
#
# $$\frac{du}{d\theta} = 3.$$
#
#
# Therefore,
#
# $$\frac{dL}{d\theta} = 6u.$$
#
#
# Since
#
#
# $$u=3\theta+1,$$
#
#
# we get
#
# $$\boxed{ \frac{dL}{d\theta} = 6(3\theta+1) }$$
#
#
# or
#
# $$\boxed{ \frac{dL}{d\theta} = 18\theta+6. }$$
#
#
# At
#
#
# $$\theta=2,$$
#
# $$u=7$$
#
#
# and
#
# $$\frac{dL}{d\theta} = 6(7) = \boxed{42}.$$
#
#
# This is exactly what PyTorch later produces.
# What does the gradient value $42$ mean?
# Students often treat gradients as symbolic objects.
# I would ask:
#
# “What does $42$ actually tell us?”
# A good answer:
#
# “Near $\theta=2$, increasing $\theta$ slightly increases the loss at approximately $42$ units of loss per unit change in $\theta$.”
# More formally,
#
#
# $$L(\theta+\Delta\theta) \approx L(\theta) + \frac{dL}{d\theta}\Delta\theta.$$
#
#
# So if
#
#
# $$\Delta\theta=0.01,$$
#
#
# then approximately
#
# $$\Delta L \approx 42(0.01) = 0.42.$$
#
#
# That makes the gradient physically interpretable as local sensitivity.
# Slide 17: The computational graph
# The graph is
#
#
# $$\theta \rightarrow u=3\theta+1 \rightarrow L=u^2.$$
#
#
# The deck explains that the forward pass builds a directed acyclic graph and the backward pass multiplies local derivatives using the chain rule.
# A useful way to visualize the reverse pass is:
# Start with
#
#
# $$\frac{dL}{dL}=1.$$
#
#
# At the $L=u^2$ node:
#
#
# $$\frac{dL}{du}=2u.$$
#
#
# At the $u=3\theta+1$ node:
#
#
# $$\frac{du}{d\theta}=3.$$
#
#
# Multiply:
#
# $$\frac{dL}{d\theta} = \frac{dL}{du}\frac{du}{d\theta} = 2u\cdot3.$$
#
#
# Autograd is not inventing a new calculus rule.
# It is automating repeated application of the chain rule.
# Why is the graph a DAG?
# DAG means directed acyclic graph.
# Directed because computation flows from inputs to outputs.
# Acyclic because, within one forward evaluation, a quantity cannot depend on itself through a closed loop.
# For example,
#
#
# $$\theta \rightarrow u \rightarrow L$$
#
#
# contains no cycle.
# This matters because the backward traversal can process dependencies in reverse topological order.
# Slide 18: Why reverse-mode autodiff is ideal for ML
# The deck states the critical reason:
#
# Many parameters, one scalar loss.
# Suppose
#
# $$\theta = \begin{bmatrix} \theta_1 & \theta_2 & \cdots & \theta_P \end{bmatrix}^{T}$$
#
#
# with
#
#
# $$P=10^6.$$
#
#
# The loss is scalar:
#
#
# $$L\in\mathbb{R}.$$
#
#
# We want
#
# $$\nabla_\theta L = \begin{bmatrix} \frac{\partial L}{\partial\theta_1} \\ \frac{\partial L}{\partial\theta_2} \\ \vdots \\ \frac{\partial L}{\partial\theta_P} \end{bmatrix}.$$
#
#
# Reverse mode computes all of these efficiently in one backward traversal.
# That is exactly the problem structure encountered in neural networks and learnable DSP systems.
# Slide 19: A vectorized linear model
# The deck uses
#
# $$\hat{y} = wx+b$$
#
#
# with mean-square loss.
# Suppose the target is
#
#
# $$y_i=2.5x_i-0.4.$$
#
#
# The model begins with
#
#
# $$w=0.2, \qquad b=0.$$
#
#
# Predictions are
#
# $$\hat{y}_i = wx_i+b.$$
#
#
# The loss is
#
# $$\boxed{ L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2 }$$
#
#
# PyTorch computes both
#
#
# $$\frac{\partial L}{\partial w}$$
#
#
# and
#
#
# $$\frac{\partial L}{\partial b}$$
#
#
# from the same backward pass.
# A useful hand derivation is
#
# $$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)x_i$$
#
#
# and
#
# $$\frac{\partial L}{\partial b} = \frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i).$$
#
#
# This starts looking very similar to the FIR gradient later.
# Slide 20: Understanding the minimal autograd API
# The deck's minimal example is:
#
# ```python
# theta = torch.tensor(2.0, requires_grad=True)
# u = 3*theta + 1
# loss = u**2
# loss.backward()
# print(theta.grad)
#
# ```
#
# which produces
#
#
# $$42.$$
#
#
# The conceptual role of each line is:
# `requires_grad=True`
# means
#
#
# $$\text{track operations downstream from } \theta.$$
#
#
# `loss.backward()`
# means
#
#
# $$\text{compute } \frac{\partial L}{\partial(\cdot)}$$
#
#
# for all relevant upstream leaf tensors.
# `theta.grad`
# stores
#
#
# $$\frac{\partial L}{\partial\theta}.$$
#
#
# Why does `.grad` usually live on leaf tensors?
# A leaf tensor is typically a parameter created directly by the programmer, such as
#
# ```python
# theta = torch.tensor(..., requires_grad=True)
#
# ```
#
# The intermediate value
#
#
# $$u=3\theta+1$$
#
#
# is temporary.
# Autograd needs $u$'s information for the backward pass, but storing a persistent `.grad` tensor for every intermediate would consume unnecessary memory.
# A good answer to the slide question is:
#
# “We normally care about gradients of the parameters we intend to update. Intermediate gradients are used internally during backpropagation but are not all retained as user-visible `.grad` fields by default.”
# Slide 21: Gradient accumulation
# This is one of the most common PyTorch surprises.
# The deck demonstrates:
#
#
# $$L=x^2$$
#
#
# at
#
#
# $$x=3.$$
#
#
# Then
#
#
# $$\frac{dL}{dx}=2x=6.$$
#
#
# After the first `backward()`,
#
#
# $$x.\text{grad}=6.$$
#
#
# After a second `backward()` without clearing,
#
#
# $$x.\text{grad}=12.$$
#
#
# The reason is:
#
#
# $$\boxed{ \text{PyTorch accumulates gradients} }$$
#
#
# rather than replacing them.
# This behavior is mathematically useful.
# If a parameter influences the loss through several paths,
#
#
# $$L=L_1+L_2,$$
#
#
# then
#
# $$\frac{\partial L}{\partial\theta} = \frac{\partial L_1}{\partial\theta} + \frac{\partial L_2}{\partial\theta}.$$
#
#
# So summing gradient contributions is exactly what the chain rule requires.
# The danger is accidentally accumulating across training iterations.
# Typical training loop
# The deck gives the standard pattern:
#
# ```python
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
#
# ```
#
# Conceptually:
#
#
# $$\text{clear old gradient}$$
#
# $$\text{compute current gradient}$$
#
# $$\text{update parameters}.$$
#
#
# If you skip the clearing step, the parameter update can use
#
#
# $$g_{\text{old}}+g_{\text{new}}$$
#
#
# instead of just
#
#
# $$g_{\text{new}}.$$
#
#
# Slide 22: `no_grad()` versus `detach()`
# These are related but different.
#
# `torch.no_grad()`
# This is a context manager.
# Operations executed inside it do not create autograd history.
# For example:
#
# ```python
# with torch.no_grad():
#     y = model(x)
#
# ```
#
# This is useful during inference because we do not need gradients.
# It saves memory and computation.
# `.detach()`
# Suppose
#
#
# $$y$$
#
#
# already belongs to a graph.
# Then
#
# ```python
# z = y.detach()
#
# ```
#
# creates a tensor with the same numerical values but disconnected from that graph.
# So gradients through $z$ do not flow backward to the operations that created $y$.
# A good use is:
#
# ```python
# plot_value = y_hat.detach().cpu().numpy()
#
# ```
#
# because plotting does not need gradient history.
# Good answer: What is the difference?
# “`no_grad()` prevents graph construction for operations executed inside a region. `detach()` takes an existing tensor and creates a view of its values that no longer participates in the current gradient graph.”
# That is a very solid answer.
# Slide 23: FIR filtering as a learnable model
# This is the strongest DSP connection in the lecture.
# A causal FIR filter is
#
# $$\boxed{ \hat{y}[n] = \sum_{k=0}^{M-1} h[k]x[n-k] }$$
#
#
# where the tap coefficients
#
#
# $$h[k]$$
#
#
# are now treated as unknown parameters.
# Suppose we know a target output
#
#
# $$y[n].$$
#
#
# Define the error
#
# $$e[n] = \hat{y}[n]-y[n].$$
#
#
# Use mean-square error:
#
# $$\boxed{ L = \frac{1}{N}\sum_n e^2[n] }$$
#
#
# The deck points out that the resulting gradient is correlation-like.
# Let's derive it.
# Deriving the FIR gradient
# We want
#
#
# $$\frac{\partial L}{\partial h[r]}.$$
#
#
# Start with
#
# $$L = \frac{1}{N}\sum_n (\hat{y}[n]-y[n])^2.$$
#
#
# Then
#
# $$\frac{\partial L}{\partial h[r]} = \frac{1}{N}\sum_n 2(\hat{y}[n]-y[n])\frac{\partial\hat{y}[n]}{\partial h[r]}.$$
#
#
# Since
#
# $$\hat{y}[n] = \sum_k h[k]x[n-k],$$
#
#
# we have
#
# $$\frac{\partial\hat{y}[n]}{\partial h[r]} = x[n-r].$$
#
#
# Therefore,
#
# $$\boxed{ \frac{\partial L}{\partial h[r]} = \frac{2}{N}\sum_n e[n]x[n-r] }$$
#
#
# This is a beautiful result for DSP students.
# It says:
#
# The gradient for tap $r$ is obtained by correlating the current output error with the input delayed by $r$ samples.
# That should feel very familiar to anyone who has seen adaptive filtering.
# Connection to LMS adaptive filtering
# This is a particularly valuable extension.
# The LMS update has the general form
#
# $$h_r[m+1] = h_r[m] + \mu e[m]x[m-r]$$
#
#
# depending on sign convention.
# Gradient descent has the generic form
#
# $$h_r^{(m+1)} = h_r^{(m)} - \eta\frac{\partial L}{\partial h[r]}.$$
#
#
# These have essentially the same structure:
#
#
# $$\boxed{ \text{coefficient update} \propto \text{error} \times \text{input}$$
#
#
# So modern autograd is not disconnected from classical adaptive signal processing.
# It generalizes the same gradient idea to arbitrarily complicated differentiable systems.
# This is one of the best conceptual bridges in the entire lecture.
# Important conv1d convention
# The slide correctly warns that `torch.nn.functional.conv1d` implements cross-correlation in the deep-learning convention.
# Textbook discrete-time convolution is
#
# $$y[n] = \sum_k h[k]x[n-k].$$
#
#
# Deep-learning “convolution” often computes something closer to
#
# $$y[n] = \sum_k w[k]x[n+k]$$
#
#
# depending on indexing and padding conventions.
# So if you want exact textbook convolution ordering, you may need to flip the kernel.
# A good student observation is:
#
# “The operation is still linear and shift structured, but the coefficient indexing convention differs.”
# Exactly.
# Slide 24: System identification by gradient descent
# The deck's FIR demo has target taps
#
#
# $$\boxed{ [0.15, \; 0.50, \; 0.25, \; -0.10] }$$
#
#
# and the learned taps converge to nearly the same values.
# This is a simple system-identification problem.
# Imagine that the actual system is
#
#
# $$h_{\text{true}}[k].$$
#
#
# We feed a known input
#
#
# $$x[n]$$
#
#
# through the system and observe
#
#
# $$y[n].$$
#
#
# Our model has unknown coefficients
#
#
# $$h[k].$$
#
#
# We calculate
#
#
# $$\hat{y}[n]$$
#
#
# and the loss
#
#
# $$L.$$
#
#
# Autograd calculates
#
#
# $$\nabla_h L.$$
#
#
# Then gradient descent updates
#
# $$\boxed{ h^{(m+1)} = h^{(m)} - \eta\nabla_h L }$$
#
#
# until the modeled filter matches the target system.
# The key teaching statement from the slide is excellent:
#
# Autograd did not “understand filters.”
# It simply differentiated the operations you programmed.
# That is exactly what students need to understand before neural networks arrive.
# A useful question: Is this “machine learning”?
# Yes.
# The model contains parameters
#
#
# $$h[k]$$
#
#
# that are learned from data by minimizing an objective.
# So even though the model is “just an FIR filter,” it satisfies the core ML pattern:
#
#
# $$\boxed{ \text{parameterized model} + \text{data} + \text{loss} + \text{optimization} }$$
#
#
# This helps demystify neural networks.
# Slide 25: Common failure modes
# The deck emphasizes that many early PyTorch mistakes are bookkeeping mistakes.
# These are worth turning into diagnosis questions.
#
# Wrong axis order
# Expected:
#
#
# $$(N,C,L)$$
#
#
# received:
#
#
# $$(N,L,C).$$
#
#
# Good diagnosis:
#
# “The numerical data are present, but the layer is interpreting time samples as channels.”
# Mixed device
# Input on CPU:
#
#
# $$x \in \text{CPU memory}$$
#
#
# parameters on GPU:
#
#
# $$\theta \in \text{GPU memory}.$$
#
#
# The operation cannot proceed because the operands are not in compatible memory spaces.
# Gradient accumulation
# Symptom:
# Training becomes unstable or parameter updates become unexpectedly large.
# Likely cause:
# Forgot
#
# ```python
# optimizer.zero_grad()
#
# ```
#
# Detached graph
# Symptom:
# A parameter's gradient is `None` even though you expect it to affect the loss.
# Likely cause:
# An intermediate tensor was detached or created under `no_grad()`.
# In-place operation
# An in-place modification can overwrite a value autograd needs for the backward pass.
# While learning, functional-style code is safer:
#
# ```python
# y = x + 1
#
# ```
#
# rather than aggressively modifying tensors in place.
# Homework 1: Tensor shapes
# The deck asks students to assign sensible shapes.
# Good answers are:
#
# One mono clip, $16000$ samples
#
#
# $$\boxed{(16000,)}$$
#
#
# rank:
#
#
# $$\boxed{1}$$
#
#
# $32$ stereo clips
#
#
# $$\boxed{(32,2,16000)}$$
#
#
# rank:
#
#
# $$\boxed{3}$$
#
#
# One STFT
#
#
# $$\boxed{(257,101)}$$
#
#
# possibly complex.
# Rank:
#
#
# $$\boxed{2}$$
#
#
# Batch of $32$ one-channel spectrograms
#
#
# $$\boxed{(32,1,257,101)}$$
#
#
# rank:
#
#
# $$\boxed{4}$$
#
#
# The important thing is not just the numbers. Students should state what each axis means.
# Homework 2: Broadcasting gains
# Given
#
#
# $$\text{x.shape} = (24,2,8000),$$
#
#
# we want
#
#
# $$g_L=0.5$$
#
#
# and
#
#
# $$g_R=1.2.$$
#
#
# Use
#
#
# $$\boxed{ \text{g.shape} = (1,2,1) }$$
#
#
# with
#
# ```python
# g = torch.tensor([0.5, 1.2]).view(1,2,1)
# y = x * g
#
# ```
#
# The result has shape
#
#
# $$\boxed{(24,2,8000)}.$$
#
#
# Homework 3: Filterbank multiplication
# Given
#
#
# $$\mathbf{P} \in \mathbb{R}^{257\times100}$$
#
#
# and
#
#
# $$\mathbf{W} \in \mathbb{R}^{40\times257},$$
#
#
# compute
#
# $$\boxed{ \mathbf{E} = \mathbf{W}\mathbf{P} }$$
#
#
# with
#
#
# $$\mathbf{E} \in \mathbb{R}^{40\times100}.$$
#
#
# Interpretation:
# $40$ filterbank bands by
# $100$ time frames.
# The solution in the appendix gives exactly this interpretation.
# Homework 4: Chain rule
# The problem uses
#
#
# $$u=2\theta-3$$
#
#
# and
#
#
# $$L=(u+1)^2.$$
#
#
# First simplify:
#
# $$u+1 = 2\theta-2.$$
#
#
# Then
#
#
# $$L=(2\theta-2)^2.$$
#
#
# By the chain rule,
#
# $$\frac{dL}{d\theta} = 2(2\theta-2)\cdot2.$$
#
#
# Therefore,
#
# $$\boxed{ \frac{dL}{d\theta} = 8\theta-8 }$$
#
#
# At
#
#
# $$\theta=4,$$
#
# $$\frac{dL}{d\theta} = 32-8 = \boxed{24}.$$
#
#
# The instructor appendix confirms this result.
# Homework 5: Accumulation
# For
#
#
# $$L=x^2$$
#
#
# at
#
#
# $$x=3,$$
#
#
# the derivative is
#
# $$\frac{dL}{dx} = 2x = 6.$$
#
#
# First backward:
#
#
# $$\boxed{6}$$
#
#
# Second backward without clearing:
#
#
# $$\boxed{12}$$
#
#
# because PyTorch accumulates gradients.
# The solution appendix states exactly this.
# Homework 6: FIR gradient interpretation
# The key result is
#
# $$\boxed{ \frac{\partial L}{\partial h[r]} = \frac{2}{N}\sum_n e[n]x[n-r]. }$$
#
#
# The strongest interpretation is:
#
# “The gradient for each tap is proportional to the correlation between the current output error and the input sequence delayed to that tap.”
# That is the DSP meaning of backpropagation through an FIR filter. The deck's solution appendix makes the same point.
# Weekly quiz: Good answers
# The quiz asks several very good conceptual questions.
#
# Q1
# $16$ mono clips, each $4000$ samples, using $(N,C,L)$:
#
#
# $$\boxed{(16,1,4000)}$$
#
#
# Q2
# For input shape
#
#
# $$(8,2,1000),$$
#
#
# one independent gain per channel should have shape
#
#
# $$\boxed{(1,2,1)}.$$
#
#
# Q3
# Every PyTorch tensor automatically tracks gradients.
# Answer:
#
#
# $$\boxed{\text{False}}$$
#
#
# A floating-point or complex tensor typically tracks gradients only when
#
# `requires_grad=True`
# or when it is downstream from such a tensor.
# Q5
# Why is reverse-mode autodiff appropriate for modern ML?
# Good answer:
#
# “Because one scalar loss depends on many parameters. Reverse mode computes all components of $\nabla_\theta L$ efficiently in a single backward traversal.”
# That matches the instructor solution.
# Q6
# What happens if gradients are never cleared?
# Good answer:
#
# “They accumulate across iterations, so later updates use the sum of old and new gradients rather than just the current gradient.”
# Again, exactly the intended concept.
# The most important conceptual connection to DSP
# I would make this connection very explicit near the end of the lecture.
# Traditional DSP often looks like
#
#
# $$x[n] \rightarrow H \rightarrow y[n]$$
#
#
# where the system $H$ is known.
# Modern ML often looks like
#
#
# $$x \rightarrow f(x;\theta) \rightarrow \hat{y}$$
#
#
# where the system contains unknown parameters
#
#
# $$\theta.$$
#
#
# Then we define
#
#
# $$L(\hat{y},y)$$
#
#
# and compute
#
#
# $$\nabla_\theta L.$$
#
#
# So the fundamental change is not:
#
# “We stopped doing signal processing and started doing machine learning.”
# It is:
#
#
# $$\boxed{ \text{We made some signal-processing operations parameterized and differentiable.} }$$
#
#
# That is a much more coherent transition for ECE students.
# What I would expect a senior student to know
# By the end of the lecture, a strong senior should be able to explain:
# $(N,C,L)$
# and why axis order matters;
# how broadcasting works;
# why a filterbank can be written as
#
#
# $$\mathbf{E}=\mathbf{W}\mathbf{P};$$
#
#
# what
#
# `requires_grad=True`
# does;
# why
#
# `loss.backward()`
# computes gradients;
# and why gradients need to be cleared between training iterations.
# What I would expect a first-year graduate student to know
# In addition, I would want the graduate student to be able to derive:
#
# $$\frac{dL}{d\theta} = \frac{dL}{du}\frac{du}{d\theta},$$
#
# explain why reverse-mode autodiff is efficient for
#
# $$L:\mathbb{R}^{P}\rightarrow\mathbb{R},$$
#
# derive
#
# $$\frac{\partial L}{\partial h[r]} = \frac{2}{N}\sum_n e[n]x[n-r],$$
#
# and recognize that this FIR gradient has the same error-times-input structure that appears in classical adaptive filtering.
# One sentence students should leave with
# The best answer to the minute-paper question at the end of the deck would be something like:
#
# “A tensor that merely contains a signal stores numerical samples and metadata, while a tensor participating in a computational graph also retains the differentiable relationships needed to compute how a final loss changes with respect to upstream parameters.”
# That captures the entire lecture. The deck itself summarizes the same progression: tensors store familiar DSP data with shape/dtype/device metadata, tensor operations express vectorized DSP, autograd constructs a dynamic graph, and unknown DSP parameters can then be optimized from
#
# $$\nabla_{\theta}L.$$

# %% [markdown]
#

# %% [markdown]
#
