Lectures 7–8 DSP + Machine Learning Instructor Package
=====================================================

Audience
-------
Senior undergraduate + first-year graduate ECE students.

Texts
-----
- Oppenheim & Schafer, Discrete-Time Signal Processing, Chapter 7 focus.
- Raschka, Machine Learning with PyTorch and Scikit-Learn, CNN / PyTorch connections.

Lecture 7 — FIR Filter Design, Windowing, & 1D Convolutions
------------------------------------------------------------
Approx. 30 slides.
Teaching arc:
1. FIR vs. IIR from the phase / ML perspective.
2. Linear phase, symmetry, and group delay.
3. Ideal lowpass derivation using the inverse DTFT.
4. Rectangular truncation and Gibbs phenomenon.
5. Window method and transition-width / sidelobe tradeoff.
6. Rectangular, Bartlett, Hann, Hamming, Blackman comparison.
7. FIR convolution as a fixed-weight PyTorch Conv1d operation.
8. Cross-correlation convention in Conv1d and tap reversal for textbook convolution.
9. Fixed DSP taps vs. learned convolution kernels.
10. Homework 1–4 + instructor solution appendix.

Lecture 8 — IIR Design, Bilinear Transformation, & Data Preprocessing
---------------------------------------------------------------------
Approx. 37 slides.
Teaching arc:
1. Butterworth, Chebyshev I/II, and elliptic analog prototypes.
2. Butterworth magnitude-squared response and order intuition.
3. Bilinear transform and stability mapping.
4. Frequency warping derivation.
5. Prewarping and a worked 0.4*pi example.
6. Explicit SciPy analog-prototype -> bilinear design path.
7. IIR nonlinear phase and group delay.
8. Forward-backward zero-phase filtering for offline ML preprocessing.
9. Why filtfilt-style processing squares magnitude, is noncausal, and can create train/deployment skew.
10. Homework 5–8 + full weekly quiz + solutions.

Code
----
lecture7_fir_windowing_conv1d.py
    Generates FIR/window figures and verifies fixed PyTorch Conv1d against textbook convolution.

lecture8_iir_bilinear_preprocessing.py
    Generates analog-prototype, bilinear/prewarp, phase, and forward-backward filtering figures.

notebooks/
    Student-facing notebooks plus executed verification copies for both lectures.

Homework and quiz
-----------------
Lectures7_8_Homework_Quiz_Solutions.md contains 8 homework problems with complete worked solutions and an 8-question weekly quiz with solutions.

PowerPoint compatibility
------------------------
The *_Microsoft_PowerPoint.pptx versions were round-tripped through python-pptx and successfully rendered after creation. Equations are embedded as LaTeX-rendered images for reliable display.

Code verification
-----------------
- Lecture 7 PyTorch/NumPy convolution mismatch: approximately 3.6e-7 in float32.
- Lecture 8 prewarped 4th-order Butterworth design: measured -3 dB cutoff approximately 0.3999*pi for target 0.4*pi.
- Executed notebooks contain no execution errors.
