# Time Frequency Project Worked Solution

Instructor version. This solution uses a local index FFT convention, periodic windows, explicit padding and normalized windowed overlap addition. Exact FFT block convolution is implemented separately. The code and notebook provide every algorithm and regenerate all reference results; numerical tables below were measured from that implementation.

## Time frequency interpretation

| L | Window ms | Bin Hz | Hop ms | Transient width ms |
| --- | --- | --- | --- | --- |
| 256 | 16.0 | 62.5 | 4.0 | 8.0 |
| 1024 | 64.0 | 15.625 | 16.0 | 32.0 |
| 4096 | 256.0 | 3.90625 | 64.0 | 192.0 |

The transient width uses the isolated supplied 5 ms pulse: sum squared STFT magnitudes over frequency, retain frame centers above 10 percent of peak frame energy, then multiply the number of retained frames by hop duration. This is a grid based spread measure, not the physical pulse duration. Students may use another clearly defined measure on the mixture; the exact values need not agree.

The 256 sample Hann main lobe is 250 Hz wide between its first nulls, so 80 Hz spaced tones are difficult to resolve. At 1024 samples that width is 62.5 Hz and separation improves. The 4096 sample window narrows stationary tones further but spreads the transient and integrates a longer section of the chirp. Increasing N from 1024 to 4096 at fixed L=1024 samples the same windowed DTFT more densely: it changes bin spacing from 15.625 to 3.90625 Hz without narrowing its main lobe.

## Window measurements at 16000 samples per second

| Window | Main lobe Hz | Peak sidelobe dB | ENBW Hz |
| --- | --- | --- | --- |
| boxcar | 31.25 | -13.26 | 15.62 |
| hann | 62.50 | -31.47 | 23.44 |
| hamming | 62.50 | -42.67 | 21.29 |
| blackman | 93.75 | -58.11 | 26.98 |

Main lobe width is twice the frequency of the first local minimum of the densely sampled window transform; sidelobe amplitude is the maximum beyond that minimum, relative to DC. The rectangular window gives narrow lobes but substantial leakage. Hamming has a lower highest sidelobe than Hann here; Blackman suppresses it further while widening the main lobe. The 1140 Hz tone is about 32.7 dB below the 1003 Hz tone. Rectangular leakage can obscure it; tapered windows improve its visibility in this example. Coherent gain correction permits a fair amplitude comparison, but does not undo off bin scalloping.

# Reconstruction and FFT Convolution

## Derivation of normalized synthesis

$$
v_m[r]=\operatorname{IFFT}_N\{X_m\}[r]=w[r]x[mH+r]
$$

$$
\sum_m w[n-mH]v_m[n-mH]=x[n]\sum_m w^2[n-mH]
$$

Divide the accumulated numerator by its denominator on each retained sample. Padding before sample zero ensures that endpoints are covered by nonzero portions of windows; crop back to the original length afterward. The analysis code adds L leading and L trailing zeros and enough extra zeros to complete the last frame. For N greater than L, the inverse transform is truncated to its first L samples. Keep complex values for I/Q.

| Overlap percent | Hop | Frames | Denominator range | Error |
| --- | --- | --- | --- | --- |
| 0 | 1024 | 65 | 1.0000 to 1.0000 | 2.5e-16 |
| 0 | 1024 | 65 | failure | NOLA |
| 25.0 | 768 | 86 | 0.0429 to 1.0000 | 3.5e-16 |
| 50.0 | 512 | 128 | 0.5000 to 1.0000 | 2.3e-16 |
| 75.0 | 256 | 255 | 1.5000 to 1.5000 | 1.9e-16 |
| 87.5 | 128 | 509 | 3.0000 to 3.0000 | 1.8e-16 |

The first row uses a rectangular window; all other rows use Hann. At 50 percent overlap, the squared Hann sum varies from 0.5 to 1; the ordinary Hann sum being constant does not make the squared sum constant. At 75 percent it is 1.5. Normalized synthesis works for either case. The 25 percent case is invertible but its smaller denominator can amplify numerical or processing perturbations. More overlap creates more frames and smoother sampling of time varying operations, rather than improving the window’s inherent frequency resolution.

# Exact FFT Overlap Add Filtering

## Exact linear FIR convolution

Split input into disjoint blocks, transform each zero padded block and the FIR, multiply, inverse transform, retain block length plus M minus 1 samples and add at the block start. Disjoint inputs create overlapping convolution tails. No STFT windows are needed. With B=512 and M=129, the minimum FFT size is 640; the solution uses 1024. The output has 64128 samples and matches direct convolution to relative error 2.81e-16. The group delay is 64 samples, or 4 ms.

| Input block B | FFT size | Block count | Relative error |
| --- | --- | --- | --- |
| 128 | 256 | 500 | 3.02e-16 |
| 512 | 1024 | 125 | 2.81e-16 |
| 2048 | 4096 | 32 | 3.13e-16 |

The final 2048 sample block is partial. The implementation keeps its correct output length and the complete tail. A shorter FFT would fold a block’s linear convolution around the FFT period. Multiplying STFT frames by an arbitrary frequency mask and cropping to L is generally a time varying operation, so it should not be claimed to equal this exact FIR convolution.

# Moving Jammer Solution and Measured Performance

Estimate the largest positive frequency magnitude within the declared search band for each frame and apply a five frame median filter. Use the estimated ridge to create a Gaussian gain dip with minimum gain 0.03. Applying the same real gain at both signs of frequency preserves Hermitian symmetry and permits a real output. The code uses the mixture only for ridge estimation.

$$
Y_m[k]=G_m[k]X_m[k]
$$

The mask changes amplitude without replacing the phase of surviving coefficients. A smooth dip is preferable to a hard bin cutoff because the latter tends to create longer oscillatory artifacts. Boundary zero frames have arbitrary peak estimates but do not contribute to retained output. The baseline uses L=1024, hop 256 and width 60 Hz.

Baseline input SNR is -9.24 dB and output SNR is 17.35 dB, an improvement of 26.59 dB. Jammer attenuation is 26.59 dB; desired waveform relative error is 0.0404 percent. The stationary notch gives only -8.48 dB output SNR.

| L | Width Hz | Output SNR dB | Jammer attenuation dB | Desired error percent |
| --- | --- | --- | --- | --- |
| 256 | 30 | -0.80 | 8.44 | 0.0243 |
| 256 | 60 | 4.84 | 14.08 | 0.0378 |
| 256 | 120 | 12.48 | 21.72 | 0.0566 |
| 1024 | 30 | 11.77 | 21.01 | 0.0281 |
| 1024 | 60 | 17.35 | 26.59 | 0.0404 |
| 1024 | 120 | 19.88 | 29.12 | 0.0575 |
| 4096 | 30 | 11.35 | 20.59 | 0.0278 |
| 4096 | 60 | 17.08 | 26.32 | 0.0403 |
| 4096 | 120 | 19.77 | 29.01 | 0.0575 |

$$
{\rm SNR}=10\log_{10}\frac{\|s\|_2^2}{\|y-s\|_2^2}
$$

Freeze the mixture estimated gains before applying them separately to clean sources. This makes attenuation and distortion meaningful; reestimating the mask on clean sources would change the operator. The baseline has little desired loss because the desired melody is well separated from the search band. Speech may lose useful energy in this band. A stronger performance score on this easy fixture is not evidence of universal speech or communication recovery.

# Phase Coherent QPSK Solution

The left and right WAV channels encode I and Q. Form a complex sample from both; averaging them would destroy the intended signal. Compute a two sided STFT, retain its complex coefficients, and synthesize with consistent frame sample positions. The supplied receiver uses known timing, matched root raised cosine filtering and a fixed symbol decision rule. It excludes 10 symbols at each edge and performs no carrier phase correction or gain fitting.

$$
{\rm EVM}_{\%}=100\frac{\|\widehat{a}-a\|_2}{\|a\|_2}
$$

| Processing | Waveform error | EVM percent | BER |
| --- | --- | --- | --- |
| original | 0.00e+00 | 0.948 | 0.0000 |
| coherent | 2.26e-16 | 0.948 | 0.0000 |
| magnitude only | 1.01e+00 | 101.029 | 0.4990 |
| frame phase reset | 1.28e+00 | 127.671 | 0.5227 |

Coherent reconstruction has the same 0.948 percent EVM as the original. This small baseline comes from finite pulse truncation and residual intersymbol interference, rather than STFT reconstruction. Both have zero bit errors. The float32 I/Q WAV round trip error is 2.38e-8; this is distinct from float64 reconstruction error.

![constellations](constellations.png)

Magnitude only processing loses the symbol bearing phase information. Independent frame rotations cause overlapping copies of the waveform to add with inconsistent phases. A constant global phase could be estimated by carrier recovery, but it cannot explain or correct an arbitrary rotation in every overlapping frame. These controls therefore show large EVM and BER near one half.

Keeping coefficient phases avoids an unnecessary phase reset, but unequal spectral attenuation can still change the waveform and its symbol phase through distortion. For an added communication jammer, measure post receiver EVM and BER; do not claim successful phase coherence merely because the mask is real.

# Instructor Notes and Extensions

## Reading the implementation

dsp_solution.py implements analysis, synthesis, exact overlap add and ridge masking. common.py supplies the signals, WAV I/O and known timing receiver. run_solution.py runs all experiments, writes figures and metrics.json, and checks real and complex reconstruction. Solution.ipynb is an executed teaching walkthrough. Students receive the generator and receiver but a separate student_dsp.py with four functions to complete.

## Graduate interpretation of chirp spreading

$$
\Delta f_{\rm chirp}\approx \left|\frac{df}{dt}\right|T_w
$$

The supplied chirp has sweep rate 300 Hz per second. Its change within a 16, 64 or 256 ms window is approximately 4.8, 19.2 or 76.8 Hz. Short windows also have broad window lobes: the Hann half main lobe width is approximately 2 times sample rate divided by L, or 125, 31.25 and 7.81 Hz. Very short windows therefore need wider attenuation; very long windows integrate more sweep. A useful initial width must account for both effects plus ridge error. This is a heuristic scale estimate, not an exact sum of bandwidths.

The measured width sweep supports this reasoning: at 30 Hz, L=256 performs poorly, and L=4096 no longer beats L=1024. A broad 120 Hz dip improves this separated band fixture. On speech or a jammer crossing desired components, widening the dip also removes useful energy. There is no mask only solution that can uniquely separate arbitrary coincident desired and interfering components without additional assumptions.

## Expected good student explanations

A good uncertainty answer connects observed onset smearing and tone separation to finite window duration, rather than only quoting bin spacing. A good overlap answer distinguishes temporal sampling density from intrinsic resolution and checks the denominator. A good leakage answer describes both sidelobes and main lobe width. A good jammer answer reports distortion alongside attenuation. A good phase answer explains why the complex coefficients and common sample timeline matter to symbol recovery.

## Common grading errors to avoid

Do not require a single universally optimal window. Do not punish an explicitly diagnosed Hann zero overlap failure. Do not interpret low audio amplitude as noise suppression without matched scales. Do not compare delayed FIR output with undelayed input for reconstruction error. Do not grade QPSK by spectrogram appearance. Do not require exactly matching timing measurements across computers.

## Using natural WAV examples

For speech, inspect voiced harmonics and brief consonants using roughly 20 to 40 ms initial windows, then compare longer settings. For plucked instruments, focus on attacks; for sustained instruments, examine harmonic leakage. These are starting choices to investigate, not universal prescriptions. Keep external audio optional so licensing, downloads and conversion do not consume the project week. The supplied synthetic fixtures provide the complete graded baseline.