# Time Frequency Processing Python Project

One week project for senior and first year graduate electrical and computer engineering students. Work in pairs or individually. Budget approximately 12 to 15 hours per student. Use NumPy, SciPy and Matplotlib; implement the four core DSP functions yourself. The supplied waveform generator and matched QPSK receiver keep the work focused on DSP rather than audio collection or modem design.

Your task is to analyze a time varying signal, remove a moving narrowband jammer, and reconstruct signals without losing phase coherence. Explain how window duration, overlap and spectral leakage change the result. Connect your reasoning to Oppenheim and Schafer topics on the DFT, windowing, FFT convolution and short time Fourier analysis. Section numbers depend on the textbook edition.

## Expected preparation

You should know complex DFT coefficients, convolution, FIR filtering and the sampling theorem. Read the relevant textbook sections before starting. No machine learning, blind synchronization or real time hardware is required.

| Day | Milestone | Hours |
| --- | --- | --- |
| 1 | Inspect signals and implement STFT analysis | 2 |
| 2 | Window and uncertainty experiments | 2.5 |
| 3 | Reassembly and overlap tests | 2 |
| 4 | FFT overlap add and jammer removal | 3.5 |
| 5 to 7 | QPSK tests and report | 3 |

## Submission

Submit one executed notebook, your student_dsp.py, a 5 to 7 page report, and three audio outputs: original mixture, cleaned mixture and filtered audio. Include requirements and a fixed random seed. Save constellation plots and numerical metrics. Identify contributions when working in pairs. Do not submit the instructor solution.

Use a common amplitude scale for audio comparisons, label axes and units, and state sample rate, window, frame length, hop and FFT length on every time frequency figure. Label magnitude displays as relative dB; do not call them calibrated PSD estimates unless you implement the proper normalization.

# Signals and Transform Conventions

All supplied test WAV files use 16000 samples per second. Float32 WAV files are deliberate; modern scientific readers support them. Use common.read_wav to handle ordinary integer PCM files and sample rate conversion. Compute in float64 or complex128.

| File | Purpose |
| --- | --- |
| tf_probe.wav | Close tones at 1000 and 1080 Hz, a chirp and a brief transient |
| leakage.wav | 1024 samples with a strong 1003 Hz off bin tone and a weak 1140 Hz tone |
| melody.wav | Desired synthetic plucked note sequence with attacks and harmonics |
| jammed.wav | Melody plus a chirp sweeping from 1800 toward 3000 Hz over four seconds |
| jammer.wav | Evaluation only; never use to estimate the ridge |
| qpsk_iq.wav | Stereo complex baseband: left channel I and right channel Q |

signals/truth.npz provides evaluation references and modem metadata. You may use clean signals to calculate scores, but the jammer removal algorithm may receive only the mixture, sample rate and declared search band. Do not hardcode the supplied chirp equation or look up the clean signal while designing the mask.

## Additional natural audio

Optional transfer test: use one 5 to 10 second speech or music excerpt. Record your own speech or instrument if convenient. Speech with vowels and consonants demonstrates both harmonic structure and transients; piano or plucked guitar exposes onset smearing. A sustained flute or violin tone is useful as a difficult desired signal that resembles a jammer. Keep this extra test brief.

External options: Open Speech Repository American English Harvard sentences, for example OSR_us_000_0010_8k.wav or OSR_us_000_0030_8k.wav. These are 8 kHz narrowband speech; resampling to 16 kHz does not restore missing bandwidth. Credit Open Speech Repository. EBU SQAM offers instrument and speech assessment excerpts; obtain material through its official page and follow its use conditions. If a source supplies FLAC, convert to WAV once before the project.

## Local index STFT definition

$$
X_m[k]=\sum_{r=0}^{L-1}x[mH+r]w[r]e^{-j2\pi kr/N}
$$

L is frame length, H is hop and N is FFT length with N at least L. Overlap fraction is 1 minus H divided by L. Use a periodic analysis window. Store complex coefficients, original signal length and padding. Frame times use the window center; subtract the initial padding offset. For I/Q use a two sided FFT and retain complex samples.

# Part A Time Frequency Analysis and Windows

## A1 Window duration and uncertainty 15 points

Implement analysis. On tf_probe.wav compare Hann windows of 256, 1024 and 4096 samples, each with 75 percent overlap and FFT length equal to L. Plot three spectrograms with identical frequency limits and dB ranges. Report frame duration, bin spacing and hop time. Inspect the two close tones away from transients and quantify the observed transient width using an explicitly stated threshold or energy criterion.

$$
T_w=L/f_s,\qquad \Delta f_{\rm bin}=f_s/N,\qquad T_H=H/f_s
$$

Explain why long windows help distinguish close steady tones but smear short events and broaden the apparent sweep of a chirp within each frame. FFT bin spacing is not the true resolving power. At L=1024, repeat with N=4096 while keeping L and H fixed. Explain what zero padding changes and what it cannot change. Use the time frequency uncertainty idea qualitatively; a continuous time Gaussian bound should not be applied blindly to finite periodic DFT samples.

## A2 Spectral leakage 15 points

On leakage.wav compare rectangular, periodic Hann, Hamming and Blackman windows with L=1024 and N=65536. Divide amplitudes by the window sum so comparisons account for coherent gain. Plot amplitudes in dB relative to 1. Explain whether the weak neighboring tone is masked by leakage.

Also compute each window transform alone, normalized by its DC amplitude. Measure null to null main lobe width and highest sidelobe outside the main lobe. State your numerical definition of a first null or minimum. Calculate equivalent noise bandwidth from the equation below. Do not measure sidelobe levels from the two tone mixture.

$$
{\rm ENBW}=f_s\frac{\sum_r w^2[r]}{\left(\sum_r w[r]\right)^2}
$$

Discuss the tradeoff: lower sidelobes help detect weak signals near strong off bin components, while a wider main lobe reduces separation of nearby components. A window is not universally best. Optionally add a brief natural speech or music example and justify a window duration for that signal.

# Part B Reconstruction and Overlap Add Filtering

## B1 STFT reassembly 15 points

Implement synthesis with IFFT, a synthesis window, overlap summation, samplewise normalization and exact cropping. For analysis and synthesis using the same real window, use the squared window denominator. Record the minimum and maximum denominator over retained samples.

$$
\widehat{x}[n]=\frac{\sum_m w[n-mH]\,v_m[n-mH]}{\sum_m w^2[n-mH]}
$$

Here v is the first L samples of the N point IFFT of a frame. An unmodified frame equals the analysis window times the input. A nonzero denominator on every retained sample is the NOLA requirement. Include padding before and after the signal. With modified spectra, this normalized synthesis is a reconstruction operation and generally is not exact LTI convolution.

Compare Hann overlaps of 0, 25, 50, 75 and 87.5 percent at L=1024. Expect zero overlap to fail because the Hann window has zeros; detect and explain the failure instead of dividing by zero. Show a rectangular zero overlap control. For valid cases report relative reconstruction error, frame count and measured runtime. Runtime is descriptive, not a grading threshold. Float64 identity error should be below 1e-10. Report audio WAV round trip error separately.

## B2 Frequency domain FIR filtering 15 points

Implement ola_filter using nonoverlapping input blocks of B=512 samples and a 129 tap Hamming lowpass FIR with cutoff 1200 Hz. Zero pad each block, multiply its FFT by the filter FFT, inverse transform, then overlap add the full block convolution outputs. Use the supplied filter design call or an equivalent one; you are not being graded on designing an optimal lowpass.

$$
N_{\rm FFT}\geq B+M-1,\qquad N_y=N_x+M-1
$$

Verify the full output against np.convolve, including the final partial block and filter tail. Relative error should be below 1e-10. Explain why too short an FFT creates circular aliasing. Repeat for B=128 and 2048 and compare FFT sizes and frame counts. Account for the 64 sample FIR group delay when making waveform comparisons. Do not apply Hann windows or squared window normalization to this exact block convolution method.

# Part C Moving Jammer and Phase Coherence

## C1 Moving jammer removal 20 points

Use L=1024, H=256 and a periodic Hann window initially. Estimate the dominant frequency in the 1600 to 3400 Hz band from each mixture frame, then smooth or track the ridge. This prior search band makes the one week task manageable; it does not assume the exact chirp trajectory. Apply a narrow tapered attenuation mask centered on the estimate. Keep the original complex phases. For real audio apply equal gains to positive and negative frequencies.

Compare Gaussian notch widths of 30, 60 and 120 Hz, defined as the standard deviation of the gain dip, and frame lengths of 256, 1024 and 4096. Use a minimum gain of 0.03 or justify another choice. Compare with one stationary notch at 2400 Hz using the same width. Plot clean, jammed and cleaned spectrograms; listen at matched playback scale.

$$
G_m[k]=1-(1-g_{\min})\exp\left(-\frac{(|f_k|-\widehat f_m)^2}{2\sigma_f^2}\right)
$$

Measure input and output SNR against the clean melody. After estimating the mask from the mixture, freeze that mask and apply it separately to clean melody and jammer for evaluation. Report jammer attenuation and desired signal relative error separately. Target at least 15 dB SNR improvement and less than 5 percent desired signal relative error on the supplied melody. Diagnose misses; a well justified experiment can earn credit even if its first parameter choice misses these targets. Explain why a desired harmonic inside the suppressed band cannot be recovered from this mask alone.

## C2 Phase coherent communications 15 points

Use supplied 1000 symbol per second QPSK, 16 samples per symbol, root raised cosine pulses with rolloff 0.35 and a 10 symbol span. Reassemble the full I/Q signal with the unchanged STFT, then pass it to the supplied matched receiver. Compare waveform error, EVM, BER and constellation plots for the original and reconstructed signals.

Repeat with two deliberately broken methods: replace every STFT coefficient by its magnitude, and multiply each frame by an independently chosen random phase. No per frame gain normalization or phase correction is allowed. Keep the known timing and supplied matched filter. Explain why an attractive spectrogram does not demonstrate successful QPSK reconstruction. Coherent BER should be zero and EVM should match the original baseline; finite pulse truncation can give a small nonzero baseline EVM.

# Assessment and Extensions

| Area | Points | Evidence for full credit |
| --- | --- | --- |
| Time frequency tradeoff | 15 | Correct STFT 5; controlled plots and measurements 6; uncertainty and zero padding reasoning 4 |
| Windows and leakage | 15 | Correct gain and window metrics 6; weak tone experiment 5; justified tradeoffs 4 |
| Overlap reconstruction | 15 | Padding and normalization 7; overlap sweep and failure control 5; error and cost reasoning 3 |
| FFT overlap add filtering | 15 | Linear convolution algorithm and tail 8; reference checks 4; aliasing and delay explanation 3 |
| Moving jammer | 20 | Mixture only ridge and symmetric tapered mask 8; sweep and stationary control 5; attenuation versus distortion 5; limitations 2 |
| QPSK phase coherence | 15 | Complex I/Q and receiver comparison 6; broken controls and constellations 5; phase explanation 4 |
| Reproducibility and communication | 5 | Executable notebook and labeled outputs 3; concise report and contributions 2 |

Grade correctness and reasoning rather than visual polish or the number of figures. Award partial credit for a correct method with a diagnosed coding error; do not award jammer removal performance credit for a solution that reads the clean reference or hardcodes the trajectory. Identity thresholds apply to in memory float64 computations, not quantized WAV exports. A zero BER requirement applies to the supplied noiseless coherent control, not a new noisy channel.

## Graduate depth

For graduate students, require a one page deeper analysis within the same 100 point rubric: derive the squared window normalization, and estimate chirp spreading across a frame from sweep rate times frame duration. Use this estimate to interpret the width sweep. Seniors may present a correct experimental explanation without the full derivation. Grade this distinction within the explanation points already listed, rather than adding workload unrelated to the DSP objectives.

## Optional extensions

Choose at most one: make the jammer cross a desired harmonic and quantify the damage; design a ridge tracker with a slope constraint; test a phase preserving mask on QPSK with an interferer in its occupied band; compare ordinary overlap add with overlap save. Do not promise that retaining individual bin phases preserves all symbol phases after unequal spectral attenuation.

## Resources

Use your course edition of Oppenheim and Schafer. Audio source links are listed under Signals and Transform Conventions; SciPy reference links are in README.md.