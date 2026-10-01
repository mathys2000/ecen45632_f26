# ECE Audio Signal Analysis: Supplementary Materials

This document contains the weekly problem set, a short weekly quiz (both with full solutions), and the Python boilerplate for the interactive classroom notebook mentioned in the slides.

---

## Part 1: Interactive Classroom Demo (Python / Jupyter)

*Instructor Note: Have the students run this in an empty Jupyter Notebook during Slide 13.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window
from scipy.io import wavfile

# Generate a synthetic non-stationary signal (Chirp + sudden transient)
fs = 16000 # 16 kHz sampling
t = np.linspace(0, 2, 2*fs)
# Chirp from 100 Hz to 2000 Hz
signal = np.sin(2 * np.pi * np.linspace(100, 2000, len(t)) * t) 
# Add a sudden transient at t=1.0s
signal[fs-50:fs+50] += 5.0 * np.random.randn(100) 

# Function to plot STFT dynamically
def plot_interactive_stft(window_len_ms=30, overlap_pct=50):
    nperseg = int((window_len_ms / 1000) * fs)
    noverlap = int(nperseg * (overlap_pct / 100))
    
    f, t_stft, Sxx = spectrogram(signal, fs, window='hann', 
                                 nperseg=nperseg, noverlap=noverlap)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_stft, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram: Window = {window_len_ms} ms, Overlap = {overlap_pct}%')
    plt.ylim(0, 4000)
    plt.colorbar(label='Power [dB]')
    plt.show()

# Students: Modify these parameters!
# Try window_len_ms = 5 (High time res) vs window_len_ms = 200 (High freq res)
plot_interactive_stft(window_len_ms=32, overlap_pct=75)
```

---

## Part 2: Weekly Problem Set

### Problem 1: Time and Frequency Resolution
An audio signal is sampled at $f_s = 44.1 \text{ kHz}$. You are computing an STFT using a Hann window.
1. You choose a window size of $N = 1024$ samples. What is the frequency resolution (in Hz) between adjacent bins in the DFT? What is the duration of the window in milliseconds?
2. To capture a very fast drum transient, you reduce the window size to $N = 128$ samples. What is the new frequency resolution and time duration? Explain the tradeoff in practical audio terms.

### Problem 2: Overlap and Frame Calculation
You have a 5-minute audio recording sampled at 16 kHz. You process it using an STFT with a frame size of $N = 512$ samples and a hop size of $H = 128$ samples (i.e., 75% overlap).
1. How many total frames will be generated?
2. If the resulting STFT matrix is stored as 32-bit complex floats (8 bytes per value), roughly how much memory (in MB) will the STFT matrix consume? (Assume you only store the positive frequencies, $\frac{N}{2} + 1$ bins).

### Problem 3: The Periodogram and Spectral Leakage
Let $x[n] = \cos(\frac{2\pi k_0 n}{N})$ for $n=0, 1, \dots, N-1$. 
1. If $k_0$ is an integer, sketch the periodogram $P[k]$.
2. If $k_0 = 3.5$ and $N=16$, explain conceptually why the periodogram will no longer consist of perfectly sharp spikes. What is this phenomenon called?

---

## Part 3: Solutions to Problem Set

**Solution 1:**
1. Frequency resolution: $\Delta f = \frac{f_s}{N} = \frac{44100}{1024} \approx 43.07 \text{ Hz}$. Time duration: $T = \frac{N}{f_s} = \frac{1024}{44100} \approx 23.2 \text{ ms}$.
2. Frequency resolution: $\Delta f = \frac{44100}{128} \approx 344.5 \text{ Hz}$. Time duration: $T = \frac{128}{44100} \approx 2.9 \text{ ms}$. 
*Tradeoff explanation:* By shrinking the window to 2.9 ms, we can pinpoint the drum hit precisely in time. However, our frequency bands are now smeared across ~345 Hz chunks, meaning we lose the ability to accurately resolve the pitch/harmonics of the drum or surrounding instruments.

**Solution 2:**
1. Total samples $L = 5 \times 60 \times 16000 = 4,800,000$. 
Number of frames $K = \lfloor \frac{L - N}{H} \rfloor + 1 = \lfloor \frac{4800000 - 512}{128} \rfloor + 1 = 37,497$ frames.
2. Bins per frame = $\frac{512}{2} + 1 = 257$ bins. 
Total elements = $257 \times 37,497 = 9,636,729$. 
Memory = $9,636,729 \times 8 \text{ bytes} \approx 77,093,832 \text{ bytes} \approx 77.1 \text{ MB}$.

**Solution 3:**
1. Because $k_0$ is an integer, the frequency aligns perfectly with the DFT bins. The periodogram will have sharp spikes exactly at bin $k_0$ and $N - k_0$. All other bins will be exactly 0.
2. Because 3.5 is not an integer, the frequency falls *between* the discrete DFT bins. Because we applied a finite rectangular window of length $N=16$, the energy will "smear" or leak into adjacent bins following a Sinc function shape. This is called **Spectral Leakage**.

---

## Part 4: Weekly Quiz (10 Minutes)

**Question 1:** Why is the standard Discrete-Time Fourier Transform (DTFT) computationally impossible to implement directly on a computer array for a 3-minute audio file? (Provide two reasons).

**Question 2:** In the Short-Time Fourier Transform (STFT), what is the purpose of applying a "window function" (like Hann or Hamming) to each frame before computing the DFT?

**Question 3:** You compute an STFT with a window size of $N=1024$ and a hop size of $H=1024$. Upon inverting the STFT (resynthesizing the audio), you hear a rapid "pulsing" or "clicking" distortion. What parameter caused this, and how should you fix it?

---

## Part 5: Quiz Solutions

**Answer 1:** 1) The DTFT expects an infinitely long signal (from $-\infty$ to $\infty$), but arrays are finite. 2) The DTFT outputs a continuous frequency spectrum, which cannot be stored in discrete computer memory (this necessitates sampling the frequency domain, leading to the DFT).

**Answer 2:** Extracting a frame is equivalent to multiplying the signal by a rectangular window, which creates sharp, artificial discontinuities at the edges of the frame. This causes severe spectral leakage. Window functions (like Hann) taper the edges to zero, smoothing out these boundaries and reducing high-frequency leakage.

**Answer 3:** A hop size equal to the window size ($H=N$) means there is **0% overlap**. Because window functions taper the edges of the frame to zero, a 0% overlap causes the signal's energy to dip to near-zero at the boundary of every frame, creating an amplitude modulation (a "pulsing" artifact). The fix is to reduce the hop size (e.g., $H = 512$ or $H = 256$) to create overlapping frames that smoothly cross-fade.

