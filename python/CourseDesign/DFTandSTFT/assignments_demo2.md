<!-- ... existing code ... -->
## Part 4: Weekly Quiz (10 Minutes)

**Question 1:** Why is the standard Discrete-Time Fourier Transform (DTFT) computationally impossible to implement directly on a computer array for a 3-minute audio file? (Provide two reasons).

**Question 2:** In the Short-Time Fourier Transform (STFT), what is the purpose of applying a "window function" (like Hann or Hamming) to each frame before computing the DFT?

**Question 3:** You compute an STFT with a window size of $N=1024$ and a hop size of $H=1024$. Upon inverting the STFT (resynthesizing the audio), you hear a rapid "pulsing" or "clicking" distortion. What parameter caused this, and how should you fix it?

---

## Part 5: Quiz Solutions

**Answer 1:** 1) The DTFT expects an infinitely long signal (from $-\infty$ to $\infty$), but arrays are finite. 2) The DTFT outputs a continuous frequency spectrum, which cannot be stored in discrete computer memory (this necessitates sampling the frequency domain, leading to the DFT).

**Answer 2:** Extracting a frame is equivalent to multiplying the signal by a rectangular window, which creates sharp, artificial discontinuities at the edges of the frame. This causes severe spectral leakage. Window functions (like Hann) taper the edges to zero, smoothing out these boundaries and reducing high-frequency leakage.

**Answer 3:** A hop size equal to the window size ($H=N$) means there is **0% overlap**. Because window functions taper the edges of the frame to zero, a 0% overlap causes the signal's energy to dip to near-zero at the boundary of every frame, creating an amplitude modulation (a "pulsing" artifact). The fix is to reduce the hop size (e.g., $H = 512$ or $H = 256$) to create overlapping frames that smoothly cross-fade.

---

## Part 6: Overlap-Add & Filterbanks (Class 2 Supplement)

### Interactive Demo (Python / Jupyter): Spectral Noise Gate

*Instructor Note: Run this demo during the "Spectral Processing" slide to show how modifying the STFT allows for noise reduction.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from IPython.display import Audio

# Generate a signal: Sine wave + White Noise
fs = 16000
t = np.linspace(0, 2, 2*fs)
clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t) # 440 Hz tone
noise = 0.2 * np.random.randn(len(t))
noisy_signal = clean_signal + noise

# 1. Forward STFT
f, t_stft, Zxx = stft(noisy_signal, fs, window='hann', nperseg=1024, noverlap=512)

# 2. Spectral Modification (Noise Gate)
# Calculate magnitude
mag = np.abs(Zxx)
# Set a threshold
threshold = 0.1 * np.max(mag)
# Create a mask: 1 if magnitude > threshold, else 0
mask = mag > threshold
# Apply mask to the complex STFT matrix
Zxx_clean = Zxx * mask

# 3. Inverse STFT (Overlap-Add)
_, reconstructed_signal = istft(Zxx_clean, fs, window='hann', nperseg=1024, noverlap=512)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Noisy Signal")
plt.plot(t[:1000], noisy_signal[:1000])
plt.subplot(2, 1, 2)
plt.title("Reconstructed Signal (Noise Gate Applied)")
plt.plot(t[:1000], reconstructed_signal[:1000])
plt.tight_layout()
plt.show()
```

### Class 2 Problem Set: Overlap-Add

**Problem 4: COLA Constraint**
You are designing an STFT/ISTFT system using a triangular window (Bartlett window) of length $N$.
1. If your hop size $H$ is $\frac{N}{2}$ (50% overlap), does this window satisfy the Constant Overlap-Add (COLA) constraint? Prove it graphically or mathematically for the constant DC amplitude.
2. If you change the hop size to $\frac{N}{4}$ (75% overlap), does the triangular window still satisfy COLA? What happens to the overall amplitude of the reconstructed signal?

### Class 2 Solutions

**Solution 4:**
1. Yes. A Bartlett window linearly ramps from 0 to 1 and back to 0. At 50% overlap, the rising edge of frame $k+1$ exactly compensates for the falling edge of frame $k$. For any sample index $n$, the sum of the overlapping windows equals 1. Thus, it satisfies COLA perfectly.
2. Yes, it still satisfies COLA, but the sum of the windows will no longer be 1. Because there are now four overlapping windows at any given point instead of two, the total amplitude will be scaled up by a factor of 2. The signal is perfectly reconstructed (no modulation distortion), but requires an overall gain reduction of $\frac{1}{2}$ to restore the original amplitude.

