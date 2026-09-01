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
# # Lecture 2 Demo
#
# Eigenfunctions and the DTFT

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# %%
# FIR impulse response
h = np.array([0.25, 0.5, 0.25])
omega0 = 0.4*np.pi
n = np.arange(200)
x = np.exp(1j*omega0*n)


# %%
# For an FIR h[k], H(e^jw) is a finite sum.
k = np.arange(len(h))
H0 = np.sum(h * np.exp(-1j*omega0*k))

# Ignore startup/transient effects by using direct finite convolution and compare where all taps overlap.
y = np.convolve(x, h, mode='full')
y_pred = H0 * np.exp(1j*omega0*np.arange(len(y)))

# The eigenfunction relation is exact for an infinite-duration exponential.
# With our finite numerical record, compare the interior samples unaffected by record boundaries.
interior = slice(len(h)-1, len(x))
error = np.max(np.abs(y[interior] - y_pred[interior]))

print('H(e^jw0) =', H0)
print('|H|       =', abs(H0))
print('phase     =', np.angle(H0), 'rad')
print('max interior error =', error)


# %%
def dtft(x, n_index=None, omega=None):
    x = np.asarray(x)
    if n_index is None:
        n_index = np.arange(len(x))
    if omega is None:
        omega = np.linspace(-np.pi, np.pi, 2048, endpoint=False)
    E = np.exp(-1j*np.outer(omega, n_index))
    return omega, E @ x

x = np.array([1.0, 2.0, -1.0])
omega, X = dtft(x)

fig = plt.figure(figsize=(9, 3))
plt.plot(omega/np.pi, np.abs(X))
plt.xlabel(r'$\omega/\pi$')
plt.ylabel(r'$|X(e^{j\omega})|$')
plt.title(f'DTFT magnitude over one 2$\\pi$ period, x={x}')
plt.grid(True, alpha=0.25)
plt.show()


# %%
# Verify periodicity numerically at arbitrary points.
probe = np.array([-0.8, 0.2, 1.4])
_, X1 = dtft(x, omega=probe)
_, X2 = dtft(x, omega=probe + 2*np.pi)
print('max periodicity error:', np.max(np.abs(X1-X2)))


# %%
# Magnitude frequency response as pole moves toward 1
fig = plt.figure(figsize=(9, 4))
for a in [0.5, 0.9, 0.99]:
    w, H = signal.freqz([1.0], [1.0, -a], worN=2048)
    plt.plot(w/np.pi, np.abs(H), label=f'a={a}')
plt.xlabel(r'$\omega/\pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.title('One-pole frequency response as the pole moves toward 1')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()


# %%
# Parseval
x = np.array([1.0, -2.0, 0.5, 1.25])
omega = np.linspace(-np.pi, np.pi, 200_000, endpoint=False)
_, X = dtft(x, omega=omega)

E_time = np.sum(np.abs(x)**2)
E_freq = np.mean(np.abs(X)**2)  # mean over one 2*pi period equals (1/(2*pi)) integral

print('time-domain energy      =', E_time)
print('frequency-domain approx =', E_freq)
print('absolute error          =', abs(E_time-E_freq))


# %%
# Simple edge/change detector on a 1-D signal
x = np.r_[np.zeros(20), np.ones(25), 0.3*np.ones(20), np.zeros(20)]
h = np.array([1.0, -1.0])
feature = np.convolve(x, h, mode='same')

fig = plt.figure(figsize=(9, 3))
plt.plot(x, lw=2.5, label='signal')
plt.plot(feature, ls='--', label='change-detector response')
plt.xlabel('n')
plt.title(f'A fixed convolution kernel as a feature detector, h={h}')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()



# %%
