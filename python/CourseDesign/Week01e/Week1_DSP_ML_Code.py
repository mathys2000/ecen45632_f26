import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.set_printoptions(precision=4, suppress=True)
print('NumPy:', np.__version__)


n = np.arange(-6, 7)
delta = (n == 0).astype(float)
step = (n >= 0).astype(float)

fig = plt.figure(figsize=(8, 3))
plt.stem(n, delta)
plt.xlabel('n')
plt.ylabel(r'$\delta[n]$')
plt.title('Unit sample')
plt.grid(True, alpha=0.25)
plt.show()

fig = plt.figure(figsize=(8, 3))
plt.stem(n, step)
plt.xlabel('n')
plt.ylabel(r'$u[n]$')
plt.title('Unit step')
plt.grid(True, alpha=0.25)
plt.show()


# A finite sequence indexed from n=-2 to n=2
n_x = np.arange(-2, 3)
x = np.array([1.0, -0.5, 2.0, 0.0, 1.5])

# Reconstruct x[n] by adding scaled shifted impulses.
reconstructed = np.zeros_like(x)
for k, xk in enumerate(x):
    reconstructed += xk * (np.arange(len(x)) == k)

print('x             =', x)
print('reconstructed =', reconstructed)
print('Exact match?  =', np.allclose(x, reconstructed))


def convolve_by_definition(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full linear convolution for 1-D finite sequences."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.zeros(len(x) + len(h) - 1)
    for n in range(len(y)):
        for k in range(len(x)):
            h_index = n - k
            if 0 <= h_index < len(h):
                y[n] += x[k] * h[h_index]
    return y

x = np.array([1.0, 2.0, 1.0])
h = np.array([1.0, -1.0])

y_manual = convolve_by_definition(x, h)
y_numpy = np.convolve(x, h, mode='full')

print('manual:', y_manual)
print('numpy :', y_numpy)
print('match :', np.allclose(y_manual, y_numpy))


fig = plt.figure(figsize=(8, 3))
plt.stem(np.arange(len(y_numpy)), y_numpy)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolution result')
plt.grid(True, alpha=0.25)
plt.show()


def test_linearity(T, x1, x2, a=1.7, b=-0.4, atol=1e-10):
    left = T(a*x1 + b*x2)
    right = a*T(x1) + b*T(x2)
    return np.allclose(left, right, atol=atol)

rng = np.random.default_rng(2)
x1 = rng.normal(size=10)
x2 = rng.normal(size=10)

T_gain = lambda x: 3*x
T_square = lambda x: x**2
T_offset = lambda x: x + 2

print('Gain system linear?   ', test_linearity(T_gain, x1, x2))
print('Square system linear? ', test_linearity(T_square, x1, x2))
print('Offset system linear? ', test_linearity(T_offset, x1, x2))


fs = 8000
N = 400
n = np.arange(N)

# Two tones plus reproducible noise
x = np.sin(2*np.pi*300*n/fs) + 0.45*np.sin(2*np.pi*2200*n/fs)
x += 0.25*np.random.default_rng(3).normal(size=N)

h = np.ones(5)/5
y = np.convolve(x, h, mode='same')

fig = plt.figure(figsize=(9, 3))
plt.plot(n[:120], x[:120], label='input')
plt.plot(n[:120], y[:120], label='5-point moving average')
plt.xlabel('n')
plt.ylabel('amplitude')
plt.title('Short segment before and after FIR smoothing')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()


try:
    import torch
    import torch.nn.functional as F

    x_t = torch.tensor([[[1., 2., 1., 0.]]])
    w_t = torch.tensor([[[1., -1.]]])
    y_corr = F.conv1d(x_t, w_t)

    x_np = x_t.numpy().ravel()
    w_np = w_t.numpy().ravel()
    # Valid cross-correlation in NumPy
    y_np_corr = np.correlate(x_np, w_np, mode='valid')
    # DSP convolution equals correlation with a flipped kernel.
    y_np_conv_valid = np.convolve(x_np, w_np, mode='valid')

    print('PyTorch conv1d (correlation):', y_corr.numpy().ravel())
    print('NumPy correlate:             ', y_np_corr)
    print('NumPy DSP convolution valid: ', y_np_conv_valid)
except ImportError:
    print('PyTorch is not installed in this environment; skip this optional cell.')


# FIR impulse response
h = np.array([0.25, 0.5, 0.25])
omega0 = 0.4*np.pi
n = np.arange(200)
x = np.exp(1j*omega0*n)

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
plt.title('DTFT magnitude over one 2π period')
plt.grid(True, alpha=0.25)
plt.show()

# Verify periodicity numerically at arbitrary points.
probe = np.array([-0.8, 0.2, 1.4])
_, X1 = dtft(x, omega=probe)
_, X2 = dtft(x, omega=probe + 2*np.pi)
print('max periodicity error:', np.max(np.abs(X1-X2)))


fig = plt.figure(figsize=(9, 4))
for a in [0.5, 0.9, 0.99]:
    w, H = signal.freqz([1.0], [1.0, -a], worN=2048)
    plt.plot(w/np.pi, np.abs(H), label=f'a={a}')
plt.xlabel(r'$\omega/\pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.title('One-pole frequency response as the pole moves toward 1')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()


x = np.array([1.0, -2.0, 0.5, 1.25])
omega = np.linspace(-np.pi, np.pi, 200_000, endpoint=False)
_, X = dtft(x, omega=omega)

E_time = np.sum(np.abs(x)**2)
E_freq = np.mean(np.abs(X)**2)  # mean over one 2π period equals (1/2π) integral

print('time-domain energy      =', E_time)
print('frequency-domain approx =', E_freq)
print('absolute error          =', abs(E_time-E_freq))


# Simple edge/change detector on a 1-D signal
x = np.r_[np.zeros(20), np.ones(25), 0.3*np.ones(20), np.zeros(20)]
h = np.array([1.0, -1.0])
feature = np.convolve(x, h, mode='same')

fig = plt.figure(figsize=(9, 3))
plt.plot(x, label='signal')
plt.plot(feature, label='change-detector response')
plt.xlabel('n')
plt.title('A fixed convolution kernel as a feature detector')
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()

