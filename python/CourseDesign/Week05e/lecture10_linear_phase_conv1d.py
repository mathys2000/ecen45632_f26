import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / 'assets'
OUT.mkdir(exist_ok=True)


def savefig(name):
    path = OUT / name
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close(); return path

# Four representative linear-phase FIR types
filters = {
    'Type I (odd N, symmetric)': np.array([0.1, 0.25, 0.3, 0.25, 0.1]),
    'Type II (even N, symmetric)': np.array([0.1, 0.4, 0.4, 0.1]),
    'Type III (odd N, antisymmetric)': np.array([0.2, 0.5, 0.0, -0.5, -0.2]),
    'Type IV (even N, antisymmetric)': np.array([0.2, 0.5, -0.5, -0.2]),
}

plt.figure(figsize=(7.4, 4.6))
for name, h in filters.items():
    w, H = signal.freqz(h, worN=2048)
    plt.plot(w/np.pi, np.abs(H), label=name)
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Magnitude')
plt.title('Representative FIR Types and Forced Endpoint Zeros')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
savefig('L10_four_types_mag.png')

# Random vs symmetric Conv1d kernel group delay
rng = np.random.default_rng(4)
h_rand = rng.normal(size=17)
h_sym = 0.5*(h_rand + h_rand[::-1])
wr, Hr = signal.freqz(h_rand, worN=2048)
ws, Hs = signal.freqz(h_sym, worN=2048)
_, gd_rand = signal.group_delay((h_rand, [1.0]), w=2048)
_, gd_sym = signal.group_delay((h_sym, [1.0]), w=2048)

plt.figure(figsize=(7.4,4.4))
plt.plot(wr/np.pi, gd_rand, label='random kernel')
plt.plot(ws/np.pi, gd_sym, label='symmetrized kernel')
plt.ylim(-30, 30)
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Group delay (samples)')
plt.title('Random Conv1d Kernels Do Not Enforce Linear Phase')
plt.legend(); plt.grid(True, alpha=0.3)
savefig('L10_random_vs_symmetric_gd.png')

# Pole-zero constellation for symmetric FIR with a complex off-unit-circle zero
r = 0.72; theta = 0.55
zeros = np.array([r*np.exp(1j*theta), r*np.exp(-1j*theta), (1/r)*np.exp(1j*theta), (1/r)*np.exp(-1j*theta)])
t = np.linspace(0, 2*np.pi, 500)
plt.figure(figsize=(5.6,5.6))
plt.plot(np.cos(t), np.sin(t), '--', linewidth=1)
plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
plt.scatter(zeros.real, zeros.imag, facecolors='none', s=100, linewidths=2)
plt.xlim(-1.7, 1.7); plt.ylim(-1.7, 1.7)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Real'); plt.ylabel('Imaginary')
plt.title('Four-Zero Constellation for Real Linear-Phase FIR')
plt.grid(True, alpha=0.3)
savefig('L10_four_zero_constellation.png')

# Demonstrate textbook FIR vs Conv1d with symmetric fixed kernel
h = signal.firwin(21, 0.25, window='hamming')
x = np.zeros(100); x[15] = 1; x[60] = -0.6
yn = np.convolve(x, h, mode='full')[:len(x)]
xt = torch.tensor(x, dtype=torch.float32).view(1,1,-1)
kt = torch.tensor(h[::-1].copy(), dtype=torch.float32).view(1,1,-1)
yt = F.conv1d(F.pad(xt, (len(h)-1,0)), kt).detach().cpu().numpy().ravel()
print('max Conv1d mismatch:', float(np.max(np.abs(yn-yt))))

# A symmetry-enforced trainable parameterization illustration
half = torch.randn(9, requires_grad=True)
center = torch.randn(1, requires_grad=True)
kernel_sym = torch.cat([half, center, torch.flip(half, dims=[0])])
assert torch.allclose(kernel_sym, torch.flip(kernel_sym, dims=[0]))
print('Symmetric learnable kernel length:', kernel_sym.numel())
print('All assets generated in', OUT)
