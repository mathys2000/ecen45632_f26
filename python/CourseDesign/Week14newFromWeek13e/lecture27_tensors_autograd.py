"""Week 14, Lecture 27: Tensors and Automatic Differentiation
DSP + ML bridge examples for senior / first-year graduate ECE.

Run with:
    python lecture27_tensors_autograd.py

Requires: numpy, matplotlib, torch
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def heading(title: str) -> None:
    print("\n" + "="*72)
    print(title)
    print("="*72)


def demo_signal_and_spectrogram() -> None:
    heading("1) DSP arrays -> PyTorch tensors")
    fs = 8000
    n = torch.arange(2000)
    t = n / fs
    x = 0.7*torch.sin(2*torch.pi*440*t) + 0.25*torch.sin(2*torch.pi*880*t)
    print("waveform shape:", tuple(x.shape), "dtype:", x.dtype, "device:", x.device)

    X = torch.stft(
        x,
        n_fft=256,
        hop_length=64,
        window=torch.hann_window(256),
        return_complex=True,
    )
    print("STFT shape (frequency, frame):", tuple(X.shape))
    print("Power spectrogram shape:", tuple(X.abs().square().shape))

    # NumPy interoperability. torch.from_numpy shares CPU memory when possible.
    a_np = np.arange(5, dtype=np.float32)
    a_t = torch.from_numpy(a_np)
    a_np[0] = 99
    print("NumPy / Torch shared-memory example:", a_t)


def demo_shapes_broadcasting_matmul() -> None:
    heading("2) Tensor operations for DSP")
    N, C, L = 4, 2, 16
    batch = torch.randn(N, C, L)
    print("batch (N,C,L):", tuple(batch.shape))

    # One gain per channel. Shape (1,C,1) broadcasts across batch and time.
    channel_gain = torch.tensor([0.5, 1.25]).view(1, C, 1)
    y = batch * channel_gain
    print("channel_gain:", tuple(channel_gain.shape), "-> output:", tuple(y.shape))

    # Dimensionality manipulation
    mono = torch.randn(N, L)             # (N,L)
    mono_ncl = mono.unsqueeze(1)         # (N,1,L)
    print("unsqueeze:", tuple(mono.shape), "->", tuple(mono_ncl.shape))

    # A filterbank is a linear transform: M = W @ P
    F_bins, frames, bands = 129, 20, 40
    P = torch.rand(F_bins, frames)        # power spectrum over time
    W = torch.rand(bands, F_bins)         # example filterbank matrix
    M = W @ P                             # (bands, frames)
    print("filterbank matmul:", tuple(W.shape), "@", tuple(P.shape), "=", tuple(M.shape))


def demo_scalar_autograd() -> None:
    heading("3) Scalar chain rule with autograd")
    theta = torch.tensor(2.0, requires_grad=True)
    u = 3*theta + 1
    loss = u**2
    loss.backward()
    print("theta =", theta.item())
    print("loss =", loss.item())
    print("autograd dL/dtheta =", theta.grad.item())
    print("hand result: dL/dtheta = 2(3theta+1)*3 = 42")


def demo_vector_gradient() -> None:
    heading("4) Vector loss: learn a line")
    x = torch.linspace(-2, 2, 41)
    y = 2.5*x - 0.4
    w = torch.tensor(0.2, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    y_hat = w*x + b
    loss = torch.mean((y_hat-y)**2)
    loss.backward()
    print(f"loss={loss.item():.4f}, dL/dw={w.grad.item():.4f}, dL/db={b.grad.item():.4f}")
    print("Gradients accumulate, so optimizers normally call zero_grad() each iteration.")


def fir_apply(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal DSP FIR convolution y[n] = sum_k h[k] x[n-k]."""
    M = h.numel()
    x_pad = F.pad(x.view(1,1,-1), (M-1, 0))
    # conv1d implements cross-correlation, so flip h for DSP convolution.
    return F.conv1d(x_pad, h.flip(0).view(1,1,-1)).view(-1)


def demo_learnable_fir() -> None:
    heading("5) DSP-native autograd: learn FIR taps")
    torch.manual_seed(7)
    x = torch.randn(512)
    h_true = torch.tensor([0.15, 0.50, 0.25, -0.10])
    with torch.no_grad():
        y_target = fir_apply(x, h_true)

    h = torch.nn.Parameter(torch.zeros_like(h_true))
    optimizer = torch.optim.SGD([h], lr=0.08)
    history = []

    for step in range(80):
        optimizer.zero_grad()          # clear accumulated gradients
        y_hat = fir_apply(x, h)        # forward pass builds the graph
        loss = torch.mean((y_hat-y_target)**2)
        loss.backward()                # reverse-mode autodiff
        optimizer.step()               # h <- h - eta * grad
        history.append(loss.item())

    print("true taps:   ", h_true.tolist())
    print("learned taps:", [round(v, 4) for v in h.detach().tolist()])
    print("final MSE:", history[-1])

    plt.figure(figsize=(6,3))
    plt.semilogy(history)
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title("Learning FIR coefficients with autograd")
    plt.tight_layout()
    plt.show()


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    demo_signal_and_spectrogram()
    demo_shapes_broadcasting_matmul()
    demo_scalar_autograd()
    demo_vector_gradient()
    demo_learnable_fir()


if __name__ == "__main__":
    main()
