import torch
import torch.nn.functional as F

# 1. Define 1D signal and kernel
# Signal (X): length M = 5
signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) 
# Kernel (H): length L = 3
kernel = torch.tensor([0.5, 1.0, 0.5])          

# 2. Reshape for PyTorch conv1d: (batch_size, num_channels, signal_length)
# Both require 3D input shape: [1, 1, M] and [1, 1, L]
X = signal.view(1, 1, -1)
H = kernel.view(1, 1, -1)

# 3. DSP Note: conv1d does cross-correlation. 
# For true mathematical convolution, flip the kernel along the time axis.
H_flipped = torch.flip(H, dims=[-1])

# 4. Set Padding
# For 'full' linear convolution output size (M + L - 1), pad by (L - 1)
padding_val = H.shape[-1] - 1

# 5. Execute Convolution
# We use functional conv1d so we can pass our custom weights directly
result = F.conv1d(X, H_flipped, padding=padding_val)

# 6. Clean up output shape back to a 1D array
output = result.squeeze()

print("Signal: ", signal.tolist())
print("Kernel: ", kernel.tolist())
print("Full Convolution Result:", output.tolist())
# Expected output: [0.5, 2.0, 4.5, 7.0, 9.5, 7.0, 2.5]
