# One gain per channel

audio.shape                   # (N,C,L)
channel_gain = torch.tensor([0.5, 1.25])
channel_gain = channel_gain.view(1, C, 1)
y = audio * channel_gain
