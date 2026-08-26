Week 13 — Lecture 25: Tensors and Automatic Differentiation
DSP + Machine Learning for senior / first-year graduate ECE

FILES
-----
Week13_Lecture25_Tensors_Autograd.pptx
    36-slide PowerPoint. Slides 1–30 are lecture/homework/quiz material.
    Slides 31–36 are clearly marked instructor solutions.

notebooks/Lecture25_Tensors_Autograd.ipynb
    Student/instructor notebook with tensor, STFT, broadcasting, matrix
    multiplication, autograd, and learnable-FIR demonstrations.

notebooks/Lecture25_Tensors_Autograd_executed.ipynb
    Executed copy used to verify that the notebook runs without errors.

lecture25_tensors_autograd.py
    Standalone Python version of the core demonstrations.

Lecture25_Homework_Quiz_Solutions.md
    Seven homework problems and a six-question weekly quiz, all with solutions.

TEACHING FLOW
-------------
0–20 min   From vectors to tensors
20–40 min  Tensor operations for DSP
40–75 min  Chain rule and torch.autograd

The main DSP-native autograd example treats FIR taps h[k] as learnable
parameters. A synthetic target filter [0.15, 0.50, 0.25, -0.10] is recovered
by gradient descent using an MSE signal-domain objective.

REQUIREMENTS
------------
Python 3
numpy
matplotlib
torch

The demonstrations are CPU-safe; a GPU is not required.
