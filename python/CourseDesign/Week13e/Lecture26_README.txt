Week 13, Lecture 26 — MLPs for Signal Classification

Files:
- Week13_Lecture26_MLPs_Signal_Classification.pptx
- lecture26_mlp_signal_classification.py
- notebooks/Lecture26_MLP_Signal_Classification.ipynb
- notebooks/Lecture26_MLP_Signal_Classification_executed.ipynb
- Lecture26_Homework_Quiz_Solutions.md

Teaching flow (75 minutes):
0–20 min: MLP architecture, shapes, affine maps, ReLU, why nonlinearity matters.
20–45 min: logits, softmax, cross-entropy, gradient descent, Adam.
45–75 min: the five-step PyTorch training loop, evaluation, failure modes, and a DSP-feature classification demonstration.

Dataset:
Uses /mnt/data/week12_dsp_ml/synthetic_audio_features.csv from the Week 12 package. It contains eight engineered DSP features and three synthetic classes (Cymbal, Kick, Snare). The dataset is intentionally clean for teaching; real audio usually produces more overlap.
