"""Student starter: complete the TODO sections. Run after installing dependencies.
Usage: python mfcc_classification_student.py
"""
from pathlib import Path
import numpy as np
import librosa
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

SR = 16000
N_MFCC = 13
OUTPUT = Path('student_wav')


def make_wavs(count=40, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    rows = []
    for label in ('low_tone', 'high_tone', 'chirp'):
        folder = OUTPUT / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            a = rng.uniform(.3, .9)
            phase = rng.uniform(0, 2 * np.pi)
            if label == 'low_tone':
                f = rng.uniform(180, 320)
                x = np.sin(2 * np.pi * f * t + phase)
            elif label == 'high_tone':
                f = rng.uniform(1200, 1700)
                x = np.sin(2 * np.pi * f * t + phase)
            else:
                f0 = rng.uniform(250, 450)
                f1 = rng.uniform(2200, 2800)
                x = np.sin(2 * np.pi * (f0 * t + (f1-f0)*t*t/2) + phase)
            x = np.clip(a*x + rng.uniform(.015,.07)*rng.normal(size=len(t)), -1, 1)
            p = folder / f'{label}_{i:03}.wav'
            wavfile.write(p, SR, (x*32767).astype(np.int16))
            rows.append((p, label))
    return rows


def extract_features(path):
    x, sr = librosa.load(path, sr=SR, mono=True)
    # TODO 1: Compute 13 MFCCs using librosa.feature.mfcc
    # Use n_fft=512, hop_length=160, center=False
    # mfcc = ...

    # TODO 2: Mean and standard deviation across time frames (axis=1)
    # means = ...
    # stds = ...

    # TODO 3: Return a 26-value vector (means followed by stds)
    raise NotImplementedError('Complete TODO 1-3')


def main():
    rows = make_wavs()
    # TODO 4: Convert recordings to feature matrix X and label vector y
    # X = ...
    # y = ...

    # TODO 5: Split with stratify=y, test_size=.25, random_state=42
    # X_train, X_test, y_train, y_test = ...

    # TODO 6: Build and train StandardScaler() + SVC(kernel='rbf')
    # model = ...
    # model.fit(...)

    # TODO 7: Predict test labels; print accuracy and classification report
    raise NotImplementedError('Complete TODO 4-7')

if __name__ == '__main__':
    main()
