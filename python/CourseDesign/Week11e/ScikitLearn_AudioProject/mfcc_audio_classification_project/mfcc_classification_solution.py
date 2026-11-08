"""MFCC mean/std audio classification: complete self-contained scikit-learn example.

Install: pip install numpy scipy librosa scikit-learn matplotlib
Run:     python mfcc_classification_solution.py

Generates independent examples of three synthetic sound classes, writes WAV files,
extracts 13 MFCC means and 13 standard deviations (26 features), trains a
StandardScaler + SVC classifier, and saves a confusion matrix and trained model.
This demonstrates feature extraction and ML, not real-world environmental audio accuracy.
"""
from pathlib import Path
import pickle
import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

SR = 16000
DURATION = 1.0
N_MFCC = 13
N_FFT = 512
HOP = 160
SEED = 42
CLASSES = ('low_tone', 'high_tone', 'chirp')
OUT = Path('mfcc_project_output')


def synthesize(label, rng):
    """One independent, slightly varied, 1-second example."""
    t = np.arange(int(SR * DURATION)) / SR
    amplitude = rng.uniform(0.3, 0.9)
    phase = rng.uniform(0, 2 * np.pi)
    if label == 'low_tone':
        f = rng.uniform(180, 320)
        x = np.sin(2 * np.pi * f * t + phase)
    elif label == 'high_tone':
        f = rng.uniform(1200, 1700)
        x = np.sin(2 * np.pi * f * t + phase)
    elif label == 'chirp':
        f0 = rng.uniform(250, 450)
        f1 = rng.uniform(2200, 2800)
        k = (f1 - f0) / DURATION
        x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2) + phase)
    else:
        raise ValueError(label)
    # Add independently sampled sensor/background noise.
    x = amplitude * x + rng.uniform(0.015, 0.07) * rng.normal(size=len(t))
    x = np.clip(x, -1, 1)
    return x.astype(np.float32)


def make_dataset(base_dir, examples_per_class=40):
    rng = np.random.default_rng(SEED)
    rows = []
    for label in CLASSES:
        folder = base_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(examples_per_class):
            x = synthesize(label, rng)
            path = folder / f'{label}_{i:03d}.wav'
            wavfile.write(path, SR, np.round(x * 32767).astype(np.int16))
            rows.append((path, label))
    return rows


def mfcc_summary(path):
    """13 means + 13 stds = fixed-size 26D vector per recording."""
    x, sr = librosa.load(path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP, center=False
    )
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    return np.concatenate([means, stds])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_dataset(OUT / 'wav')
    X = np.vstack([mfcc_summary(path) for path, _ in rows])
    y = np.array([label for _, label in rows])
    paths = np.array([str(path) for path, _ in rows])
    print(f'Dataset: {len(X)} WAV files; X shape={X.shape}; y shape={y.shape}')
    print('One feature vector: [MFCC1 mean, ..., MFCC13 mean, MFCC1 std, ..., MFCC13 std]')

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.25, stratify=y, random_state=SEED
    )
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=3.0, gamma='scale'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print('\nClassification report:\n', classification_report(y_test, y_pred, digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap='Blues')
    plt.title('MFCC mean/std audio classifier: held-out test set')
    plt.tight_layout()
    plt.savefig(OUT / 'confusion_matrix.png', dpi=160)
    plt.close()

    # Predict one held-out file, with no retraining.
    example = Path(paths_test[0])
    predicted = model.predict(mfcc_summary(example)[None, :])[0]
    print(f'Example file: {example}')
    print(f'Actual label: {y_test[0]}; predicted: {predicted}')

    with open(OUT / 'classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'\nSaved WAVs, confusion matrix, and classifier in {OUT.resolve()}')

if __name__ == '__main__':
    main()
