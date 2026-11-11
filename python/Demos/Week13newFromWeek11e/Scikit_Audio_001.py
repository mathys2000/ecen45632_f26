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
# # Scikit Audio 001
#
# Scikit Learn Audio Project. Sound classification based on MFCC means and standard deviations 

# %%
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# %%
# Parameters
SR = 16000
N_MFCC = 13
#N_MFCC = 2
MFCCsplit = False
#MFCCsplit = True
N_FFT = 512
HOP = 160
SEED = 42
CLASSES = ('low_tone', 'high_tone', 'chirp')
OUT = Path('mfcc_project_output')
#CLASSES = ('low_tone', 'high_tone', 'chirp_up', 'chirp_down')
#OUT = Path('mfcc_project_output2')
rng = np.random.default_rng(SEED)


# %%
def mfcc_summary(path, MFCCsplit=False):
    """
    N_MFCC means and N_MFCC stds = fixed size 2*N_MFCC-D vector per recording.
    """
    x, sr = librosa.load(path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP, center=False)
    if MFCCsplit is True:
        L2 = mfcc.shape[1]//2
        means = np.concatenate([np.mean(mfcc[:,:L2], axis=1), np.mean(mfcc[:,:L2], axis=1)])
        stds = np.concatenate([np.std(mfcc[:,:L2], axis=1), np.std(mfcc[:,:L2], axis=1)])
    else:
        means = np.mean(mfcc, axis=1)
        stds = np.std(mfcc, axis=1)
    return np.concatenate([means, stds])
    #return np.concatenate([means, np.zeros_like(stds)])
    #return np.concatenate([np.zeros_like(means), stds])
    


# %%
# Get paths to features and labels
rows = []
base_dir = OUT/'wav'
for label in CLASSES:
    folder = base_dir/label
    for file_path in folder.iterdir():
        if file_path.is_file():
            rows.append((file_path, label))


# %%
# Prepare train and test feature and label sets
X = np.vstack([mfcc_summary(path, MFCCsplit=MFCCsplit) for path,_ in rows])
y = np.array([label for _,label in rows])
paths = np.array([str(path) for path,_ in rows])
print(f'Dataset: {len(X)} WAV files; X shape={X.shape}; y shape={y.shape}')
#print(f'One feature vector: [MFCCmean ({N_MFCC}), MFCCstd ({N_MFCC})]')
print(f'One feature vector: [MFCCmean ({X.shape[1]//2}), MFCCstd ({X.shape[1]//2})], MFCCsplit={MFCCsplit}')

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, paths, test_size=0.25, stratify=y, random_state=SEED)


# %%
# Train model
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=3.0, gamma='scale'))
#model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=3.0, gamma='scale'))
#model = make_pipeline(SVC(kernel='linear', C=10.0, gamma='scale'))
model.fit(X_train, y_train)


# %%
# Prediction on test set
y_pred = model.predict(X_test)
print(f'\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'N_MFCC={N_MFCC}, MFCCsplit={MFCCsplit}')
print('\nClassification report:\n', classification_report(y_test, y_pred, digits=3))


# %%
# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap='Blues')
plt.title(f'MFCC mean/std audio classifier: held-out test set\nN_MFCC={N_MFCC}, MFCCsplit={MFCCsplit}')
plt.tight_layout()
plt.show()

# %%
# Predict one held-out file, with no retraining.
ix = 3
example = Path(paths_test[ix])
predicted = model.predict(mfcc_summary(example, MFCCsplit=MFCCsplit)[None, :])[0]
print(f'Example file: {example}')
print(f'Actual label: {y_test[ix]}; predicted: {predicted}')

# %%
