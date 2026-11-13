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
# # Week 13 Classical ML Baseline
#
# Week 13: Audio feature engineering using Urban Sound dataset 

# %%
"""Week 13: audio feature engineering + classical ML baseline."""


# %%
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, librosa
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import time


# %%
# Parameters
TARGET_SR=22050; DURATION=4.0; TARGET_LEN=int(TARGET_SR*DURATION); N_FFT=2048; HOP=512; N_MELS=64; N_MFCC=20
root = Path('../../../../pytorch01/UrbanSound8K')


# %%
files = [f.name for f in root.iterdir() if f.is_file()]
print(files)


# %%

def load_audio_fixed(path,target_sr=TARGET_SR,duration=DURATION):
    y,sr=librosa.load(path,sr=None,mono=True)
    if sr!=target_sr: y=librosa.resample(y,orig_sr=sr,target_sr=target_sr)
    n=int(round(target_sr*duration)); y=np.pad(y,(0,max(0,n-y.size)))[:n]
    return y.astype(np.float32,copy=False)



# %%

def mel_db(y,sr=TARGET_SR):
    M=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,power=2.0)
    return librosa.power_to_db(M,ref=np.max)



# %%

def mfcc_matrix(y,sr=TARGET_SR): return librosa.feature.mfcc(S=mel_db(y,sr),n_mfcc=N_MFCC)
def mfcc_summary(y,sr=TARGET_SR):
    C=mfcc_matrix(y,sr); return np.concatenate([C.mean(1),C.std(1)]).astype(np.float32)



# %%

def urbansound_path(root,row): return root/'audio'/f"fold{int(row['fold'])}"/str(row['slice_file_name'])



# %%

def featurize_rows(rows,root):
    X=[]; y=[]
    for _,row in rows.iterrows(): X.append(mfcc_summary(load_audio_fixed(urbansound_path(root,row)))); y.append(row['class'])
    return np.vstack(X),np.asarray(y)



# %%

def run_urbansound_baseline(root):
    root=Path(root); meta=pd.read_csv(root/'metadata'/'UrbanSound8K.csv'); train=meta[meta.fold.isin(range(1,9))]; valid=meta[meta.fold.eq(9)]; test=meta[meta.fold.eq(10)]
    Xtr,ytr=featurize_rows(train,root); Xv,yv=featurize_rows(valid,root); Xte,yte=featurize_rows(test,root)
    clf=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight='balanced_subsample').fit(Xtr,ytr)
    print('Validation accuracy:',accuracy_score(yv,clf.predict(Xv))); yp=clf.predict(Xte); print('Test accuracy:',accuracy_score(yte,yp)); print(classification_report(yte,yp));
    ConfusionMatrixDisplay.from_predictions(yte, yp, cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.tight_layout()
    plt.show()
    return clf



# %%
# Run random forest classifier
start_ns = time.perf_counter_ns()

run_urbansound_baseline(root)   # Execution of main routine

end_ns = time.perf_counter_ns()
elapsed_ms = (end_ns - start_ns) / 1e9
print(f"Execution time: {elapsed_ms:.3f} sec")


# %%
#ConfusionMatrixDisplay.from_predictions(yte, yp, labels=CLASSES, cmap='Blues')
ConfusionMatrixDisplay.from_predictions(yte, yp, cmap='Blues')


# %%
#if __name__=='__main__': print('Set a local UrbanSound8K path and call run_urbansound_baseline(path).')

