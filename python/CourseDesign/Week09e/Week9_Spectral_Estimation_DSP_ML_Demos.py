"""Week 9: Spectral Estimation DSP + ML demos.
Run section-by-section or convert to a notebook. Requires numpy, scipy, matplotlib, scikit-learn.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rng = np.random.default_rng(9)

# 1) Wiener-Khinchin: estimated autocorrelation -> PSD
N = 4096
x = rng.normal(size=N)
maxlag = 100
lags = np.arange(-maxlag, maxlag+1)
rhat = np.array([
    np.dot(x[max(0,k):N+min(0,k)], x[max(0,-k):N+min(0,-k)])/(N-abs(k))
    for k in lags
])
S_from_r = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(rhat), 4096)))
w = np.linspace(-np.pi, np.pi, len(S_from_r), endpoint=False)
plt.figure(); plt.plot(w/np.pi, S_from_r); plt.xlabel('normalized frequency / pi'); plt.ylabel('PSD-like estimate'); plt.title('Wiener-Khinchin numerical check'); plt.show()

# 2) LTI filtering: Syy = Sxx |H|^2
h = signal.firwin(51, 0.2)
y = signal.lfilter(h, [1.0], x)
f, Px = signal.welch(x, fs=1.0, nperseg=512)
_, Py = signal.welch(y, fs=1.0, nperseg=512)
wH, H = signal.freqz(h, worN=2048)
plt.figure(); plt.semilogy(f, Py, label='estimated output PSD'); plt.semilogy(wH/(2*np.pi), np.mean(Px)*np.abs(H)**2, label='mean input PSD x |H|^2'); plt.legend(); plt.xlabel('cycles/sample'); plt.show()

# 3) Raw periodogram inconsistency: repeated white-noise records
for N in [128, 512, 2048]:
    vals=[]
    for _ in range(500):
        z=rng.normal(size=N)
        Z=np.fft.rfft(z)
        P=np.abs(Z)**2/N
        vals.append(P[len(P)//3])
    vals=np.asarray(vals)
    print(f'N={N}: mean={vals.mean():.3f}, std={vals.std():.3f}, CV={vals.std()/vals.mean():.3f}')

# 4) Bartlett method from scratch
def bartlett_psd(x, fs, K):
    L=len(x)//K
    x=x[:K*L]
    estimates=[]
    for k in range(K):
        f,p=signal.periodogram(x[k*L:(k+1)*L], fs=fs, window='boxcar', scaling='density')
        estimates.append(p)
    return f, np.mean(estimates, axis=0)

fs=1000.0
N=4096
t=np.arange(N)/fs
sig=np.sin(2*np.pi*120*t)+0.6*np.sin(2*np.pi*170*t)+1.2*rng.normal(size=N)
fP,PP=signal.periodogram(sig, fs=fs)
fB,PB=bartlett_psd(sig, fs, K=8)
fW,PW=signal.welch(sig, fs=fs, window='hann', nperseg=512, noverlap=256)
plt.figure(); plt.semilogy(fP,PP,label='periodogram'); plt.semilogy(fB,PB,label='Bartlett'); plt.semilogy(fW,PW,label='Welch'); plt.xlim(50,250); plt.legend(); plt.xlabel('Hz'); plt.ylabel('PSD'); plt.show()

# 5) Welch segment-length trade-off
plt.figure()
for L in [128, 512, 2048]:
    f,p=signal.welch(sig, fs=fs, window='hann', nperseg=L, noverlap=L//2)
    plt.semilogy(f,p,label=f'nperseg={L}')
plt.xlim(80,210); plt.legend(); plt.xlabel('Hz'); plt.ylabel('PSD'); plt.show()

# 6) Zero-padding is plotting density, not new resolution
short=sig[:256]
plt.figure()
for nfft in [256, 2048, 8192]:
    f,p=signal.periodogram(short, fs=fs, nfft=nfft)
    plt.plot(f,10*np.log10(p+1e-15),label=f'nfft={nfft}')
plt.xlim(100,190); plt.legend(); plt.xlabel('Hz'); plt.ylabel('dB/Hz'); plt.title('Zero-padding samples the same finite-record spectrum more densely'); plt.show()

# 7) Welch bandpower features for classical ML
def psd_features(x, fs=1000.0):
    f,p=signal.welch(x, fs=fs, window='hann', nperseg=256, noverlap=128)
    bands=[(40,130),(130,190),(190,280),(280,400)]
    feats=[]
    for lo,hi in bands:
        mask=(f>=lo)&(f<hi)
        feats.append(np.trapz(p[mask], f[mask]))
    # add spectral centroid
    feats.append(np.sum(f*p)/(np.sum(p)+1e-15))
    return np.asarray(feats)

X=[]; y=[]
for cls in [0,1]:
    for _ in range(250):
        n=1024; tt=np.arange(n)/fs
        noise=rng.normal(size=n)
        if cls==0:
            s=1.0*np.sin(2*np.pi*90*tt)+0.25*np.sin(2*np.pi*220*tt)+0.9*noise
        else:
            s=0.25*np.sin(2*np.pi*90*tt)+1.0*np.sin(2*np.pi*220*tt)+0.9*noise
        X.append(psd_features(s,fs)); y.append(cls)
X=np.asarray(X); y=np.asarray(y)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=0,stratify=y)
svm=make_pipeline(StandardScaler(), SVC(kernel='rbf', C=2.0))
svm.fit(Xtr,ytr)
print('SVM test accuracy:', svm.score(Xte,yte))
print(classification_report(yte,svm.predict(Xte)))
rf=RandomForestClassifier(n_estimators=250,random_state=0)
print('RF 5-fold CV:', cross_val_score(rf,X,y,cv=5).mean())
