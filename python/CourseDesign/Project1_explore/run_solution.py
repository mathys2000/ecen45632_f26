"""Run from this directory: python run_solution.py. Outputs are deterministic."""
from pathlib import Path
import json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from common import FS, fixtures, read_wav, write_wav, relerr, snr, receive
from dsp_solution import analysis, synthesis, ola_filter, remove_jammer

def plot_tf(ax, x, L, hop, title):
    X, m = analysis(x, L, hop)
    f = np.fft.fftfreq(L, 1/FS)
    keep = f >= 0
    t = (np.arange(len(X))*hop-m['left']+(L-1)/2)/FS
    A = np.abs(X[:,keep]); A /= max(A.max(), 1e-15)
    ax.pcolormesh(t, f[keep], 20*np.log10(np.maximum(A.T, 1e-5)),
                  shading='auto', vmin=-70, vmax=0, cmap='magma')
    ax.set(xlim=(0,len(x)/FS), ylim=(0,4000), xlabel='Time (s)',
           ylabel='Frequency (Hz)', title=title)

def main():
    root = Path(__file__).resolve().parent
    out = root/'results'; out.mkdir(exist_ok=True)
    a = fixtures(root/'signals'); metrics = {}
    fig, axes = plt.subplots(3,1,figsize=(8,9),constrained_layout=True)
    for ax,L in zip(axes,[256,1024,4096]):
        plot_tf(ax, a['probe'], L, L//4, f'L={L}, duration={1000*L/FS:.0f} ms')
    fig.savefig(out/'uncertainty.png',dpi=160); plt.close(fig)
    tfrows=[]
    for L in [256,1024,4096]:
        H=L//4; X,m=analysis(a['transient'],L,H)
        energy=np.sum(np.abs(X)**2,axis=1)
        keep=energy>=.1*energy.max()
        tfrows.append(dict(L=L,window_ms=1000*L/FS,bin_hz=FS/L,hop_ms=1000*H/FS,
                           transient_10percent_energy_width_ms=1000*np.count_nonzero(keep)*H/FS))
    metrics['time_frequency']=tfrows
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    n=np.arange(1024); tone=np.cos(2*np.pi*1000*n/FS)+np.cos(2*np.pi*1080*n/FS)
    w=signal.get_window('hann',1024)
    for N in [1024,4096]:
        A=2*np.abs(np.fft.rfft(tone*w,N))/w.sum()
        ax.plot(np.fft.rfftfreq(N,1/FS),A,'.-',label=f'L=1024, N={N}')
    ax.set(xlim=(900,1200),xlabel='Frequency (Hz)',ylabel='Amplitude',title='Zero padding samples the same windowed spectrum');ax.legend()
    fig.savefig(out/'zero_padding.png',dpi=160);plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.5),constrained_layout=True)
    L=1024; N=65536; n=np.arange(L); window_rows=[]
    for name in ['boxcar','hann','hamming','blackman']:
        w=signal.get_window(name,L,fftbins=True)
        W=np.abs(np.fft.fft(w,N)); W/=W[0]
        minima=signal.find_peaks(-W[:N//2])[0]; first=int(minima[0])
        psl=20*np.log10(np.max(W[first:N//2]))
        window_rows.append(dict(window=name, null_to_null_hz=2*first*FS/N,
                                peak_sidelobe_db=float(psl),
                                enbw_hz=float(FS*np.sum(w*w)/np.sum(w)**2)))
        # Leakage experiment uses coherent-gain correction, not energy normalization.
        tone=.65*np.cos(2*np.pi*1003*n/FS)+.015*np.cos(2*np.pi*1140*n/FS)
        A=2*np.abs(np.fft.rfft(tone*w,N))/w.sum()
        ax.plot(np.fft.rfftfreq(N,1/FS),20*np.log10(np.maximum(A,1e-8)),label=name)
    ax.set(xlim=(850,1350),ylim=(-90,5),xlabel='Frequency (Hz)',ylabel='Amplitude (dB re 1)',title='Off-bin strong tone and weak neighbor'); ax.legend()
    fig.savefig(out/'windows.png',dpi=160); plt.close(fig)
    metrics['windows']=window_rows

    rows=[]
    for overlap in [0,.25,.5,.75,.875]:
        H=int(1024*(1-overlap))
        for win in (['boxcar','hann'] if overlap==0 else ['hann']):
            start=time.perf_counter(); X,m=analysis(a['probe'],1024,H,win)
            row=dict(overlap_percent=100*overlap,hop=H,window=win,frames=len(X))
            try:
                y,d=synthesis(X,m)
                row.update(relative_error=relerr(a['probe'],y), min_denominator=float(d.min()),max_denominator=float(d.max()))
                assert row['relative_error']<1e-11
            except ValueError as e:
                row['expected_failure']=str(e)
                assert win=='hann' and overlap==0
            row['elapsed_seconds']=time.perf_counter()-start; rows.append(row)
    metrics['overlap']=rows
    h=signal.firwin(129,1200,fs=FS,window='hamming')
    ola=ola_filter(a['mixture'],h,B=512)
    ref=np.convolve(a['mixture'],h)
    metrics['ola']=dict(relative_error=relerr(ref,ola), output_samples=len(ola),
                        fft_length=1024,group_delay_samples=64)
    assert relerr(ref,ola)<1e-11 and len(ola)==len(a['mixture'])+128
    metrics['ola_block_sweep']=[]
    for B in [128,512,2048]:
        yy=ola_filter(a['mixture'],h,B)
        metrics['ola_block_sweep'].append(dict(B=B,nfft=1<<(B+len(h)-2).bit_length(),
                blocks=int(np.ceil(len(a['mixture'])/B)),relative_error=relerr(ref,yy)))
    # Independent edge-case checks, not only one friendly audio example.
    rng=np.random.default_rng(52)
    for N,B,M in [(1,1,1),(997,127,83),(1031,512,129)]:
        x=rng.normal(size=N)+1j*rng.normal(size=N); hh=rng.normal(size=M)
        assert relerr(np.convolve(x,hh),ola_filter(x,hh,B))<1e-11
    write_wav(out/'ola_lowpass.wav',ola,scale=.8)

    cleaned,diag=remove_jammer(a['mixture'])
    write_wav(out/'input_mixture.wav',a['mixture'],scale=.8)
    write_wav(out/'desired_reference.wav',a['melody'],scale=.8)
    write_wav(out/'cleaned.wav',cleaned,scale=.8)
    metrics['jammer']=dict(input_snr_db=snr(a['melody'],a['mixture']), output_snr_db=snr(a['melody'],cleaned))
    # Apply the estimated mask unchanged to each source to separate attenuation from distortion.
    def apply_gain(src,G,L,H):
        X,m=analysis(src,L,H); return synthesis(X*G,m)[0]
    jamrows=[]
    for L in [256,1024,4096]:
        for width in [30,60,120]:
            y,dg=remove_jammer(a['mixture'],L,L//4,width)
            jc=apply_gain(a['jammer'],dg['G'],L,L//4)
            sc=apply_gain(a['melody'],dg['G'],L,L//4)
            jamrows.append(dict(L=L,width_hz=width,output_snr_db=snr(a['melody'],y),
                               jammer_attenuation_db=float(20*np.log10(np.linalg.norm(a['jammer'])/np.linalg.norm(jc))),
                               clean_relative_error=relerr(a['melody'],sc)))
    metrics['jammer_sweep']=jamrows
    # Equal-width stationary notch is a negative control.
    X,m=analysis(a['mixture']); f=np.fft.fftfreq(1024,1/FS)
    stationary=1-.97*np.exp(-.5*((np.abs(f)-2400)/60)**2)
    ys,_=synthesis(X*stationary,m)
    metrics['jammer']['stationary_notch_snr_db']=snr(a['melody'],ys)
    fig,axes=plt.subplots(3,1,figsize=(8,9),constrained_layout=True)
    for ax,x,title in zip(axes,[a['melody'],a['mixture'],cleaned],['Desired melody','Chirp jammer added','Estimated ridge suppression']):
        plot_tf(ax,x,1024,256,title)
    fig.savefig(out/'jammer.png',dpi=160); plt.close(fig)

    # Quantized WAV round trip is separate from float64 algorithm accuracy.
    _,txwav=read_wav(root/'signals/qpsk_iq.wav',iq=True)
    metrics['iq_wav_roundtrip_error']=relerr(a['tx'],txwav)
    X,m=analysis(a['tx']); coherent,_=synthesis(X,m)
    magnitude,_=synthesis(np.abs(X).astype(complex),m)
    phase=rng.uniform(-np.pi,np.pi,len(X))
    randomized,_=synthesis(X*np.exp(1j*phase[:,None]),m)
    cases={'original':a['tx'],'coherent':coherent,'magnitude_only':magnitude,'frame_phase_reset':randomized}
    metrics['communications']={}
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),constrained_layout=True)
    for name,x in cases.items():
        row,z=receive(x,a['symbols'],a['bits'],a['pulse'])
        row['waveform_relative_error']=relerr(a['tx'],x)
        metrics['communications'][name]=row
        if name!='original':
            ax=axes[['coherent','magnitude_only','frame_phase_reset'].index(name)]
            ax.scatter(z.real,z.imag,s=3,alpha=.4)
            ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),xlabel='I',ylabel='Q',title=name.replace('_',' ')); ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.savefig(out/'constellations.png',dpi=160); plt.close(fig)
    assert relerr(a['tx'],coherent)<1e-11
    assert metrics['communications']['coherent']['ber']==0
    metrics['checks']='PASS: identity, expected NOLA failure, real and complex full OLA, QPSK BER'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    main()
