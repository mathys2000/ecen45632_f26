# One week time frequency DSP project

Read Time_Frequency_Processing_Project.docx, open Student.ipynb,
and complete the three functions in student_dsp.py. Starter cells intentionally
raise NotImplementedError until the algorithms are written.

Only NumPy, SciPy and Matplotlib are required by the scripts. Jupyter is needed
for interactive notebooks. No internet download is needed to run the fixtures.

WAV conventions: audio is mono float32, fs=16000. qpsk_iq.wav is stereo float32,
channel 0 is I and channel 1 is Q. Do not average channels of the I/Q file.
Waveform references and receiver parameters are in signals/truth.npz.

SciPy reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html
