# One week time frequency DSP project

Student package: read DSP_Week_Project_Assignment.docx, open Student.ipynb,
and complete the four functions in student_dsp.py. Starter cells intentionally
raise NotImplementedError until the algorithms are written.

Instructor package: read DSP_Week_Project_Solution.docx and open Solution.ipynb.
The complete implementation is in dsp_solution.py; run_solution.py regenerates
signals and every figure, metric and audio result.

Create a Python virtual environment if desired, then from this extracted folder:

    pip install -r requirements.txt
    jupyter lab

Instructor command:

    python run_solution.py

Only NumPy, SciPy and Matplotlib are required by the scripts. Jupyter is needed
for interactive notebooks. No internet download is needed to run the fixtures.
Tested with NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8; compatible earlier
versions may also work. The requirements are minimums, not a frozen environment.

WAV conventions: audio is mono float32, fs=16000. qpsk_iq.wav is stereo float32,
channel 0 is I and channel 1 is Q. Do not average channels of the I/Q file.
Waveform references and receiver parameters are in signals/truth.npz.
Evaluation references must not be used to estimate the jammer ridge.

SciPy references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html.
The optional natural audio sources are listed in the handout. No third-party
audio has been redistributed. All supplied WAVs were generated for this project.
The listening comparison should use one common amplitude scale. The solution
audio exports use scale=0.8; rescale the input by 0.8 for matched comparison.
