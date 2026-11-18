# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ecen45632 [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-ecen45632
# ---

# %% [markdown]
# ## ECEN 4/5632 Theory and Applications of Digital Filtering, Fall 2026
# # UrbanSound Classifier Project
#

# %% [markdown]
# **Overview:** This project is a cross-disciplinary exploration of the fields of digital signal
# processing (DSP) and artificial intelligence (AI). The goal is to start from a basic convolutional
# neural network (CNN) in PyTorch and use the knowledge from class to test several options for
# performance improvement. A byproduct of this exploration is to gain experience working with machine learning (ML) concepts and neural network (NN) architectures.
#
# ---

# %% [markdown]
# **Specifics:** The Jupyter notebook `BasicUrbanSound_001.ipynb` is a working minimal CNN implementation using
# PyTorch for the classification of sound files from the
# [urbansound8k dataset](https://urbansounddataset.weebly.com/urbansound8k.html). For your convenience, the
# `UrbanSound8K.tar.gz` data set (6 GB) is also available on the Canvas class website.
#
# The `BasicUrbanSound_001.ipynb` Jupyter notebook consists essentially of three parts:
#
# 1. The preparation of the sound files and their labels and encapsulating them into a DataLoader object
# 2. The specification of the CNN architecture in terms of its layers and their data processing features
# 3. The train and validation functions to optimize the learning of the model parameters
#
# The `UrbanSound8K` dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
# air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer,
# siren, and street_music. The audio files are in .wav format and they are split over 10 folds (each fold
# containing about 800 files) and the lengths and sampling rates of the files vary (they were crowd-sourced
# through [freesound](https://freesound.org/)). The .csv file `UrbanSound8k.csv` contains meta-data information about every audio file in the dataset. This includes:
#
# * slice_file_name in format [fsID]-[classID]-[occurrenceID]-[sliceID].wav
# * fsID: freesound ID
# * start (in original freesound recording)
# * end (in original freesound recording)
# * salience: subjective rating, 1=foreground, 2=background
# * fold: the fold number (1-10) in which this file is located
# * classID (numeric): 0=air_conditioner, 1=car_horn, 2=children_playing, 3=dog_bark, 4=drilling, 5=engine_idling, 6=gun_shot, 7=jackhammer, 8=siren, 9=street_music
# * classID (name)
#
# ---

# %% [markdown]
# **Your Task:** Explore, learn, be creative, and ask questions! 
#
# **Asking good and creative questions is something that (current) AI is not good at**, so that is something
# where you as a curious and educated student can shine!
#
# Some things to explore and try out:
#
# * Run the `BasicUrbanSound_001.ipynb` notebook to establish an initial baseline performance (expect something like a 20% accuracy).
# * Visualize the spectrogram images that the CNN is processing and ask how they could be improved.
# * Use your knowledge about `Mel Spectrograms` and `log Mel Spectrograms`.
# * What parameters should you use for the `Mel Spectrograms` and why would you expect them to permorm better?
# * Would an increase of the audio data taken in the time domain help improve classification accuracy?
# * Would an increase of the frequency data/resolution help improve classification accuracy?
# * Could you use more than one fold or different folds for training the CNN?
# * What about reshuffling the data in the DataLoader?
# * Write a program for testing specific audio files and check if there is a pattern to missclassifications.
# * Use a different CNN architecture (e.g., AlexNet) and/or tinker with some setings of the CNN.
#
# Feel free to come up with your own questions. Some may be related to improving classification accuracy,
# but some may just be directed towards learning about ML, CNNs, and AI in connection with digital signal
# processing.
#
# ---

# %% [markdown]
# **Your Deliverables:** Jupyter notebooks for all additional Python code developed and all (labeled)
# graphs generated. Clearly state the topics that you explored, why you explored them, and the answers that you got and how they may have differed from what you expected. This can be done in markdown cells in the Jupyter notebook or in a separate Word/pdf file. You may have more questions than can be answered in the limited time for the project. Write down those questions, your reasoning, and the answers that you would expect and think of them as future topics to explore. Also, do not hesitate to include things that you tried but that didn't work (and why). Documentation of failures are important milestones whenever you explore new topics!
#
# To make the grading process easier, please render your Jupyter notebooks as pdf files in addition to
# submitting the actual .ipynb file.
#
# ---

# %% [markdown]
# **Grading Criteria:** As a basis you should demonstrate a reasonable effort at exploring and documenting
# your exploration of this topic. A well written presentation with labeled graphs and commented Python code
# is expected. To obtain full points some creativity and asking good questions must be demonstrated. Overall the process of exploring and reasoning and its documentation is more important than the actual improvement in classification accuracy that you obtained.
#
# **AI Help:** Use AI to help you with specific code constructions and explanations of what different code snippets do, but solutions that were mostly AI generated and don't show any clear reasoning on your part will not be accepted. Document and cite your use of AI by stating the AI agent(s) and the prompts that you used.
#
# ---

# %%
