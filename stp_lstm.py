# -*- coding: utf-8 -*-
"""STP-LSTM.ipynb

# **Speech Tone Prediction (STP) using Neural Network**

**Speech Recognition | Sound Classification | Audio Processing | Sound Analysis using Python | Deep Learning Project **

I am going to build a speech tone prediction using neural network. First we need to learn about what is speech tone perdiction (STP) and why are we building in this project? Well, few of the reasons are- First lets define STP,  speech tone perdiction abbreviated as STP,  It is the act of attempting to prediction human voice and affective states from speech. This is capitalizing on the fact that voice often reflects underlying speech through tone and pitch.

Why do we need it? Speech tone prediction is the part of speech recognition that is gaining more popularity and its need is increasing greatly. While there are methods to detect tones using machine learning techniques, this project attempts to use deep learning to infer tones from voice data.

STP is used in call centers to classify calls according to tone and can be used as a performance parameter for conversation analysis. Thus helping companies to improve their services by identifying customer satisfaction and dissatisfaction etc.

This project explores speech analysis, classification and deep learning techniques to decode speech tunes from audio speech. It can recognize and classify speech recognition in spoken words. The generated model enhances your skills in various audio processing, machine learning and unlocks the possibilities of sound analysis.

In this project we are going to analyze and classify various audio files into a corresponding category and visualize the sound frequency through a plot.

# Dataset Information

These recordings were made from three type of speech:

*   Confident
*   Nervous
*   Neutral
*   PublicSpeaking

There are a total of 600 voice audio data files produced by male and female speakers. The audio file format is a .WAV format.

Download the dataset from [here](https://drive.google.com/file/d/1aZi9ssxbgBUtezW-L6EsWygjdH5F0KhH/view?usp=sharing)


Here is the filename identifiers for the dataset:

Modality ( file format is .wav, and audio-only).

Vocal channel (speech).

Speech Label= confident, neutral, nervous and public speaking

Vocalilst (Total audio is 600, 203 voice are female and 397 voice are male).

# Mounting Drive
We are mounting the audio dataset from Google Drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Extracting the Dataset

Now we unzip the train dataset from the drive
"""

!pip install unrar
!unrar x '/content/drive/MyDrive/Project/Dataset.rar' #extracted the contents

"""# Import necessary libraries"""

!pip install pydub

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import os
import sys

# librosa is a Python library for analyzing audio
import librosa
import librosa.display
import seaborn as sns
from librosa.display import waveshow
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.io import wavfile          #reading and writing audio files
from scipy import stats
from glob import glob #to find all pathnames matching a specific pattern

# to play the audio files
from IPython.display import Audio
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

#from keras.utils import np_utils, to_categorical
from keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import glob
import random
# %pylab inline

# Math
import numpy as np    #numerical or mathematical operations on arrays
from scipy.fftpack import fft
from scipy import signal
from sklearn.decomposition import PCA

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# %matplotlib inline
from os.path import isdir, join
from pathlib import Path

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""# Loading the Dataset
Now we load the dataset for processing.
Where filenames were split and appended as labels
"""

# Initialize empty lists to store audio file paths and labels
paths = []
labels = []

# Walk through the directory containing audio files and collect paths and labels
for dirname, _, filenames in os.walk('/content/Dataset'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        # Extract label from filename
        label = filename.rsplit('_', 1)[0]
        label = label.split('.')[0]
        labels.append(label.lower())
        # Stop after collecting 311 paths and labels
        if len(paths) == 600:
            break
print('Dataset is Loaded')

"""# Read the CSV file containing additional data

classes - Name of the audio file

label - Name of the output class the audio file belongs to
"""

# Read the CSV file containing additional data
df = pd.read_csv('/content/drive/MyDrive/Project/speech_gender_wise.csv')
df.head(5)

# Select only 'label' and 'gender' columns from the CSV
df = df[['label', 'gender']]
df.head(5)

"""classes - Name of the audio file

label - Name of the output class the audio file belongs to

Now we will view the different class distribution in the data set

**Label of the Dataset**
"""

#display all label in dataset
dir_list = os.listdir('Dataset/')
dir_list.sort()
dir_list[:]

# @title Label

from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('label').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

"""seaborn - built on top of matplotlib with similar functionalities

Visualization through a bar graph for the no. of samples for each class.
"""

# @title Label distribution by Pie

df['label'].value_counts().plot(kind='pie')

# @title Gender

from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('gender').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Label distribution by Gender

df.groupby(['gender', 'label']).size().unstack().plot(kind='bar')

# @title Gender distribution by label

df.groupby(['label', 'gender']).size().unstack().plot(kind='bar')

"""# Data Analysis

In this step we will visualize different audio sample of the data through wave plots.

*   sampling_rate - number of splits or samples per second

**Loads an audio file**
"""

# Constants
train_audio_path ='/content/Dataset/'
filename = 'Confident/confident_1.wav'
file_path = train_audio_path + filename

# Creates the  plot
#samples, sample_rate = librosa.load(file_path, sr = 48000)
sample_rate, samples = wavfile.read(str(train_audio_path) + filename)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + file_path)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(samples) / sample_rate, len(samples)), samples)

# Loads audio file
#file_path = glob(train_audio_path + filename)
ipd.Audio(file_path)

# Get all subdirectories (classes/labels) in the dataset folder
labels = [label for label in os.listdir(train_audio_path) if os.path.isdir(os.path.join(train_audio_path, label))]

# Dictionary to store label counts
label_counts = {}

# Loop through each label and count the number of files
for label in labels:
    label_folder = os.path.join(train_audio_path, label)
    file_count = len([f for f in os.listdir(label_folder) if f.endswith('.wav')])  # Adjust the file extension accordingly
    label_counts[label] = file_count

# Print or further analyze the label counts
for label, count in label_counts.items():
    print(f'{label}: {count} files')
print('Total audio in the Dataset:', len(paths))

#Understand the number of recordings
labels=os.listdir(train_audio_path)
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

#plot
plt.figure(figsize=(12,6)) #size of the plot graph
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Labels', fontsize=18)
plt.ylabel('No of recordings', fontsize=20)
plt.xticks(index, labels, fontsize=14, rotation=60)
plt.title('No. of recordings for each Labels', fontsize=20)
plt.show()

labels=["Confident", "Nervous", "Neutral", "PublicSpeaking"]

"""**Sampling rate of an Audio file**"""

samples, sample_rate = librosa.load(file_path)

#Sampling rate
ipd.Audio(samples, rate=sample_rate)

#Sampling rate and sample numbers
print('Sampling rate:',sample_rate)
print('Sample number: ' + str(len(samples)))
print('Sample number:', samples)

"""sampling_rate - number of splits or samples per second

**Let us display an audio file**
"""

ipd.Audio(file_path)

"""Sound bar display of the audio file from the data

# Exploratory Data Analysis

In this step we will visualize different audio sample of the data through wave plots.

We will load the audio file into an array

Audio files loaded into values

Each value is a frequency value of the data

Next we will view the sampling rate

Output value determines the amount of samples per second.

Now we plot some graphs of the audio files
"""

#Duration of recordings
duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
plt.hist(np.array(duration_of_recordings))

"""# Converting audio speech into Text"""

# To install SpeechRecognition package
!pip install SpeechRecognition

import speech_recognition as sr
# Initialize the recognizer
recognizer = sr.Recognizer()

# Open the audio file
file_path = '/content/Dataset/Confident/confident_2.wav'
with sr.AudioFile(file_path) as source:
    # Record the audio data
    audio_data = recognizer.record(source)

    try:
        # Recognize the speech
        text = recognizer.recognize_google(audio_data)
        print("Recognized speech: ", text)
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from service; {e}")

"""# Ramdomly picked audio from audio dataset"""

audio_path = "/content/Dataset/"

tess_directory_list = os.listdir(audio_path)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(audio_path + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[0]
        file_emotion.append(part)
        file_path.append(audio_path + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Speech_label'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()

# prompt: Using dataframe Crema_df: emotion

Tess_df.groupby('Speech_label').count()

# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Tess_df], axis = 0)
data_path.to_csv("speech_recognition.csv",index=False)
data_path.head()

"""# Ramdomly select a audio file from dataset"""

def waveshow(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} Label'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} Label'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

speech_label='confident'
path = np.array(data_path.Path[data_path.Speech_label==speech_label])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech_label)
create_spectrogram(data, sampling_rate, speech_label)
Audio(path)

speech_label='nervous'
path = np.array(data_path.Path[data_path.Speech_label==speech_label])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech_label )
create_spectrogram(data, sampling_rate, speech_label)
Audio(path)

"""# Initialize Label Encoder to encode the 'label' column"""

# Initialize Label Encoder to encode
label_encoder = LabelEncoder()

# Encode the 'label' column and create a new 'label_encoded' column
df['label_encoded'] = label_encoder.fit_transform(df['label'])

"""# Merge audio paths with labels and genders

Note: This assumes that the number of audio files matches the number of rows in the CSV
"""

#matches the number of rows in the CSV
df['path'] = paths[:len(df)]
df = df[['path', 'gender', 'label_encoded']]

# Split the data into training and testing sets (80% training, 20% testing)
X = df[['path', 'gender']]
y = df['label_encoded']

"""# Remove silence from Audio Data"""

# random_state=42 to set the seed of the random generator in the Scikit-learn algorithm
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to remove silence from audio files
def remove_silence(audio_path):
    # Load audio file using pydub
    audio = AudioSegment.from_file(audio_path, format="wav")

    # Split audio on silence
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    # Concatenate non-silent chunks
    output = AudioSegment.empty()
    for chunk in chunks:
        output += chunk
    return output

# Apply silence removal function to audio paths in the training set
x_train['path'] = x_train['path'].apply(remove_silence)

print('Data preprocessing completed.')

"""**MFCC**- Mel-frequency cepstral coefficients technique to extract audio file features

feature - array of the features extracted form the data
"""

#def extract_mfcc(filename):
 #    y, sr = librosa.load(filename, duration=3, offset=0.5)
  #   mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
   #  return mfcc

#X_mfcc = df['path'].apply(lambda x: extract_mfcc(x))
#X = [x for x in X_mfcc]
#X = np.array(X)
#X.shape

## input split
#X = np.expand_dims(X, -1)
#X.shape

#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#y = enc.fit_transform(df[['label_encoded']])
#y = y.toarray()
#y.shape

"""**Data Preparation**

As we are working with our manual created datasets, so i will be creating a dataframe storing all speech of the data in dataframe with their paths. We will use this dataframe to extract features for our model training.
"""

# Paths for data.
audio_path = "/content/Dataset/"

audio_directory_list = os.listdir(audio_path)

file_speech = []
file_path = []

for dir in audio_directory_list:
    directories = os.listdir(audio_path + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[0]
        file_speech.append(part)
        file_path.append(audio_path + dir + '/' + file)

# dataframe for speech of files
speech_df = pd.DataFrame(file_speech, columns=['Speech'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([speech_df, path_df], axis=1)
Tess_df.head()

# count of speech in each label

Tess_df.groupby('Speech').count()

# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Tess_df], axis = 0)
data_path.head()

"""**Loads an audio file**

We can also plot waveplots and spectograms for audio signals.

Waveplots - Waveplots let us know the loudness of the audio at a given time.

Spectograms - A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals as they vary with time. It’s a representation of frequencies changing with respect to time for given audio/music signals.
"""

# Constants
audio_path ='/content/Dataset/'
filename = 'Confident/confident_1.wav'
file_path = audio_path + filename

# Creates the  plot
#samples, sample_rate = librosa.load(file_path, sr = 48000)
sample_rate, samples = wavfile.read(str(audio_path) + filename)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + file_path)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(samples) / sample_rate, len(samples)), samples)

# Loads audio file
#file_path = glob(train_audio_path + filename)
ipd.Audio(file_path)

"""**Sampling rate of an Audio file**

sampling_rate - number of splits or samples per second
"""

samples, sample_rate = librosa.load(file_path)

#Sampling rate
ipd.Audio(samples, rate=sample_rate)

#Sampling rate and sample numbers
print('Sampling rate:',sample_rate)
print('Sample number: ' + str(len(samples)))
print('Sample number:', samples)

"""# Converting audio speech into Text"""

# To install SpeechRecognition package
!pip install SpeechRecognition

import speech_recognition as sr
# Initialize the recognizer
recognizer = sr.Recognizer()

# Open the audio file
file_path = '/content/Dataset/Confident/confident_2.wav'
with sr.AudioFile(file_path) as source:
    # Record the audio data
    audio_data = recognizer.record(source)

    try:
        # Recognize the speech
        text = recognizer.recognize_google(audio_data)
        print("Recognized speech: ", text)
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from service; {e}")

"""**Data Augmentation**

Data augmentation is the process by which we create new synthetic data samples by adding small perturbations on our initial training set.
To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed.
The objective is to make our model invariant to those perturbations and enhace its ability to generalize.
In order to this to work adding the perturbations must conserve the same label as the original training sample.
In images data augmention can be performed by shifting the image, zooming, rotating.
First, let's check which augmentation techniques works better for our dataset.
"""

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)

# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)

"""**Simple Audio**"""

plt.figure(figsize=(14,4))
librosa.display.waveshow(y=data, sr=sample_rate)
Audio(path)

"""**Noise Injection**"""

x = noise(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

"""

We can see noise injection is a very good augmentation technique because of which we can assure our training model is not overfitted.

**Stretching**"""

x = librosa.effects.time_stretch(data, rate=0.8)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

"""**Shifting**"""

x = shift(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

"""**Pitch**"""

x = pitch(data, sample_rate)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

"""From the above types of augmentation techniques i am using noise, stretching(ie. changing speed) and some pitching.

# Feature Extraction

Extraction of features is a very important part in analyzing and finding relations between different things. Provided data of audio cannot be understood by the models directly so we need to convert them into an understandable format for which feature extraction is used. The audio signal is a three-dimensional signal in which three axes represent time, amplitude and frequency.

Mel Frequency Cepstral Coefficients (MFCC) form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.

Chroma Vector : A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing). Chroma Deviation : The standard deviation of the 12 chroma coefficients.
"""

rms =librosa.feature.rms(y=data).T
rms.shape

def rmse(data):
    hop_length = 512
    frame_length = 1024
    n_fft = 1

    rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse = rmse[0]

    energy = np.array([ sum(abs(x[i:i+frame_length]**2))
    for i in range(0, len(x), hop_length)])

def mler(rms):
    lef=0
    delta=0.06
    lowthresh=rms.mean()*delta
    for val in rms:
        lef+=np.sign(lowthresh-val)+1
    mler=lef/len(rms)
    return mler

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

     #spectral centroid
    spec_cent=np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spec_cent)) # stacking horizontally

    #spectral flux
    onset_env =np.mean( librosa.onset.onset_strength(sr=sample_rate, S=librosa.amplitude_to_db(data, ref=np.max)))
    result=np.hstack((result,onset_env))

    #mler
    Mler=mler(rms)
    result=np.hstack((result,Mler))

    #chroma_sens
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=data, sr=sample_rate))
    result=np.hstack((result,chroma_cens))

    #entropy
    # ee=np.round(ent.spectral_entropy(data, sf=100, method='fft'), 2)
    #result=np.np.hstack((result,ee))
    #rmse
    #Rmse=rmse(data)
   # result=np.hstack((result,rmse))
    #spectral roll off
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sample_rate)[0])
    result=np.hstack((result,spec_rolloff))
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data,rate=0.8)

    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically

    return result

"""**Save label speech and path in X, Y**"""

X, Y = [], []
for path, speech in zip(data_path.Path, data_path.Speech):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending speech 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(speech)

len(X), len(Y), data_path.Path.shape

"""**Save features in a .csv file**"""

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
Features.head()

"""we have extracted the data, now we need to normalize and split our data for training and testing."""

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

"""# Split the data into training and testing sets

 (80% training, 20% testing)
"""

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

"""# scaling our data with sklearn's Standard scaler"""

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

## input split for LSTM-----------
X = np.expand_dims(X, -1)
X.shape

"""# Run the Model"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(167,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

"""model is used to detect and classify objects in an image."""

history=model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_test, y_test))

# Train the model
#history=model.fit(X_train, y_train, batch_size=64, validation_split=0.2, epochs=20, validation_data=(X_test, y_test))
#history = model.fit(X, y, validation_split=0.2, epochs=20, batch_size=64)

"""# Results of accuracy and loss
Now we visualize the results through plot graphs
"""

# Print the training and validation accuracy
from matplotlib import pyplot
print(f"Training accuracy: {history.history['accuracy'][-1]}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(40)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

"""# Save and Load the model"""

# Save the model
model.save('/content/drive/MyDrive/Project/model5.h5')

#Loading the best model
from keras.models import load_model
model=load_model('/content/drive/MyDrive/Project/model5.h5')

"""# Predicting on test data."""

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(7)

"""# Confusion Matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 5))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

"""# Precision  &  Recall report"""

print(classification_report(y_test, y_pred))

"""We can see our model is more accurate in predicting surprise, angry emotions and it makes sense also because audio files of these emotions differ to other audio files in a lot of ways like pitch, speed etc.. We overall achieved 61% accuracy on our test data and its decent but we can improve it more by applying more augmentation techniques and using other feature extraction methods."""

# Import necessary modules
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Instead of predict_proba, use predict to get the output
y_pred_probs = model.predict(x_test)

# Binarize the labels
label_binarizer = LabelBinarizer()
y_test_binary = label_binarizer.fit_transform(y_test)

# Check the shape of y_test_binary
print(y_test_binary.shape)  # This should give you (n_samples, n_classes)

# Compute ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_binarizer.classes_)):
    # Reshape y_test_binary to 2D array
    y_test_binary_2d = y_test_binary[:, i].reshape(-1, 1)
    # Fix the indexing issue
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred_probs[:, i] if y_pred_probs.ndim > 1 else y_pred_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(label_binarizer.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (ROC AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()

"""#Save the file in .npy"""

#Save the NumPy array to a file
np.save('/content/drive/MyDrive/Project/file_stp5.npy', data)

"""# Final Thoughts
Deep learning models give more accuracy results compared to machine learning algorithms

This model can be reused differently depending on the data set and parameters, including speech tone prediction or other sound related tracks.
"""

