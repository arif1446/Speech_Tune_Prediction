# -*- coding: utf-8 -*-
"""STP-CNN.ipynb

# **Speech Tone Prediction (STP) using Neural Network**

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

There are a total of 600 voice audio data files produced by male and female speakers. The audio file format is a .WAV format.

Download the dataset from [here](https://drive.google.com/file/d/1aZi9ssxbgBUtezW-L6EsWygjdH5F0KhH/view?usp=drive_link)

Here is the filename identifiers for the dataset:

Modality ( file format is .wav, and audio-only).

Vocal channel (speech).

Speech Label= confident, neutral and nervous

Vocalilst (Total audio is 600, 203 voice are female and 397 voice are male).

# Mounting Drive
We are mounting the audio dataset from Google Drive
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""# Extracting the Dataset

Now we unzip the train dataset from the drive
"""

!pip install unrar
!unrar x '/content/drive/MyDrive/Project/Dataset.rar' #extracted the contents

"""# Import necessary libraries"""

!pip install pydub

import numpy as np          # numerical or mathematical operations on arrays
import sys
import pandas as pd         # data analysis or manipulation, CSV file I/O
import os                   # operating system-related functions

#data visualization and graphical plotting, display sound files as images and hearing audio
import librosa                 # audio processing or analyze audio files
import librosa.display          # to display sound data as images
import seaborn as sns           #built on top of matplotlib with similar functionalities
import matplotlib.pyplot as plt
from librosa.display import waveshow
import IPython.display as ipd
from IPython.display import Audio     #used to display and hear the audio
import soundfile as sf
from scipy.io import wavfile          #reading and writing audio files
from scipy import stats
from glob import glob                 #to find all pathnames matching a specific pattern
import random                       # used for randomizing
from scipy.fftpack import fft
from scipy import signal
from sklearn.decomposition import PCA

# to play the audio files
from IPython.display import Audio
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

#from keras.utils import np_utils, to_categorical
from keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils

import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings                   #to manipulate warnings details
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

## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

# Read the CSV file
df = pd.read_csv('/content/drive/MyDrive/Project/speech_gender_wise.csv')
df.head(5)

# Select only 'label' and 'gender' columns from the CSV
df = df[['label', 'gender']]
df.head(5)

df['label'].value_counts()

df['speech'].value_counts()

"""# Initialize Label Encoder to encode the 'label' column"""

# Read the CSV file containing additional data
df = pd.read_csv('/content/drive/MyDrive/Project/speech_gender_wise.csv')

# Select only 'label' and 'gender' columns from the CSV
df = df[['label', 'gender']]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'label' column and create a new 'label_encoded' column
df['label_encoded'] = label_encoder.fit_transform(df['label'])

"""# Merge audio paths with labels and genders

Note: This assumes that the number of audio files matches the number of rows in the CSV
"""

#matches the number of rows in the CSV
df['path'] = paths[:len(df)]
df = df[['path', 'gender', 'label_encoded']]

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
speech_df = pd.DataFrame(file_speech, columns=['Label'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([speech_df, path_df], axis=1)
Tess_df.head()

# count of speech in each label
Tess_df.groupby('Label').count()

# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Tess_df], axis = 0)
data_path.head()

# @title Label

Tess_df.groupby('Label').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.ylabel('No of Speech', size=12)

# @title Gender

df.groupby('gender').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Label distribution by Gender

df.groupby(['gender', 'label_encoded']).size().unstack().plot(kind='bar')

# @title Gender distribution by label

df.groupby(['label_encoded', 'gender']).size().unstack().plot(kind='bar')

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

"""**Count audio file in each of the Label**"""

# Get all subdirectories (classes/labels) in the dataset folder
labels = [label for label in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, label))]

# Dictionary to store label counts
label_counts = {}

# Loop through each label and count the number of files
for label in labels:
    label_folder = os.path.join(audio_path, label)
    file_count = len([f for f in os.listdir(label_folder) if f.endswith('.wav')])  # Adjust the file extension accordingly
    label_counts[label] = file_count

# Print or further analyze the label counts
for label, count in label_counts.items():
    print(f'{label}: {count} files')
print('Total audio in the Dataset:', len(paths))

#Understand the number of recordings
labels=os.listdir(audio_path)
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

#plot
plt.figure(figsize=(10,6)) #size of the plot graph
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Labels', fontsize=15)
plt.ylabel('No of recordings', fontsize=15)
plt.xticks(index, labels, fontsize=12, rotation=60)
plt.title('No. of recordings for each Labels', fontsize=15)
plt.show()

labels=["Confident", "Nervous", "Neutral","PublicSpeaking"]

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

"""# Ramdomly select a audio file from dataset"""

def waveshow(data, sr, sp):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} speech'.format(sp), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, sp):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} speech'.format(sp), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

speech='confident'
path = np.array(data_path.Path[data_path.Label==speech])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech )
create_spectrogram(data, sampling_rate, speech)
Audio(path)

speech='nervous'
path = np.array(data_path.Path[data_path.Label==speech])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech )
create_spectrogram(data, sampling_rate, speech)
Audio(path)

speech='neutral'
path = np.array(data_path.Path[data_path.Label==speech])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech )
create_spectrogram(data, sampling_rate, speech)
Audio(path)

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
for path, speech in zip(data_path.Path, data_path.Label):
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

# Split the data into training and testing sets
X = df[['path', 'gender']]
Y = df['label_encoded']

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

"""# Remove silence"""

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

"""# Run the Model"""

model=Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)

history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

"""Epochs the total number of iterations of all the training data in one cycle for training the machine learning model

# Save and Load the model
"""

# Save the model
model.save('/content/drive/MyDrive/Project/model4.h5')

#Loading the best model
from keras.models import load_model
model=load_model('/content/drive/MyDrive/Project/model4.h5')

"""# Results of accuracy and loss
Now we visualize the results through plot graphs
"""

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(50)]
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

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
    # Compute ROC curve and ROC AUC for each class
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred[:, i])
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
np.save('/content/drive/MyDrive/Project/file_stp.npy', data)

"""# Final Thoughts
Deep learning models give more accuracy results compared to machine learning algorithms

This model can be reused differently depending on the data set and parameters, including speech tone prediction or other sound related tracks.
"""