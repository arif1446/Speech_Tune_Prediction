# -*- coding: utf-8 -*-
"""# Speech tone prediction

I am going to build a speech tone prediction classifier. But first we need to learn about what is speech tone perdiction (STP) and why are we building this project? Well, few of the reasons are- First, lets define STP, speech tone perdiction (STP), abbreviated as STP, is the act of attempting to prediction human voice and affective states from speech. This is capitalizing on the fact that voice often reflects underlying speech through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human speech tone.

Why we need it? Speech tone prediction is the part of speech recognition which is gaining more popularity and need for it increases enormously. Although there are methods to recognize tone using machine learning techniques, this project attempts to use deep learning to prediction the tone from voice data.

STP is used in call center for classifying calls according to tone and can be used as the performance parameter for conversational analysis thus identifying the unsatisfied customer, customer satisfaction and so on for helping companies improving their services.

It can also be used in-car board system based on information of the mental state of the driver can be provided to the system to initiate his/her safety preventing accidents to happen.
"""
# Import libraries


!pip install pydub

import pandas as pd
import numpy as np
import os
import sys

# librosa is a Python library for analyzing audio
import librosa
import librosa.display
import seaborn as sns
from librosa.display import waveshow
import matplotlib.pyplot as plt

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

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""**Data Preparation**

As we are working with our manual created datasets, so i will be creating a dataframe storing all speech of the data in dataframe with their paths. We will use this dataframe to extract features for our model training.

Here is the filename identifiers for the dataset:

Modality ( file format is .wav, and audio-only).

Vocal channel (speech).

Speech (01 = neutral, 02 = confident, 03 = nervous).

Vocalilst (Total audio is 345, 89 voice are male and 256 voice are female).

"""

!pip install unrar
!unrar x '/content/drive/MyDrive/Project/Dataset.rar'

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

# Read the CSV file containing additional data
df = pd.read_csv('/content/drive/MyDrive/Project/speech_gender_wise.csv')

# Select only 'label' and 'gender' columns from the CSV
df = df[['label', 'gender']]

# Initialize LabelEncoder to encode the 'label' column
label_encoder = LabelEncoder()

# Encode the 'label' column and create a new 'label_encoded' column
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Merge audio paths with labels and genders
# Note: This assumes that the number of audio files matches the number of rows in the CSV
df['path'] = paths[:len(df)]
df = df[['path', 'gender', 'label_encoded']]

# Split the data into training and testing sets (80% training, 20% testing)
X = df[['path', 'gender']]
y = df['label_encoded']

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
data_path.to_csv("speech_recognition.csv",index=False)
data_path.head()

# @title Speech

from matplotlib import pyplot as plt
import seaborn as sns
data_path.groupby('Speech').size().plot(kind='bar', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

"""We can also plot waveplots and spectograms for audio signals.

Waveplots - Waveplots let us know the loudness of the audio at a given time.

Spectograms - A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals as they vary with time. It’s a representation of frequencies changing with respect to time for given audio/music signals.
"""

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
path = np.array(data_path.Path[data_path.Speech==speech])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech )
create_spectrogram(data, sampling_rate, speech)
Audio(path)

speech='nervous'
path = np.array(data_path.Path[data_path.Speech==speech])[1]
data, sampling_rate = librosa.load(path)
waveshow(data,sampling_rate, speech )
create_spectrogram(data, sampling_rate, speech)
Audio(path)

speech='neutral'
path = np.array(data_path.Path[data_path.Speech==speech])[1]
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

Extraction of features is a very important part in analyzing and finding relations between different things. As we already know that the data provided of audio cannot be understood by the models directly so we need to convert them into an understandable format for which feature extraction is used. The audio signal is a three-dimensional signal in which three axes represent time, amplitude and frequency.

Mel Frequency Cepstral Coefficients (MFCC) form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale. Chroma Vector : A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing). Chroma Deviation : The standard deviation of the 12 chroma coefficients.

In this project we are not going deep in feature selection process to check which features are good for our dataset rather i am only extracting 5 features:

Zero Crossing Rate
Chroma_stft
MFCC RMS(root mean square) value MelSpectogram to train our model.
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

X, Y = [], []
for path, speech in zip(data_path.Path, data_path.Speech):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending speech 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(speech)

len(X), len(Y), data_path.Path.shape

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

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(None, 1)))
model.add(Dense(units=4, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

"""# Model"""

history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))

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

# Save the model
model.save('/content/drive/MyDrive/Project/model6.h5')

#Loading the best model
from keras.models import load_model
model=load_model('/content/drive/MyDrive/Project/model6.h5')

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(7)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 5))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test, y_pred))

# Import necessary modules
import matplotlib.pyplot as plt
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
    # Compute ROC curve and ROC AUC for each class
    # Use y_pred_probs[:, i] for predicted probabilities if multi-class
    # If binary classification, use y_pred_probs directly
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

"""We can see our model is more accurate in predicting surprise, angry emotions and it makes sense also because audio files of these emotions differ to other audio files in a lot of ways like pitch, speed etc.. We overall achieved 61% accuracy on our test data and its decent but we can improve it more by applying more augmentation techniques and using other feature extraction methods."""

#Save the NumPy array to a file
np.save('/content/drive/MyDrive/Project/file_stp6.npy', data)