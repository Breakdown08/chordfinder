import os
import keras
import librosa
import numpy as np
import pandas as pd
import json
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import soundfile as sf


def envelope(y, sr, threshold):
    mask = []
    y1 = pd.Series(y).apply(np.abs)
    y_mean = y1.rolling(window=int(sr/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def load_audio(path, sr):
    rate = librosa.get_samplerate(path)
    y1, _ = librosa.load(path, sr=rate)
    wav = librosa.resample(y=y1, orig_sr=rate, target_sr=sr)   
    return wav


def get_data(path, sr, dt, n_classes, n_mels, hop_length, n_fft, threshold, frames):
    wav_paths = glob('{}/**'.format(path), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    
    classes = sorted(os.listdir(path))
    le = LabelEncoder()
    le.fit(classes)

    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    X = np.empty((1, int(sr * dt)), dtype=np.float32)
    Y = np.empty((1, n_classes), dtype=np.float32)
    
    for (file, label) in zip(wav_paths, labels):
        y1 = load_audio(file, sr)
        mask = envelope(y1, sr, threshold)
        y = y1[mask]
        n = y.shape[0]
        m = n // sr
        for k in range(m):
            a = np.expand_dims(y[k * sr:(k + 1) * sr], axis=0)
            X = np.append(X, a, axis=0)
            b = np.expand_dims(to_categorical(label, num_classes=n_classes), axis=0)
            Y = np.append(Y, b, axis=0)

    X = np.delete(X, 0, axis=0)
    Y = np.delete(Y, 0, axis=0)
    
    n_samples = X.shape[0]
    X_mel = np.empty((n_samples, n_mels, frames), dtype=np.float32)
    
    for i in np.arange(n_samples):
        X_mel[i,] = get_melspectogram(X[i,], sr, n_fft, hop_length, n_mels)

    X = X_mel
    
    return X, Y


def get_melspectogram(waveform, sr, n_fft, hop_length, n_mels):
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels) 
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram


def serialize_dict(dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent=4)


def deserialize_dict(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        meta = json.load(file)
    return meta
