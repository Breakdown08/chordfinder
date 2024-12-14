from keras.saving import load_model
from .scripts import utils
import os, sys
import numpy as np


if getattr(sys, 'frozen', False):
     application_path = sys._MEIPASS
else:
     application_path = os.path.dirname(os.path.abspath(__file__))

model = None
meta = None


def predict_chord_from_audio(audio):
    mask = utils.envelope(audio, meta["sr"], meta["threshold"])
    y = audio[mask]
    if len(y) >= meta["sr"]:
        y = y[:meta["sr"]]
    else:
        y = np.pad(y, (0, meta["sr"] - len(y)), mode='constant')
    mel_spec = utils.get_melspectogram(y, meta["sr"], meta["n_fft"], meta["hop_length"], meta["n_mels"])
    mel_spec = mel_spec[np.newaxis, ...]
    return predict_chord_from_mel(mel_spec)


def predict_chord_from_mel(mel_data):
    y_pred = model.predict(mel_data, verbose=0)
    predicted_label_idx = np.argmax(y_pred)
    chance = y_pred[0][predicted_label_idx]
    predicted_label = meta["classes"][predicted_label_idx]
    return predicted_label, chance


def start():
    global model
    global meta
    model = load_model(os.path.join(application_path, "resources/trained_model.keras"))
    meta = utils.deserialize_dict(os.path.join(application_path, "resources/meta.json"))

# microphone_audio = np.array(json.loads(message.decode('utf-8')))
# res = predict_chord_from_audio(microphone_audio)
