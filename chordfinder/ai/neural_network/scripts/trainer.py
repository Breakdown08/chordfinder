import utils, os
import model as keras_model
import matplotlib.pyplot as plt
import numpy as np


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
resources_dir = os.path.join(base_dir, "app", "resources")
dataset_dir = os.path.join(base_dir, "dataset")
train_path = os.path.join(dataset_dir, "training")
test_path = os.path.join(dataset_dir, "test")
model_path = os.path.join(resources_dir, "trained_model.keras")
meta_path = os.path.join(resources_dir, "meta.json")

os.makedirs(resources_dir, exist_ok=True)

sr = 16000
dt = 1.0
N_CLASSES = 8
n_mels = 64
hop_length = 256
n_fft = 1024
threshold = 0.01
frames = (sr // hop_length) + 1
classes = sorted(os.listdir(test_path))

meta = {
    "sr": sr,
    "dt": dt,
    "n_classes": N_CLASSES,
    "n_mels": n_mels,
    "hop_length": hop_length,
    "n_fft": n_fft,
    "threshold": threshold,
    "frames": frames,
    "classes": classes
}

# X_train, Y_train = utils.get_data(train_path, sr, dt, N_CLASSES, n_mels, hop_length, n_fft, threshold, frames)
# X_train.shape, Y_train.shape

# model = keras_model.model_Conv2D(N_CLASSES, n_mels, frames)

# batch_size = 32
# epochs = 30
# history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

# history_dict = history.history
# loss_list = history_dict['loss']
# accuracy_list = history_dict['accuracy']
# print(f"Minimum Loss = {np.round(np.min(loss_list), decimals=2)}")
# print(f"Maximum Accuracy = {np.round(np.max(accuracy_list), decimals=2)}")

# plt.figure(figsize=(12,4))
# plt.subplot(1, 2, 1)
# plt.plot(np.arange(epochs), loss_list)
# plt.title("Loss")
# plt.xlabel("Epochs")
# plt.subplot(1, 2, 2)
# plt.plot(np.arange(epochs), accuracy_list)
# plt.title("Accuracy")
# plt.xlabel("Epochs")
# plt.tight_layout()

# X_test, Y_test = utils.get_data(test_path, sr, dt, N_CLASSES, n_mels, hop_length, n_fft, threshold, frames)
# X_test.shape, Y_test.shape

# model.save(model_path)
utils.serialize_dict(meta, meta_path)

# y_pred = model.predict(X_test)
# y_pred.shape

# correct = 0
# incorrect = 0
# N = y_pred.shape[0]

# for n in range(N):
#     a = np.argmax(y_pred[n,:])
#     b = np.argmax(Y_test[n,:])
#     if a==b:
#         correct += 1
#     else:
#         incorrect += 1 
        
# print(f'Total={N}, Correct={correct}, Incorrect={incorrect}')
# print(f'Test Set Accuracy = {np.round(correct*100/N, decimals=1)} %')
