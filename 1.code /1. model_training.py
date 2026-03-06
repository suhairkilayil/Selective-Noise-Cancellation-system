from google.colab import drive
drive.mount('/content/drive')

!pip install librosa
!pip install tensorflow

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Dataset path
base_path = "/content/drive/MyDrive/UrbanSound8K/UrbanSound8K/audio/"

# Parameters
SR = 22050
N_MELS = 64
DURATION = 4
SAMPLES_PER_TRACK = SR * DURATION

# Mel Spectrogram Extraction Function
def extract_mel(file_path):
    signal, sr = librosa.load(file_path, sr=SR)
    if len(signal) < SAMPLES_PER_TRACK:
        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_TRACK]
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# Load Data
X = []
y = []

for fold in range(1, 11):
    fold_path = os.path.join(base_path, f"fold{fold}")
    for file in os.listdir(fold_path):
        if file.endswith(".wav"):
            label = int(file.split("-")[1])  # Extract label
            mel = extract_mel(os.path.join(fold_path, file))
            X.append(mel)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Reshape for CNN input
X = X[..., np.newaxis]

# Convert labels to categorical
y_cat = to_categorical(y, num_classes=10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print("✅ Data Loaded: ", X_train.shape, y_train.shape)

def build_cnn_model(input_shape, num_classes=10):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    # Conv Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    # Conv Block 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))

    return model

input_shape = (64, X.shape[2], 1)
model = build_cnn_model(input_shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("/content/drive/MyDrive/SelectiveNoiseCancellationOutput/UrbanSound_CNN_BestModel.h5", save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

model.save("/content/drive/MyDrive/SelectiveNoiseCancellationOutput/UrbanSound_CNN_FinalModel.h5")
print("✅ Model saved to Drive")
