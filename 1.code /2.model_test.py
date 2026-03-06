# 🚀 Upload & classify one or more WAV files – CNN + pass/reject flag
# ------------------------------------------------------------------
from google.colab import files
import librosa, numpy as np, pickle, tensorflow as tf

# ------------------------------------------------------------------
# 1 │ constants
SR          = 22_050          # model’s sample-rate
N_MELS      = 64              # mel bands used when training
DURATION    = 4               # sec – clips were 4 s long
SAMPLES     = SR * DURATION

CLASSES = [
    "Air Conditioner","Car Horn","Children Playing","Dog Bark","Drilling",
    "Engine Idling","Gun Shot","Jackhammer","Siren","Street Music"
]
PASS_CLASSES = {"Car Horn","Dog Bark","Gun Shot","Siren"}

# ------------------------------------------------------------------
# 2 │ load the CNN you trained
MODEL_PATH = "/content/drive/MyDrive/SelectiveNoiseCancellationOutput/UrbanSound_CNN_FinalModel.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------------
# 3 │ feature extractor - same mel-spectrogram recipe used in training
def extract_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES-len(y)))
    else:
        y = y[:SAMPLES]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ------------------------------------------------------------------
# 4 │ single-clip inference helper
def classify_clip(path):
    mel = extract_mel(path)
    mel = mel[np.newaxis, ..., np.newaxis]          # (1, 64, time, 1)
    probs = model.predict(mel)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx]) * 100
    label = CLASSES[idx]
    verdict = "pass" if label in PASS_CLASSES else "reject"
    return label, conf, verdict

# ------------------------------------------------------------------
# 5 │ upload file(s) & show results
print("Upload one or more .wav files …")
uploaded = files.upload()

for fname in uploaded:
    try:
        label, conf, decision = classify_clip(fname)
        print(f"‣ {fname}  →  {label}  ({conf:4.1f}% confidence)  |  {decision}")
    except Exception as e:
        print(f"⚠️ Could not process {fname}: {e}")
