# ============================================================
# PART 1 — CONNECT TO DRIVE (Safe Remount)
# ============================================================
from google.colab import drive
import os, shutil

mount_point = '/content/drive'

# Unmount if previously mounted
if os.path.exists(mount_point):
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    if os.path.isdir(mount_point):
        shutil.rmtree(mount_point, ignore_errors=True)

# Fresh mount
drive.mount(mount_point, force_remount=True)
print("✅ Google Drive mounted successfully and ready to use.")


# ============================================================
# PART 2 — INSTALLATION (Demucs + Noise Filtering)
# ============================================================
!pip install -q demucs==4.0.1 torch torchvision torchaudio
!pip install -q librosa==0.10.2 soundfile numpy scipy tqdm ffmpeg-python noisereduce

print("✅ Installation complete — Demucs + noise filtering ready.")


# ============================================================
# PART 3 — MAIN PROCESS (Demucs + Adaptive Noise Reduction)
# ============================================================
from google.colab import files
import torchaudio, torch, soundfile as sf, numpy as np, librosa, noisereduce as nr, shutil, os
from demucs.pretrained import get_model
from demucs.apply import apply_model
from scipy.signal import stft, istft

# Step 1 — Upload noisy song
print("🎵 Please upload your noisy song (.mp3, .wav, .opus, etc.)...")
uploaded = files.upload()
input_audio = list(uploaded.keys())[0]
print(f"✅ Uploaded file: {input_audio}")

# Safe mono load
waveform, sr = torchaudio.load(input_audio)
waveform = waveform.mean(dim=0, keepdim=True)     # convert to mono [1, samples]
waveform = waveform.unsqueeze(0).repeat(1, 2, 1)  # Demucs expects stereo

# Step 2 — Demucs cleaning
print("🔹 Running Demucs cleaning (multi-stage)...")
model = get_model('htdemucs').cpu()

with torch.no_grad():
    out = apply_model(model, waveform, device='cpu')

# 🔧 Fix: Convert to numpy & flatten correctly
# Demucs output: [batch, sources, channels, samples]
out_np = out.squeeze().cpu().numpy()  # → [sources, channels, samples] or [channels, samples]

if out_np.ndim == 3:  # multiple sources
    out_np = out_np.mean(axis=0)      # average all sources

if out_np.ndim == 2:  # stereo → mono
    out_np = out_np.mean(axis=0)

main_audio = out_np.astype(np.float32)
main_audio = np.nan_to_num(main_audio)
main_audio /= np.max(np.abs(main_audio)) + 1e-9

# ✅ Safe save
sf.write("/content/clean_stage1_demucs.wav", main_audio, int(sr),
         format='WAV', subtype='PCM_16')
print("✅ Stage 1 complete — Demucs separation saved successfully.")


# Step 3 — Adaptive noise reduction
print("🔹 Applying adaptive noise reduction...")
clean_refined = nr.reduce_noise(y=main_audio.flatten(), sr=sr,
                                stationary=False, prop_decrease=0.8)
sf.write("/content/final_cleaned.wav", clean_refined.astype(np.float32),
         int(sr), format='WAV', subtype='PCM_16')
print("✅ Stage 2 complete — noise-reduced audio saved.")

# Step 4 — Background reconstruction (residual masking)
print("🔹 Extracting background / environmental noise...")
orig_audio, _ = librosa.load(input_audio, sr=sr, mono=True)

min_len = min(len(orig_audio), len(clean_refined))
orig_audio = orig_audio[:min_len]
clean_refined = clean_refined[:min_len]

# Frequency-domain residual
f, t, Zx = stft(orig_audio, fs=sr, nperseg=1024)
_, _, Zc = stft(clean_refined, fs=sr, nperseg=1024)

mask_clean = np.abs(Zc) / (np.abs(Zx) + 1e-9)
mask_bg = np.clip(1 - mask_clean, 0, 1)
Zbg = Zx * mask_bg

_, background = istft(Zbg, fs=sr)
background = np.nan_to_num(background)
background /= np.max(np.abs(background)) + 1e-9
background = np.expand_dims(background.astype(np.float32), axis=1)

sf.write("/content/background_noise.wav", background, int(sr),
         format='WAV', subtype='PCM_16')
print("✅ Stage 3 complete — background noise isolated.")

# Step 5 — Save to Drive
drive_path = "/content/drive/MyDrive/project"
os.makedirs(drive_path, exist_ok=True)
shutil.copy("/content/final_cleaned.wav", f"{drive_path}/final_cleaned.wav")
shutil.copy("/content/background_noise.wav", f"{drive_path}/background_noise.wav")

print("\n✅ All processing complete!")
print(f"🎧 Music-only file: {drive_path}/final_cleaned.wav")
print(f"🌫️ Background-only file: {drive_path}/background_noise.wav")
