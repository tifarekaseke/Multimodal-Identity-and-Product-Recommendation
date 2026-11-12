import os, subprocess, tempfile
import numpy as np
import pandas as pd
import librosa
import joblib

def _extract(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    feats = {}
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["energy_mean"] = float(np.mean(rms))
    return feats

def verify_voice(audio_path, claimed_speaker, model_path="voice_model.pkl"):
    """Returns a dict: approved (bool), score, predicted, claimed, threshold."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    bundle = joblib.load(model_path)
    model, cols, thr = bundle["model"], bundle["features"], bundle["threshold"]

    # convert any audio format (wav, opus, mp4, m4a, mp3) to 16 kHz mono wav
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    result = subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"Could not convert audio file: {audio_path}")
    y, sr = librosa.load(tmp, sr=16000)
    os.remove(tmp)

    feats = _extract(y, sr)
    x = pd.DataFrame([feats])[cols]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    claimed = claimed_speaker.lower()
    score = float(proba[classes.index(claimed)]) if claimed in classes else 0.0
    predicted = classes[int(np.argmax(proba))]
    return {"approved": score >= thr, "score": round(score, 3),
            "predicted": predicted, "claimed": claimed, "threshold": thr}
