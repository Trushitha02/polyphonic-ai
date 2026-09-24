import hashlib
import os
import threading
import soundfile as sf
import numpy as np
import scipy.signal as signal

# ============================================================
# INSTRUMENT FILTER PRESETS
# Tuned acoustic frequency ranges and formant profiles
# ============================================================

FILTER_PROFILES = {
    "guitar": {
        "bandpass": [90, 4200],
        "order": 4,
        "peak_freq": 2200,
        "peak_gain": 0.35,
        "q": 2.0
    },
    "acoustic guitar": {
        "bandpass": [85, 4000],
        "order": 4,
        "peak_freq": 2000,
        "peak_gain": 0.35,
        "q": 2.0
    },
    "sitar": {
        "bandpass": [100, 4500],
        "order": 4,
        "peak_freq": 2400,
        "peak_gain": 0.4,
        "q": 2.5
    },
    "veena": {
        "bandpass": [80, 4200],
        "order": 4,
        "peak_freq": 1800,
        "peak_gain": 0.35,
        "q": 2.0
    },
    "piano": {
        "bandpass": [50, 4800],
        "order": 4,
        "peak_freq": 1200,
        "peak_gain": 0.2,
        "q": 1.5
    },
    "keyboard": {
        "bandpass": [55, 5200],
        "order": 4,
        "peak_freq": 1500,
        "peak_gain": 0.2,
        "q": 1.5
    },
    "flute": {
        "bandpass": [280, 3400],
        "order": 4,
        "peak_freq": 1400,
        "peak_gain": 0.45,
        "q": 2.2
    },
    "bansuri": {
        "bandpass": [260, 3200],
        "order": 4,
        "peak_freq": 1300,
        "peak_gain": 0.45,
        "q": 2.2
    },
    "strings": {
        "bandpass": [180, 6000],
        "order": 4,
        "peak_freq": 2800,
        "peak_gain": 0.3,
        "q": 1.8
    },
    "violin": {
        "bandpass": [200, 6500],
        "order": 4,
        "peak_freq": 3000,
        "peak_gain": 0.35,
        "q": 2.0
    },
    "cello": {
        "bandpass": [100, 4000],
        "order": 4,
        "peak_freq": 1200,
        "peak_gain": 0.3,
        "q": 1.8
    },
    "brass": {
        "bandpass": [160, 3800],
        "order": 4,
        "peak_freq": 1500,
        "peak_gain": 0.45,
        "q": 2.0
    },
    "trumpet": {
        "bandpass": [180, 4200],
        "order": 4,
        "peak_freq": 1600,
        "peak_gain": 0.5,
        "q": 2.2
    },
    "saxophone": {
        "bandpass": [140, 3400],
        "order": 4,
        "peak_freq": 1400,
        "peak_gain": 0.4,
        "q": 1.8
    },
    "synth": {
        "bandpass": [260, 7500],
        "order": 4,
        "peak_freq": 3200,
        "peak_gain": 0.35,
        "q": 1.6
    },
    "organ": {
        "bandpass": [70, 3200],
        "order": 4,
        "peak_freq": 800,
        "peak_gain": 0.25,
        "q": 1.5
    },
    "harmonium": {
        "bandpass": [80, 3000],
        "order": 4,
        "peak_freq": 900,
        "peak_gain": 0.3,
        "q": 1.6
    },
    "harmonica": {
        "bandpass": [380, 4200],
        "order": 4,
        "peak_freq": 1800,
        "peak_gain": 0.4,
        "q": 2.0
    },
    "harp": {
        "bandpass": [120, 5000],
        "order": 4,
        "peak_freq": 2200,
        "peak_gain": 0.35,
        "q": 2.0
    }
}


def get_instrument_output_dir(audio_id):
    base_folder = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "separated",
            str(audio_id),
            "instruments"
        )
    )
    os.makedirs(base_folder, exist_ok=True)
    return base_folder


def get_matching_profile(instrument_name):
    clean = (instrument_name or "").lower().strip()
    for key, profile in FILTER_PROFILES.items():
        if key in clean or clean in key:
            return profile
    return {
        "bandpass": [120, 4500],
        "order": 4,
        "peak_freq": 1600,
        "peak_gain": 0.3,
        "q": 1.8
    }


def isolate_instrument_from_audio(source_path, instrument_name, output_path):
    """
    Applies instrument-specific DSP isolation (Butterworth bandpass + resonance shaping + normalization).
    """
    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    data, sr = sf.read(source_path)
    if data.ndim == 1:
        data = data[:, None]

    profile = get_matching_profile(instrument_name)
    low, high = profile["bandpass"]

    # Pitched instruments: keep the harmonic (sustained) part and drop
    # leftover percussive hits before band-passing.
    lower_name = (instrument_name or "").lower()
    if not any(word in lower_name for word in PERCUSSIVE_WORDS):
        try:
            import librosa
            channels = []
            for ch in range(data.shape[1]):
                spec = librosa.stft(data[:, ch].astype(np.float32), n_fft=2048, hop_length=512)
                harmonic, _ = librosa.decompose.hpss(spec, margin=2.0)
                channels.append(librosa.istft(harmonic, hop_length=512, length=data.shape[0]))
            data = np.stack(channels, axis=1)
        except Exception as error:
            print("Harmonic filtering skipped:", error)

    # Clamp bounds to Nyquist limit
    nyquist = sr / 2.0 - 100
    low = max(20, min(low, nyquist - 200))
    high = max(low + 100, min(high, nyquist))

    # 4th order bandpass filter
    sos = signal.butter(profile.get("order", 4), [low, high], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=0)

    # Optional resonant peak filter for formant clarity
    peak_freq = profile.get("peak_freq", 1800)
    peak_gain = profile.get("peak_gain", 0.3)
    q = profile.get("q", 2.0)
    if peak_freq < nyquist and peak_gain > 0:
        try:
            b, a = signal.iirpeak(peak_freq, q, fs=sr)
            peaked = signal.lfilter(b, a, filtered, axis=0)
            filtered = filtered + (peak_gain * peaked)
        except Exception:
            pass

    # Normalize peak amplitude to 0.85
    peak = np.max(np.abs(filtered))
    if peak > 1e-4:
        filtered = (filtered / peak) * 0.85

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write to a temp file then swap in, so a half-written file is never streamed.
    temp_path = output_path + ".part.wav"
    sf.write(temp_path, filtered.astype(np.float32), sr)
    os.replace(temp_path, output_path)
    return output_path


PERCUSSIVE_WORDS = ("drum", "percussion", "tabla", "dhol", "mridangam", "ghatam", "kanjira", "pakhawaj", "cymbal")
PIANO_WORDS = ("piano", "keyboard", "keys")
GUITAR_WORDS = ("guitar",)


def _usable(path):
    return bool(path) and os.path.isfile(path)


def get_or_create_isolated_instrument(audio_id, instrument_name, stems_dict=None):
    """
    Resolves or extracts the playable audio path for a detected instrument.
    - Drums / Percussion / Tabla ...   -> drums.wav
    - Bass                               -> bass.wav
    - Vocals / Voice                     -> vocals.wav
    - Guitar / Piano (6-stem Demucs)     -> guitar.wav / piano.wav
    - Other                              -> other.wav
    - Any other instrument               -> isolated from other.wav (or bgm.wav)
      with a harmonic filter + instrument band-pass, cached as WAV.
    """
    if not instrument_name:
        return None

    clean_name = str(instrument_name).strip()
    lower = clean_name.lower()

    if stems_dict is None:
        from services.instrument_separation import find_stems_for_audio
        stems_dict = find_stems_for_audio(audio_id) or {}

    if not stems_dict:
        return None

    # Direct stem mappings
    if any(word in lower for word in PERCUSSIVE_WORDS) and _usable(stems_dict.get("drums")):
        return stems_dict["drums"]

    if "bass" in lower and _usable(stems_dict.get("bass")):
        return stems_dict["bass"]

    if ("vocal" in lower or "voice" in lower) and _usable(stems_dict.get("vocals")):
        return stems_dict["vocals"]

    if any(word in lower for word in GUITAR_WORDS) and _usable(stems_dict.get("guitar")):
        return stems_dict["guitar"]

    if any(word in lower for word in PIANO_WORDS) and _usable(stems_dict.get("piano")):
        return stems_dict["piano"]

    if lower in ("bgm", "accompaniment", "no_vocals", "instrumental"):
        bgm_path = stems_dict.get("bgm") or stems_dict.get("other")
        if _usable(bgm_path):
            return bgm_path

    if lower == "other" and _usable(stems_dict.get("other")):
        return stems_dict["other"]

    # Source stem to isolate from: preferably other.wav, then bgm.wav
    source_stem = stems_dict.get("other")
    if not _usable(source_stem):
        source_stem = stems_dict.get("bgm")
    if not _usable(source_stem):
        return None

    # Cached isolated file. The name includes a fingerprint of the exact
    # source stem, so a re-separated song (or a reused audio_id after a
    # database reset) can never play another song's instrument audio.
    out_dir = get_instrument_output_dir(audio_id)
    stat = os.stat(source_stem)
    fingerprint = hashlib.md5(
        f"{os.path.abspath(source_stem)}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")
    ).hexdigest()[:10]
    safe_filename = "".join([c if c.isalnum() else "_" for c in clean_name]) + f"_{fingerprint}.wav"
    cached_path = os.path.join(out_dir, safe_filename)

    if os.path.isfile(cached_path) and os.path.getsize(cached_path) > 1000:
        return cached_path

    with _path_lock(cached_path):
        # Another request may have finished it while we waited.
        if os.path.isfile(cached_path) and os.path.getsize(cached_path) > 1000:
            return cached_path
        try:
            return isolate_instrument_from_audio(source_stem, clean_name, cached_path)
        except Exception as e:
            print(f"Error isolating instrument {clean_name} for audio {audio_id}: {e}")
            return None


_locks_guard = threading.Lock()
_path_locks = {}


def _path_lock(path):
    with _locks_guard:
        if path not in _path_locks:
            _path_locks[path] = threading.Lock()
        return _path_locks[path]
