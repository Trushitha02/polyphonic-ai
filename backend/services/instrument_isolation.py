import os
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
            "separated_audio",
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
    sf.write(output_path, filtered.astype(np.float32), sr)
    return output_path


def get_or_create_isolated_instrument(audio_id, instrument_name, stems_dict=None):
    """
    Resolves or extracts the isolated audio path for a given instrument.
    - Drums / Percussion -> drums.wav
    - Bass -> bass.wav
    - Vocals / Voice -> vocals.wav
    - Other instruments -> isolated from other.wav (or bgm.wav) and cached.
    """
    if not instrument_name:
        return None

    clean_name = instrument_name.strip()
    lower = clean_name.lower()

    if stems_dict is None:
        from services.instrument_separation import scan_existing_stems
        base_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "separated_audio", str(audio_id))
        )
        stems_dict = scan_existing_stems(base_folder)

    # Direct stem mappings
    if "drum" in lower or "percussion" in lower or "tabla" in lower or "dhol" in lower or "mridangam" in lower:
        drums_path = stems_dict.get("drums")
        if drums_path and os.path.isfile(drums_path):
            return drums_path

    if "bass" in lower:
        bass_path = stems_dict.get("bass")
        if bass_path and os.path.isfile(bass_path):
            return bass_path

    if "vocal" in lower or "voice" in lower:
        vocals_path = stems_dict.get("vocals")
        if vocals_path and os.path.isfile(vocals_path):
            return vocals_path

    if lower in ("bgm", "accompaniment"):
        bgm_path = stems_dict.get("bgm") or stems_dict.get("other")
        if bgm_path and os.path.isfile(bgm_path):
            return bgm_path

    # Check cached isolated file
    out_dir = get_instrument_output_dir(audio_id)
    safe_filename = "".join([c if c.isalnum() else "_" for c in clean_name]) + ".wav"
    cached_path = os.path.join(out_dir, safe_filename)

    if os.path.isfile(cached_path) and os.path.getsize(cached_path) > 1000:
        return cached_path

    # Source stem to isolate from: preferably other.wav, then bgm.wav
    source_stem = stems_dict.get("other")
    if not source_stem or not os.path.isfile(source_stem):
        source_stem = stems_dict.get("bgm")

    if not source_stem or not os.path.isfile(source_stem):
        return None

    try:
        isolated_path = isolate_instrument_from_audio(source_stem, clean_name, cached_path)
        return isolated_path
    except Exception as e:
        print(f"Error isolating instrument {clean_name} for audio {audio_id}: {e}")
        return None
