"""
Turns an AI-detected instrument name into playable audio.

All detected instruments of a song are built together with *competitive
masking*: at every moment and frequency, the sound is shared between the
sources (vocals, drums, bass, guitar, piano, other ...) in proportion to how
strongly each one is present. This

  * removes backing vocals / singing that leaked into instrument stems, and
  * gives every instrument only what it dominates, so guitar, piano, synth,
    flute ... sound clearly different instead of like the same BGM.

Where each instrument comes from:
  1. Its own Demucs stem, cleaned (drums, bass, other, and with the 6-stem
     model also guitar and piano).
  2. With only a 2-stem separation (vocals + no_vocals), drums and bass are
     *extracted* from the BGM (percussive part / low band).
  3. Other instruments (flute, strings, brass, synth, organ, sitar ...) get
     their own NON-overlapping frequency range of the cleaned remainder.
"""

import hashlib
import math
import os
import threading

import numpy as np
import soundfile as sf


# Hand percussion vs. drum kit
DRUM_KIT_WORDS = ("drum",)
HAND_PERCUSSION_WORDS = ("percussion", "tabla", "dhol", "mridangam", "ghatam", "kanjira", "pakhawaj", "cymbal", "conga", "bongo")
PERCUSSIVE_WORDS = DRUM_KIT_WORDS + HAND_PERCUSSION_WORDS
PIANO_WORDS = ("piano", "keyboard", "keys")
GUITAR_WORDS = ("guitar",)

# Typical centre of each instrument's energy (Hz). Used to split the
# remaining spectrum between the filtered instruments of one song.
INSTRUMENT_CENTRES = [
    ("cello", 220),
    ("organ", 330),
    ("harmonium", 420),
    ("saxophone", 520),
    ("sax", 520),
    ("trumpet", 700),
    ("brass", 650),
    ("veena", 800),
    ("guitar", 850),
    ("sitar", 1000),
    ("piano", 1100),
    ("keyboard", 1100),
    ("harp", 1300),
    ("shehnai", 1400),
    ("nadaswaram", 1400),
    ("strings", 1600),
    ("harmonica", 1700),
    ("violin", 1900),
    ("sarangi", 1900),
    ("flute", 2100),
    ("bansuri", 2100),
    ("santoor", 2400),
    ("synth", 3000),
]
DEFAULT_CENTRE = 1200
LOW_EDGE = 180.0      # below this belongs to bass
HIGH_EDGE = 9000.0
BASS_CUTOFF = 220.0

N_FFT = 4096
HOP = 1024

POWER = 3.0            # >2 = sharper decisions, more difference between instruments
VOCAL_WEIGHT = 2.0     # extra push to keep singing out of instruments
GATE_DB = -40.0        # drop sound more than 40 dB below the song at that frequency (faint bleed)
MAX_BOOST_DB = 12.0    # quiet instruments may be raised, but never blown up to full volume
PRESENT_DB = -18.0     # instrument level vs. the whole BGM
QUIET_DB = -32.0
BLOCK_SECONDS = 20.0   # processed in blocks so long songs fit in memory
OVERLAP_SECONDS = 2.0
BUILD_VERSION = "v4"
import json


def _usable(path):
    return bool(path) and os.path.isfile(path)


def _has(words, lower):
    return any(word in lower for word in words)


def _centre(name):
    lower = name.lower()
    for key, centre in INSTRUMENT_CENTRES:
        if key in lower:
            return centre
    return DEFAULT_CENTRE


def get_instrument_output_dir(audio_id):
    base_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "separated", str(audio_id), "instruments")
    )
    os.makedirs(base_folder, exist_ok=True)
    return base_folder


def _fingerprint(*parts):
    text = "|".join(str(p) for p in parts)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def _file_id(path):
    st = os.stat(path)
    return f"{os.path.abspath(path)}:{st.st_size}:{int(st.st_mtime)}"


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


# ------------------------------------------------------------------
# Classification: where does this instrument's audio come from?
# ------------------------------------------------------------------

def classify_instrument(name, stems):
    """Return (kind, detail).

    kind: "stem"     -> vocals / bgm, played exactly as separated
          "clean"    -> a Demucs stem (drums, bass, other, guitar, piano)
                        with vocal bleed and other instruments' leftovers removed
          "derived"  -> detail is "drums" | "bass" | "hand_percussion" (computed)
          "filtered" -> own frequency range of the cleaned instrument remainder
    """
    lower = (name or "").lower().strip()
    stems = stems or {}

    if ("vocal" in lower or "voice" in lower) and _usable(stems.get("vocals")):
        return "stem", "vocals"
    if lower in ("bgm", "accompaniment", "no_vocals", "instrumental"):
        return "stem", "bgm" if _usable(stems.get("bgm")) else "other"
    if lower == "other" and _usable(stems.get("other")):
        return "clean", "other"

    if _has(DRUM_KIT_WORDS, lower):
        return ("clean", "drums") if _usable(stems.get("drums")) else ("derived", "drums")
    if _has(HAND_PERCUSSION_WORDS, lower):
        return "derived", "hand_percussion"
    if "bass" in lower:
        return ("clean", "bass") if _usable(stems.get("bass")) else ("derived", "bass")
    if _has(GUITAR_WORDS, lower) and _usable(stems.get("guitar")):
        return "clean", "guitar"
    if _has(PIANO_WORDS, lower) and _usable(stems.get("piano")):
        return "clean", "piano"
    return "filtered", None


def describe_instrument_source(name, stems):
    """Short human label shown on the Separation page card."""
    kind, detail = classify_instrument(name, stems)
    if kind == "stem":
        return "stem", f"Real Demucs {detail} stem"
    if kind == "clean":
        return "stem", f"Demucs {detail} stem · vocals removed"
    if detail == "hand_percussion":
        return "derived", "Percussion hits (high part of the drums)"
    if kind == "derived":
        return "derived", f"{detail.capitalize()} extracted from the BGM · vocals removed"
    return "filtered", "Approximate · own frequency range, vocals removed"


# ------------------------------------------------------------------
# DSP helpers
# ------------------------------------------------------------------

def _read(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2], sr


def _write(path, data, sr, peak_target=0.85, gain=None):
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if gain is None:
        gain = peak_target / peak if peak > 1e-4 else 1.0
    data = data * gain
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.99:
        data = data * (0.99 / peak)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".part.wav"
    sf.write(temp_path, data.astype(np.float32), sr)
    os.replace(temp_path, path)   # never stream a half-written file
    return path


def _freqs(sr):
    return np.fft.rfftfreq(N_FFT, 1.0 / sr)[:, None]


def _soft_band(freqs, low, high, width_octaves=0.15):
    """1 inside [low, high], smooth log-frequency fade outside."""
    f = np.maximum(freqs, 1.0)
    rise = 1.0 / (1.0 + np.exp(-(np.log2(f / low)) / (width_octaves / 4)))
    fall = 1.0 / (1.0 + np.exp((np.log2(f / high)) / (width_octaves / 4)))
    return (rise * fall).astype(np.float32)


def _partition(names):
    """Split LOW_EDGE..HIGH_EDGE into one non-overlapping band per instrument."""
    ordered = sorted(set(names), key=lambda n: (_centre(n), n.lower()))
    centres = [_centre(n) for n in ordered]
    # instruments sharing a centre get neighbouring slices around it
    adjusted = []
    for i, c in enumerate(centres):
        same_before = sum(1 for x in centres[:i] if x == c)
        adjusted.append(c * (2 ** (same_before * 0.35)))
    edges = [LOW_EDGE]
    for a, b in zip(adjusted, adjusted[1:]):
        edges.append(math.sqrt(a * b))
    edges.append(HIGH_EDGE)
    return {name: (edges[i], edges[i + 1]) for i, name in enumerate(ordered)}


def _competition(mags, weights=None):
    """Soft 'winner takes most' masks for a dict of magnitude spectrograms."""
    weights = weights or {}
    powered = {k: (weights.get(k, 1.0) * m) ** POWER for k, m in mags.items()}
    total = sum(powered.values()) + 1e-12
    return {k: (v / total).astype(np.float32) for k, v in powered.items()}


def _load_all(paths):
    arrays, rate = {}, None
    for key, path in paths.items():
        data, sr = _read(path)
        if rate is not None and sr != rate:
            import librosa
            data = librosa.resample(data.T, orig_sr=sr, target_sr=rate).T.astype(np.float32)
        arrays[key] = data
        rate = rate or sr
    length = min(a.shape[0] for a in arrays.values())
    return {k: a[:length] for k, a in arrays.items()}, rate, length


def _block_windows(length, sr):
    """Overlapping blocks with linear cross-fades that sum to 1."""
    block = int(BLOCK_SECONDS * sr)
    overlap = int(OVERLAP_SECONDS * sr)
    step = block - overlap
    starts = [0]
    while starts[-1] + block < length:
        starts.append(starts[-1] + step)
    for i, start in enumerate(starts):
        end = min(start + block, length)
        weight = np.ones(end - start, dtype=np.float32)
        fade = min(overlap, end - start)
        if i > 0:
            weight[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if i < len(starts) - 1:
            weight[-fade:] = np.minimum(weight[-fade:], np.linspace(1.0, 0.0, fade, dtype=np.float32))
        yield start, end, weight[:, None]


# ------------------------------------------------------------------
# Whole-song builder
# ------------------------------------------------------------------

def _build_song(stems, jobs, out_paths):
    """jobs: {instrument name: (kind, detail)}. Writes every requested file."""
    import librosa

    sources = {k: stems[k] for k in ("vocals", "drums", "bass", "other", "guitar", "piano") if _usable(stems.get(k))}
    two_stem = not any(k in sources for k in ("drums", "bass", "other"))
    if two_stem or not sources:
        sources["bgm"] = stems.get("bgm") or stems.get("other")
    arrays, sr, length = _load_all(sources)
    base_key = "bgm" if "bgm" in arrays else "other"

    filtered = sorted([n for n, (kind, _) in jobs.items() if kind == "filtered"], key=str.lower)
    bands = _partition(filtered) if filtered else {}
    freqs = _freqs(sr)
    band_masks = {n: _soft_band(freqs, lo, min(hi, sr / 2 - 100), width_octaves=0.08) for n, (lo, hi) in bands.items()}
    bass_band = _soft_band(freqs, 30.0, BASS_CUTOFF)
    high_band = _soft_band(freqs, 1500.0, 14000.0, width_octaves=0.4)
    above_bass = 1.0 - bass_band

    outputs = {n: np.zeros((length, 2), dtype=np.float32) for n in jobs}

    for start, end, weight in _block_windows(length, sr):
        specs = {
            k: [librosa.stft(np.ascontiguousarray(a[start:end, ch]), n_fft=N_FFT, hop_length=HOP) for ch in range(2)]
            for k, a in arrays.items()
        }
        mags = {k: 0.5 * (np.abs(v[0]) + np.abs(v[1])) for k, v in specs.items()}
        masks = _competition(mags, {"vocals": VOCAL_WEIGHT})
        # Noise gate: whatever is far below the whole song at that
        # frequency is leakage (mostly faint singing) -> remove it.
        mix_level = sum(mags.values())
        floor = np.max(mix_level, axis=1, keepdims=True) * (10 ** (GATE_DB / 20))
        for k in masks:
            if k != "vocals":
                masks[k] = masks[k] * ((mags[k] * masks[k]) > floor)

        # Instrument-only remainder used for derived / filtered instruments
        base_mag = mags[base_key] * masks[base_key]
        harmonic, percussive = librosa.decompose.hpss(base_mag, mask=True, margin=1.5)
        harmonic = harmonic.astype(np.float32)
        percussive = percussive.astype(np.float32)

        drums_perc = None
        for name, (kind, detail) in jobs.items():
            if kind == "clean":
                src, mask = detail, masks[detail]
            elif kind == "derived" and detail == "hand_percussion":
                if "drums" in specs:
                    if drums_perc is None:
                        _, drums_perc = librosa.decompose.hpss(mags["drums"] * masks["drums"], mask=True, margin=1.5)
                        drums_perc = drums_perc.astype(np.float32)
                    src, mask = "drums", masks["drums"] * drums_perc * high_band
                else:
                    src, mask = base_key, masks[base_key] * percussive * high_band
            elif kind == "derived" and detail == "drums":
                src, mask = base_key, masks[base_key] * percussive
            elif kind == "derived" and detail == "bass":
                src, mask = base_key, masks[base_key] * harmonic * bass_band
            else:  # filtered melodic instrument
                src, mask = base_key, masks[base_key] * harmonic * above_bass * band_masks[name]
            block = np.stack([
                librosa.istft(specs[src][ch] * mask, hop_length=HOP, length=end - start)
                for ch in range(2)
            ], axis=1)
            outputs[name][start:end] += block * weight

    # One shared gain for the song (so loud/quiet instruments stay honest),
    # plus a limited boost for quiet ones so they are still audible.
    reference = arrays.get("bgm")
    if reference is None:
        reference = sum(a for k, a in arrays.items() if k != "vocals")
    ref_rms = float(np.sqrt(np.mean(reference ** 2))) + 1e-9
    ref_peak = float(np.max(np.abs(reference))) + 1e-9
    song_gain = 0.85 / ref_peak

    levels = {}
    for name, data in outputs.items():
        rms = float(np.sqrt(np.mean(data ** 2))) + 1e-12
        level_db = 20 * math.log10(rms / ref_rms)
        levels[name] = round(level_db, 1)
        peak = float(np.max(np.abs(data))) + 1e-12
        boost = min(10 ** (MAX_BOOST_DB / 20), 0.85 / (peak * song_gain))
        _write(out_paths[name], data, sr, gain=song_gain * max(boost, 1.0))

    info_path = os.path.join(os.path.dirname(next(iter(out_paths.values()))), "levels.json")
    try:
        existing = json.load(open(info_path, encoding="utf-8")) if os.path.isfile(info_path) else {}
    except Exception:
        existing = {}
    for name, path in out_paths.items():
        existing[name.lower()] = {"file": os.path.basename(path), "level_db": levels[name]}
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=1)
    return out_paths


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def presence_label(level_db):
    if level_db is None:
        return None, None
    if level_db >= PRESENT_DB:
        return "present", "Clearly present in this song"
    if level_db >= QUIET_DB:
        return "quiet", "Quiet in this song"
    return "absent", "Barely in this song — the AI detection may be wrong"


def get_instrument_presence(audio_id, instrument_name, stems_dict=None):
    """Level of an already-built instrument vs. the song's BGM (or None)."""
    info_path = os.path.join(get_instrument_output_dir(audio_id), "levels.json")
    if not os.path.isfile(info_path):
        return None
    try:
        info = json.load(open(info_path, encoding="utf-8")).get(str(instrument_name).strip().lower())
    except Exception:
        return None
    if not info:
        return None
    if not os.path.isfile(os.path.join(os.path.dirname(info_path), info.get("file", ""))):
        return None
    status, label = presence_label(info.get("level_db"))
    return {"level_db": info.get("level_db"), "presence": status, "presence_label": label}


_locks_guard = threading.Lock()
_path_locks = {}


def _lock_for(key):
    with _locks_guard:
        if key not in _path_locks:
            _path_locks[key] = threading.Lock()
        return _path_locks[key]


def _detected_names(audio_id):
    try:
        from services.instrument_detection import get_saved_instrument_detections
        return [
            (d.get("instrument") or d.get("instrument_name") or "").strip()
            for d in get_saved_instrument_detections(audio_id)
        ]
    except Exception as error:
        print("Could not read detections for isolation:", error)
        return []


def get_or_create_isolated_instrument(audio_id, instrument_name, stems_dict=None, group_names=None):
    """Return a playable WAV path for a detected instrument (or None).

    All detected instruments of a song are built together in one pass (they
    compete for the same sound), then cached; later calls return instantly.
    """
    if not instrument_name:
        return None

    clean_name = str(instrument_name).strip()

    if stems_dict is None:
        from services.instrument_separation import find_stems_for_audio
        stems_dict = find_stems_for_audio(audio_id) or {}
    if not stems_dict:
        return None

    kind, detail = classify_instrument(clean_name, stems_dict)
    if kind == "stem":
        path = stems_dict.get(detail) or (stems_dict.get("other") if detail == "bgm" else None)
        return path if _usable(path) else None

    if not (_usable(stems_dict.get("other")) or _usable(stems_dict.get("bgm"))):
        return None

    names = group_names if group_names is not None else _detected_names(audio_id)
    jobs = {}
    for n in list(names) + [clean_name]:
        n = (n or "").strip()
        if not n or n.lower() in {j.lower() for j in jobs}:
            continue
        job = classify_instrument(n, stems_dict)
        if job[0] != "stem":
            jobs[n] = job
    target_name = next(n for n in jobs if n.lower() == clean_name.lower())

    source_ids = [_file_id(stems_dict[k]) for k in sorted(stems_dict) if k != "bgm" and _usable(stems_dict.get(k))]
    key = _fingerprint(BUILD_VERSION, *source_ids, *sorted(n.lower() for n in jobs))
    out_dir = get_instrument_output_dir(audio_id)
    out_paths = {n: os.path.join(out_dir, f"{_safe(n)}_{key}.wav") for n in jobs}
    target = out_paths[target_name]

    with _lock_for(key):
        if _usable(target) and os.path.getsize(target) > 1000:
            return target
        try:
            _build_song(stems_dict, jobs, out_paths)
            return target if _usable(target) else None
        except Exception as error:
            print(f"Error isolating {clean_name} for audio {audio_id}: {error}")
            return None


def isolate_instrument_from_audio(source_path, instrument_name, output_path):
    """Kept for older callers: isolate one instrument from a single file."""
    _build_song({"other": source_path}, {instrument_name: ("filtered", None)}, {instrument_name: output_path})
    return output_path
