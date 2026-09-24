import os
import struct

import librosa
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIDI_FOLDER = os.path.join(BASE_DIR, "midi")

PERCUSSIVE_WORDS = ("drum", "percussion", "tabla", "dhol", "mridangam", "ghatam", "kanjira", "pakhawaj", "cymbal")

# Pitch search range per instrument family (narrower = faster + fewer octave errors)
PITCH_RANGES = [
    (("bass",), "E1", "G4"),
    (("flute", "bansuri", "piccolo", "shehnai", "nadaswaram"), "C4", "C7"),
    (("violin", "strings"), "G3", "A7"),
    (("cello",), "C2", "C6"),
    (("trumpet", "brass"), "E3", "C6"),
    (("saxophone", "sax"), "C3", "A5"),
    (("vocal", "voice"), "E2", "C6"),
    (("guitar", "sitar", "veena"), "E2", "E6"),
    (("piano", "keyboard", "organ", "synth", "harmonium", "harmonica", "harp"), "A1", "C7"),
]

# General MIDI programs so the exported MIDI sounds like the instrument
GM_PROGRAMS = {
    "piano": 0, "keyboard": 0, "organ": 19, "harmonium": 20, "harmonica": 22,
    "acoustic guitar": 25, "guitar": 25, "electric guitar": 27, "bass": 33,
    "violin": 40, "cello": 42, "strings": 48, "harp": 46, "trumpet": 56,
    "brass": 61, "saxophone": 65, "flute": 73, "bansuri": 73, "synth": 80,
    "sitar": 104, "veena": 104, "vocal": 52, "voice": 52,
}


def is_percussive(instrument):
    name = (instrument or "").lower()
    return any(word in name for word in PERCUSSIVE_WORDS)


def _pitch_range(instrument):
    name = (instrument or "").lower()
    for words, low, high in PITCH_RANGES:
        if any(word in name for word in words):
            return librosa.note_to_hz(low), librosa.note_to_hz(high)
    return librosa.note_to_hz("C2"), librosa.note_to_hz("C7")


def _gm_program(instrument):
    name = (instrument or "").lower()
    for key in sorted(GM_PROGRAMS, key=len, reverse=True):
        if key in name:
            return GM_PROGRAMS[key]
    return 0


def _group_pitch_frames(pitches, voiced, times, rms, min_frames=3):
    """Merge consecutive voiced frames of the same semitone into notes."""
    notes = []
    start = None
    values = []
    current_midi = None

    def close(end_index):
        if start is not None and len(values) >= min_frames:
            notes.append((start, end_index, list(values)))

    for index, pitch in enumerate(pitches):
        is_valid = np.isfinite(pitch) and voiced[index] >= 0.45
        midi = int(round(librosa.hz_to_midi(pitch))) if is_valid else None

        if is_valid and start is not None and midi == current_midi:
            values.append(float(pitch))
            continue

        close(index)
        start, values, current_midi = (index, [float(pitch)], midi) if is_valid else (None, [], None)

    close(len(pitches))

    peak_rms = float(np.max(rms)) if rms.size else 1.0
    results = []
    for start_index, end_index, vals in notes:
        pitch = float(np.median(vals))
        end_index = min(end_index, len(times) - 1)
        loudness = float(np.mean(rms[start_index:end_index + 1])) / (peak_rms or 1.0)
        results.append({
            "note": librosa.hz_to_note(pitch).replace("♯", "#"),
            "pitch": round(pitch, 2),
            "midi": int(round(librosa.hz_to_midi(pitch))),
            "time": round(float(times[start_index]), 3),
            "duration": round(max(float(times[end_index] - times[start_index]), 0.05), 3),
            "velocity": int(np.clip(40 + loudness * 87, 30, 127)),
        })
    return results


def _transcribe_percussion(audio, sample_rate):
    """Drums/tabla: detect hits (onsets) instead of pitches."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate, units="frames", backtrack=False)
    times = librosa.frames_to_time(onsets, sr=sample_rate)
    strength = onset_env[onsets] if len(onsets) else np.array([])
    peak = float(np.max(strength)) if strength.size else 1.0

    # Split hits into low (kick / bayan) and high (snare, hats / dayan) by spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    notes = []
    for frame, time_value, value in zip(onsets, times, strength):
        c = float(centroid[min(frame, len(centroid) - 1)])
        low = c < 1500
        notes.append({
            "note": "Low hit" if low else "High hit",
            "pitch": round(c, 2),
            "midi": 36 if low else 38,  # GM kick / snare
            "time": round(float(time_value), 3),
            "duration": 0.1,
            "velocity": int(np.clip(40 + (float(value) / (peak or 1.0)) * 87, 30, 127)),
        })
    return notes


def _var_len(value):
    buffer = value & 0x7F
    out = bytearray()
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def write_midi(notes, path, instrument=None, tempo_bpm=120):
    """Write a single-track Standard MIDI File (no extra libraries needed)."""
    ticks_per_beat = 480
    seconds_per_tick = 60.0 / tempo_bpm / ticks_per_beat
    percussion = is_percussive(instrument)
    channel = 9 if percussion else 0

    events = []
    for note in notes:
        start = int(round(note["time"] / seconds_per_tick))
        end = int(round((note["time"] + note["duration"]) / seconds_per_tick))
        pitch = int(np.clip(note.get("midi", 60), 0, 127))
        velocity = int(np.clip(note.get("velocity", 90), 1, 127))
        events.append((start, 1, bytes([0x90 | channel, pitch, velocity])))
        events.append((max(end, start + 1), 0, bytes([0x80 | channel, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    tempo = int(60_000_000 / tempo_bpm)
    track += b"\x00\xFF\x51\x03" + tempo.to_bytes(3, "big")
    name = (instrument or "Polyphonic AI").encode("ascii", "ignore")[:100]
    track += b"\x00\xFF\x03" + _var_len(len(name)) + name
    if not percussion:
        track += b"\x00" + bytes([0xC0 | channel, _gm_program(instrument)])

    last = 0
    for tick, _, data in events:
        track += _var_len(tick - last) + data
        last = tick
    track += b"\x00\xFF\x2F\x00"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
    return path


def transcribe_audio(audio_path, instrument=None, midi_path=None):
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sample_rate = 16000
    audio, sample_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return {
            "audio_path": os.path.abspath(audio_path),
            "instrument": instrument,
            "notes": [],
            "message": "The selected audio file is empty.",
        }

    tempo = librosa.feature.tempo(y=audio, sr=sample_rate)
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0]) or 120.0

    if is_percussive(instrument):
        notes = _transcribe_percussion(audio, sample_rate)
        kind = "hits"
    else:
        hop = 512
        fmin, fmax = _pitch_range(instrument)
        pitches, voiced, _ = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=2048,
            hop_length=hop,
        )
        times = librosa.times_like(pitches, sr=sample_rate, hop_length=hop)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0][: len(pitches)]
        notes = _group_pitch_frames(pitches, voiced, times, rms)
        kind = "notes"

    result = {
        "audio_path": os.path.abspath(audio_path),
        "instrument": instrument,
        "tempo_bpm": round(tempo_bpm, 1),
        "notes": notes,
        "note_count": len(notes),
        "kind": kind,
        "message": f"Detected {len(notes)} {'drum hits' if kind == 'hits' else 'musical notes'}.",
    }

    if midi_path:
        write_midi(notes, midi_path, instrument, tempo_bpm=round(tempo_bpm) or 120)
        result["midi_path"] = midi_path

    return result
