import os

import librosa
import numpy as np
from scipy.signal import correlate


SAMPLE_RATE = 22050
HOP = 512
ONSET_TOLERANCE = 0.07  # seconds


def _similarity(first, second):
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def _score(value):
    return round(float(np.clip(value, 0, 1) * 100), 1)


def _find_offset(reference_chroma, performance_chroma):
    """Find where the practice take starts inside the reference (in frames).

    A practice recording is usually one section of the song, so we slide
    it along the reference and keep the position where the harmony
    (chroma) matches best."""
    ref_len = reference_chroma.shape[1]
    perf_len = performance_chroma.shape[1]
    if perf_len >= ref_len:
        return 0

    ref = reference_chroma / (np.linalg.norm(reference_chroma, axis=0, keepdims=True) + 1e-8)
    perf = performance_chroma / (np.linalg.norm(performance_chroma, axis=0, keepdims=True) + 1e-8)
    total = np.zeros(ref_len - perf_len + 1)
    for bin_index in range(ref.shape[0]):
        total += correlate(ref[bin_index], perf[bin_index], mode="valid", method="fft")
    return int(np.argmax(total))


def _onset_f1(reference_onsets, performance_onsets):
    """How many played onsets land on the reference onsets (±70 ms)."""
    if len(reference_onsets) == 0 and len(performance_onsets) == 0:
        return 1.0
    if len(reference_onsets) == 0 or len(performance_onsets) == 0:
        return 0.0
    used = np.zeros(len(reference_onsets), dtype=bool)
    matches = 0
    for onset in performance_onsets:
        distance = np.abs(reference_onsets - onset)
        distance[used] = np.inf
        best = int(np.argmin(distance))
        if distance[best] <= ONSET_TOLERANCE:
            used[best] = True
            matches += 1
    precision = matches / len(performance_onsets)
    recall = matches / len(reference_onsets)
    return 0.0 if matches == 0 else 2 * precision * recall / (precision + recall)


def analyze_performance(reference_audio, performance_audio):
    if not os.path.isfile(reference_audio) or not os.path.isfile(performance_audio):
        raise FileNotFoundError("Reference or performance audio file was not found")

    reference, sample_rate = librosa.load(reference_audio, sr=SAMPLE_RATE, mono=True)
    performance, _ = librosa.load(performance_audio, sr=sample_rate, mono=True)

    if not reference.size or not performance.size:
        raise ValueError("Reference or performance audio is empty")

    performance, _ = librosa.effects.trim(performance, top_db=40)
    if performance.size < sample_rate:
        raise ValueError("The performance recording is too short (less than 1 second of sound)")

    reference_chroma = librosa.feature.chroma_stft(y=reference, sr=sample_rate, hop_length=HOP)
    performance_chroma = librosa.feature.chroma_stft(y=performance, sr=sample_rate, hop_length=HOP)

    # Line the practice take up with the matching section of the reference
    offset = _find_offset(reference_chroma, performance_chroma)
    length = min(performance_chroma.shape[1], reference_chroma.shape[1] - offset)
    ref_chroma = reference_chroma[:, offset:offset + length]
    perf_chroma = performance_chroma[:, :length]

    start_sample = offset * HOP
    ref_segment = reference[start_sample:start_sample + length * HOP]
    perf_segment = performance[:length * HOP]

    # Notes/pitch are judged only where both recordings actually sound
    active = (ref_chroma.max(axis=0) > 0.5) & (perf_chroma.max(axis=0) > 0.5)

    # Pitch: moment-by-moment pitch-class (chroma) similarity
    if np.any(active):
        ref_active = ref_chroma[:, active]
        perf_active = perf_chroma[:, active]
        frame_cosine = np.sum(ref_active * perf_active, axis=0) / (
            np.linalg.norm(ref_active, axis=0) * np.linalg.norm(perf_active, axis=0) + 1e-8
        )
        pitch_score = _score(float(np.mean(frame_cosine)))
    else:
        pitch_score = 0.0

    # Notes: frame-by-frame dominant pitch class agreement
    if np.any(active):
        note_matches = np.argmax(ref_chroma[:, active], axis=0) == np.argmax(perf_chroma[:, active], axis=0)
        note_score = round(float(note_matches.mean() * 100), 1)
    else:
        note_score = 0.0

    # Rhythm: shape of the onset (attack) envelope
    reference_onset = librosa.onset.onset_strength(y=ref_segment, sr=sample_rate, hop_length=HOP)
    performance_onset = librosa.onset.onset_strength(y=perf_segment, sr=sample_rate, hop_length=HOP)
    onset_length = min(reference_onset.size, performance_onset.size)
    rhythm_score = _score(_similarity(reference_onset[:onset_length], performance_onset[:onset_length]))

    # Timing: are the notes played at the right moments + at the right tempo
    ref_onsets = librosa.onset.onset_detect(onset_envelope=reference_onset, sr=sample_rate, hop_length=HOP, units="time")
    perf_onsets = librosa.onset.onset_detect(onset_envelope=performance_onset, sr=sample_rate, hop_length=HOP, units="time")
    onset_accuracy = _onset_f1(np.asarray(ref_onsets), np.asarray(perf_onsets))
    reference_tempo = float(np.asarray(librosa.feature.tempo(onset_envelope=reference_onset, sr=sample_rate, hop_length=HOP)).reshape(-1)[0])
    performance_tempo = float(np.asarray(librosa.feature.tempo(onset_envelope=performance_onset, sr=sample_rate, hop_length=HOP)).reshape(-1)[0])
    tempo_score = 1 - min(abs(reference_tempo - performance_tempo) / max(reference_tempo, 1), 1)
    timing_score = _score((onset_accuracy * 0.7) + (tempo_score * 0.3))

    overall_score = round(
        (pitch_score * 0.3)
        + (rhythm_score * 0.25)
        + (timing_score * 0.2)
        + (note_score * 0.25),
        1,
    )

    matched_at = round(offset * HOP / sample_rate, 2)
    return {
        "reference_audio": reference_audio,
        "performance_audio": performance_audio,
        "pitch_score": pitch_score,
        "rhythm_score": rhythm_score,
        "timing_score": timing_score,
        "note_score": note_score,
        "overall_score": overall_score,
        "matched_section_start": matched_at,
        "compared_seconds": round(length * HOP / sample_rate, 2),
        "reference_tempo": round(reference_tempo, 1),
        "performance_tempo": round(performance_tempo, 1),
        "message": f"Performance compared with the reference section starting at {matched_at:.1f} s.",
    }
