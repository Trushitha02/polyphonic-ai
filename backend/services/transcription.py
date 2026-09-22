import os

import librosa
import numpy as np


def _group_pitch_frames(pitches, voiced, times, min_frames=2):
	notes = []
	start = None
	values = []

	for index, pitch in enumerate(pitches):
		is_valid = np.isfinite(pitch) and voiced[index] >= 0.45
		if is_valid:
			if start is None:
				start = index
				values = []
			values.append(float(pitch))
			continue

		if start is not None and len(values) >= min_frames:
			notes.append((start, values))
		start = None
		values = []

	if start is not None and len(values) >= min_frames:
		notes.append((start, values))

	results = []
	for start_index, values in notes:
		pitch = float(np.median(values))
		note_name = librosa.hz_to_note(pitch).replace("♯", "#")
		results.append({
			"note": note_name,
			"pitch": round(pitch, 2),
			"time": round(float(times[start_index]), 3),
			"duration": round(float(times[min(start_index + len(values), len(times) - 1)] - times[start_index]), 3),
		})
	return results


def transcribe_audio(audio_path, instrument=None):
	if not os.path.isfile(audio_path):
		raise FileNotFoundError(f"Audio file not found: {audio_path}")

	audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
	if audio.size == 0:
		return {
			"audio_path": os.path.abspath(audio_path),
			"instrument": instrument,
			"notes": [],
			"message": "The selected audio file is empty.",
		}

	pitches, voiced, _ = librosa.pyin(
		audio,
		fmin=librosa.note_to_hz("C2"),
		fmax=librosa.note_to_hz("C7"),
		sr=sample_rate,
		frame_length=2048,
		hop_length=512,
	)
	times = librosa.times_like(pitches, sr=sample_rate, hop_length=512)
	notes = _group_pitch_frames(pitches, voiced, times)

	return {
		"audio_path": os.path.abspath(audio_path),
		"instrument": instrument,
		"notes": notes,
		"message": f"Detected {len(notes)} musical notes.",
	}
