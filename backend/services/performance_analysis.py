import os

import librosa
import numpy as np


def _similarity(first, second):
	first_norm = np.linalg.norm(first)
	second_norm = np.linalg.norm(second)
	if first_norm == 0 or second_norm == 0:
		return 0.0
	return float(np.dot(first, second) / (first_norm * second_norm))


def _score(value):
	return round(float(np.clip(value, 0, 1) * 100), 1)


def analyze_performance(reference_audio, performance_audio):
	if not os.path.isfile(reference_audio) or not os.path.isfile(performance_audio):
		raise FileNotFoundError("Reference or performance audio file was not found")

	reference, sample_rate = librosa.load(reference_audio, sr=22050, mono=True)
	performance, _ = librosa.load(performance_audio, sr=sample_rate, mono=True)

	if not reference.size or not performance.size:
		raise ValueError("Reference or performance audio is empty")

	reference_chroma = librosa.feature.chroma_stft(y=reference, sr=sample_rate)
	performance_chroma = librosa.feature.chroma_stft(y=performance, sr=sample_rate)
	reference_onset = librosa.onset.onset_strength(y=reference, sr=sample_rate)
	performance_onset = librosa.onset.onset_strength(y=performance, sr=sample_rate)

	chroma_length = min(reference_chroma.shape[1], performance_chroma.shape[1])
	onset_length = min(reference_onset.size, performance_onset.size)

	pitch_score = _score(_similarity(
		reference_chroma[:, :chroma_length].mean(axis=1),
		performance_chroma[:, :chroma_length].mean(axis=1),
	))

	note_matches = np.argmax(reference_chroma[:, :chroma_length], axis=0) == np.argmax(
		performance_chroma[:, :chroma_length], axis=0
	)
	note_score = round(float(note_matches.mean() * 100), 1) if chroma_length else 0.0

	rhythm_score = _score(_similarity(
		reference_onset[:onset_length],
		performance_onset[:onset_length],
	))

	reference_tempo = float(np.asarray(librosa.feature.tempo(y=reference, sr=sample_rate)).reshape(-1)[0])
	performance_tempo = float(np.asarray(librosa.feature.tempo(y=performance, sr=sample_rate)).reshape(-1)[0])
	tempo_score = 1 - min(abs(reference_tempo - performance_tempo) / max(reference_tempo, 1), 1)
	duration_score = 1 - min(
		abs(librosa.get_duration(y=reference) - librosa.get_duration(y=performance))
		/ max(librosa.get_duration(y=reference), 1),
		1,
	)
	timing_score = _score((tempo_score * 0.6) + (duration_score * 0.4))

	overall_score = round(
		(pitch_score * 0.3)
		+ (rhythm_score * 0.25)
		+ (timing_score * 0.2)
		+ (note_score * 0.25),
		1,
	)

	return {
		"reference_audio": reference_audio,
		"performance_audio": performance_audio,
		"pitch_score": pitch_score,
		"rhythm_score": rhythm_score,
		"timing_score": timing_score,
		"note_score": note_score,
		"overall_score": overall_score,
		"message": "Performance compared with the selected reference track.",
	}
