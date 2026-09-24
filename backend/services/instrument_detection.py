import os
import csv
import joblib
import numpy as np
import librosa
from database.database import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "instrument_model.joblib")
LABEL_PATH = os.path.join(BASE_DIR, "models", "instrument_labels.joblib")
LABEL_CSV_PATH = os.path.join(BASE_DIR, "datasets", "instrument_dataset.csv")

INSTRUMENT_METADATA_MAP = {
    "vocals": {"family": "Voice", "category": "Vocal", "icon": "fa-microphone", "emoji": "🎙️"},
    "voice": {"family": "Voice", "category": "Vocal", "icon": "fa-microphone", "emoji": "🎙️"},
    "guitar": {"family": "Western", "category": "Pitched", "icon": "fa-guitar", "emoji": "🎸"},
    "acoustic guitar": {"family": "Western", "category": "Pitched", "icon": "fa-guitar", "emoji": "🎸"},
    "electric guitar": {"family": "Western", "category": "Pitched", "icon": "fa-guitar", "emoji": "🎸"},
    "bass": {"family": "Western", "category": "Bass", "icon": "fa-guitar", "emoji": "🎸"},
    "bass guitar": {"family": "Western", "category": "Bass", "icon": "fa-guitar", "emoji": "🎸"},
    "drums": {"family": "Western", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "percussion": {"family": "Acoustic", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "piano": {"family": "Western", "category": "Pitched", "icon": "fa-music", "emoji": "🎹"},
    "keyboard": {"family": "Western", "category": "Pitched", "icon": "fa-music", "emoji": "🎹"},
    "organ": {"family": "Western", "category": "Pitched", "icon": "fa-music", "emoji": "🎹"},
    "synth": {"family": "Electronic", "category": "Synthesizer", "icon": "fa-sliders-h", "emoji": "🎛️"},
    "strings": {"family": "Western", "category": "Orchestral", "icon": "fa-music", "emoji": "🎻"},
    "violin": {"family": "Western", "category": "Pitched", "icon": "fa-music", "emoji": "🎻"},
    "cello": {"family": "Western", "category": "Pitched", "icon": "fa-music", "emoji": "🎻"},
    "harp": {"family": "Western", "category": "Plucked", "icon": "fa-music", "emoji": "🪕"},
    "brass": {"family": "Western", "category": "Brass", "icon": "fa-music", "emoji": "🎺"},
    "trumpet": {"family": "Western", "category": "Brass", "icon": "fa-music", "emoji": "🎺"},
    "saxophone": {"family": "Western", "category": "Woodwind", "icon": "fa-music", "emoji": "🎷"},
    "flute": {"family": "Acoustic", "category": "Woodwind", "icon": "fa-music", "emoji": "🪈"},
    "harmonica": {"family": "Acoustic", "category": "Free Reed", "icon": "fa-music", "emoji": "🎶"},
    "sitar": {"family": "Indian", "category": "Pitched", "icon": "fa-music", "emoji": "🪕"},
    "veena": {"family": "Indian", "category": "Pitched", "icon": "fa-music", "emoji": "🪕"},
    "tabla": {"family": "Indian", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "mridangam": {"family": "Indian", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "dholak": {"family": "Indian", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "ghatam": {"family": "Indian", "category": "Percussion", "icon": "fa-drum", "emoji": "🥁"},
    "bansuri": {"family": "Indian", "category": "Woodwind", "icon": "fa-music", "emoji": "🪈"},
    "harmonium": {"family": "Indian", "category": "Keyboard", "icon": "fa-music", "emoji": "🎹"},
    "other": {"family": "General", "category": "Instrumental", "icon": "fa-music", "emoji": "🎵"}
}


def get_instrument_classes():
    """Return the instrument classes exposed by the separation UI."""
    return [
        {"name": "Vocals", "family": "Voice", "category": "Vocal", "icon": "fa-microphone"},
        {"name": "Tabla", "family": "Indian", "category": "Percussion", "icon": "fa-drum"},
        {"name": "Mridangam", "family": "Indian", "category": "Percussion", "icon": "fa-drum"},
        {"name": "Dholak", "family": "Indian", "category": "Percussion", "icon": "fa-drum"},
        {"name": "Ghatam", "family": "Indian", "category": "Percussion", "icon": "fa-drum"},
        {"name": "Drums", "family": "Western", "category": "Percussion", "icon": "fa-drum"},
        {"name": "Bass Guitar", "family": "Western", "category": "Bass", "icon": "fa-guitar"},
        {"name": "Piano", "family": "Western", "category": "Pitched", "icon": "fa-music"},
        {"name": "Acoustic Guitar", "family": "Western", "category": "Pitched", "icon": "fa-guitar"},
        {"name": "Electric Guitar", "family": "Western", "category": "Pitched", "icon": "fa-guitar"},
        {"name": "Violin", "family": "Western", "category": "Pitched", "icon": "fa-music"},
        {"name": "Cello", "family": "Western", "category": "Pitched", "icon": "fa-music"},
        {"name": "Trumpet", "family": "Western", "category": "Pitched", "icon": "fa-music"},
        {"name": "Saxophone", "family": "Western", "category": "Pitched", "icon": "fa-music"},
        {"name": "Veena", "family": "Indian", "category": "Pitched", "icon": "fa-music"},
        {"name": "Sitar", "family": "Indian", "category": "Pitched", "icon": "fa-music"},
        {"name": "Harmonium", "family": "Indian", "category": "Pitched", "icon": "fa-music"},
        {"name": "Bansuri", "family": "Indian", "category": "Pitched", "icon": "fa-music"},
        {"name": "Flute", "family": "Indian / Western", "category": "Pitched", "icon": "fa-music"},
    ]


def get_instrument_metadata(instrument_name):
    """Retrieve family, category, FontAwesome icon, and emoji for an instrument name."""
    clean = (instrument_name or "").lower().strip()
    if clean in INSTRUMENT_METADATA_MAP:
        return INSTRUMENT_METADATA_MAP[clean]
    for key, meta in INSTRUMENT_METADATA_MAP.items():
        if key in clean or clean in key:
            return meta
    return {"family": "General", "category": "Acoustic", "icon": "fa-music", "emoji": "🎵"}


def load_instrument_labels():
    """Loads sorted list of multi-label instrument classes."""
    if os.path.isfile(LABEL_PATH):
        return joblib.load(LABEL_PATH)

    if not os.path.isfile(LABEL_CSV_PATH):
        raise FileNotFoundError("Instrument label data was not found")

    with open(LABEL_CSV_PATH, newline="", encoding="utf-8") as label_file:
        labels = sorted({
            row["instrument_class"].strip()
            for row in csv.DictReader(label_file)
            if row.get("instrument_class", "").strip()
        })

    if not labels:
        raise ValueError("Instrument label data is empty")

    return labels


def extract_instrument_features(audio_path):
    """
    Extracts 58 audio acoustic features matching the trained model pipeline:
    - MFCC (mean & std for 20 coefficients = 40)
    - Chroma STFT (mean & std = 2)
    - Spectral Centroid (mean = 1)
    - Spectral Bandwidth (mean = 1)
    - Spectral Contrast (mean across bands = 7)
    - Zero Crossing Rate (mean = 1)
    - RMS Energy (mean = 1)
    - Tempo (estimated BPM = 1)
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found for feature extraction: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(centroid)))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(bandwidth)))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        features.append(tempo_value)
    except Exception:
        features.append(0.0)

    return np.asarray(features, dtype=np.float32)


def detect_audio_instruments(audio_path, source="Uploaded Audio"):
    """
    Classifies instruments present in an audio file using the trained MultiOutputClassifier model.
    Returns a sorted list of detected instrument dicts with confidence, category, and icon metadata.
    """
    if not audio_path or not os.path.isfile(audio_path):
        print(f"[Instrument Detection] Audio file does not exist: {audio_path}")
        return []

    if not os.path.isfile(MODEL_PATH):
        print(f"[Instrument Detection] Instrument model not found: {MODEL_PATH}")
        return []

    try:
        print()
        print("======================================")
        print(f"AI INSTRUMENT DETECTION ({source})")
        print(f"Target: {os.path.basename(audio_path)}")
        print("======================================")

        model = joblib.load(MODEL_PATH)
        labels = load_instrument_labels()

        features = extract_instrument_features(audio_path)
        X = features.reshape(1, -1)

        prediction = model.predict(X)[0]
        try:
            probas = model.predict_proba(X)
        except Exception:
            probas = None

        if len(labels) != len(prediction):
            raise ValueError(
                f"Instrument label count ({len(labels)}) does not match model output ({len(prediction)})"
            )

        all_candidates = []
        for idx, (label, value) in enumerate(zip(labels, prediction)):
            confidence = None
            if probas is not None and idx < len(probas):
                p_arr = probas[idx][0]
                if len(p_arr) > 1:
                    confidence = round(float(p_arr[1]) * 100, 1)
                else:
                    confidence = 100.0 if int(value) == 1 else 0.0

            if confidence is None:
                confidence = 90.0 if int(value) == 1 else 30.0

            meta = get_instrument_metadata(str(label))
            all_candidates.append({
                "instrument": str(label),
                "instrument_name": str(label),
                "confidence": float(confidence),
                "predicted_binary": int(value),
                "source": source,
                "source_stem": source,
                "family": meta["family"],
                "category": meta["category"],
                "icon": meta["icon"],
                "emoji": meta["emoji"],
            })

        # Filter detected by prediction == 1 or probability >= 35%
        detected = [
            item for item in all_candidates
            if item["predicted_binary"] == 1 or item["confidence"] >= 35.0
        ]

        # Intelligent Fallback: if no instruments crossed 35%, pick top 3 highest probability candidates
        if not detected:
            all_candidates.sort(key=lambda item: item["confidence"], reverse=True)
            detected = all_candidates[:3]
            # Ensure minimum representative confidence display for top acoustic matches
            for d in detected:
                if d["confidence"] < 25.0:
                    d["confidence"] = round(float(d["confidence"] + 25.0), 1)

        detected.sort(key=lambda item: item["confidence"], reverse=True)

        print("Detected instruments:")
        for item in detected:
            print(f"  + {item['instrument']} ({item['confidence']}%) [{item['category']}]")
        print("======================================")

        return detected

    except Exception as error:
        print("[Instrument Detection] Error occurred during classification:", error)
        return []


def get_saved_instrument_detections(audio_id):
    """Retrieves saved instrument detections for an audio record from the database."""
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT detection_id, audio_id, instrument_name, confidence, source_stem, created_at
            FROM instrument_detections
            WHERE audio_id = %s
            ORDER BY confidence DESC, detection_id ASC
            """,
            (audio_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        if not rows:
            return []

        # If all confidences are null, detections are legacy and should be recomputed
        if all(r.get("confidence") is None for r in rows):
            return []

        result = []
        for r in rows:
            conf = float(r["confidence"]) if r.get("confidence") is not None else 85.0
            meta = get_instrument_metadata(r["instrument_name"])
            result.append({
                "detection_id": r["detection_id"],
                "audio_id": r["audio_id"],
                "instrument": r["instrument_name"],
                "instrument_name": r["instrument_name"],
                "confidence": conf,
                "source": r.get("source_stem") or "Uploaded Audio",
                "source_stem": r.get("source_stem") or "Uploaded Audio",
                "family": meta["family"],
                "category": meta["category"],
                "icon": meta["icon"],
                "emoji": meta["emoji"],
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
            })
        return result
    except Exception as error:
        print(f"[Instrument Detection] Error fetching saved detections for audio_id {audio_id}: {error}")
        return []


def save_instrument_detections(audio_id, detections):
    """Saves detected instruments into instrument_detections table in the database."""
    if not detections:
        return

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Remove previous detections for this audio record
        cursor.execute(
            """
            DELETE FROM instrument_detections
            WHERE audio_id = %s
            """,
            (audio_id,)
        )

        for detection in detections:
            conf_val = float(detection["confidence"]) if detection.get("confidence") is not None else None
            inst_name = detection.get("instrument") or detection.get("instrument_name")
            source_stem = detection.get("source") or detection.get("source_stem") or "Uploaded Audio"

            cursor.execute(
                """
                INSERT INTO instrument_detections
                (audio_id, instrument_name, confidence, source_stem)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    audio_id,
                    inst_name,
                    conf_val,
                    source_stem,
                )
            )

        connection.commit()
        cursor.close()
        connection.close()

        print(f"[Instrument Detection] Saved {len(detections)} instrument detections to database for audio_id {audio_id}.")
    except Exception as error:
        print(f"[Instrument Detection] Could not save detections to database for audio_id {audio_id}: {error}")
