import os
import joblib
import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "babyslakh_16k"
)

LABEL_CSV = os.path.join(
    BASE_DIR,
    "datasets",
    "instrument_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "instrument_model.joblib"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "instrument_labels.joblib"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
        duration=30
    )

    if len(y) < sr:
        y = np.pad(
            y,
            (0, sr - len(y))
        )

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )

    features.extend(
        np.std(chroma, axis=1)
    )

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(
        float(np.mean(centroid))
    )

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(
        float(np.mean(bandwidth))
    )

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(contrast, axis=1)
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(
        float(np.mean(zcr))
    )

    # RMS energy
    rms = librosa.feature.rms(y=y)

    features.append(
        float(np.mean(rms))
    )

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        features.append(
            float(np.asarray(tempo).reshape(-1)[0])
        )

    except Exception:
        features.append(0.0)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# FIND AUDIO FOR MIDI
# ============================================================

def find_audio_for_midi(midi_path):

    folder = os.path.dirname(midi_path)

    candidates = []

    for root, _, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".wav", ".flac", ".mp3")
            ):

                candidates.append(
                    os.path.join(root, file)
                )

    # Prefer mixture.wav
    for file in candidates:

        if os.path.basename(file).lower() == "mixture.wav":
            return file

    if candidates:
        return candidates[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING INSTRUMENT CLASSIFIER")
    print("=" * 60)

    if not os.path.exists(LABEL_CSV):
        print("instrument_dataset.csv not found!")
        return

    df = pd.read_csv(LABEL_CSV)

    # Group labels per MIDI file
    grouped = (
        df.groupby("midi_file")["instrument_class"]
        .apply(list)
        .reset_index()
    )

    print("Training tracks:", len(grouped))
    print()

    X = []
    y_labels = []
    valid_files = 0

    for index, row in grouped.iterrows():

        midi_path = row["midi_file"]
        labels = list(set(row["instrument_class"]))

        audio_path = find_audio_for_midi(
            midi_path
        )

        if not audio_path:
            print(
                "Audio not found:",
                midi_path
            )
            continue

        try:

            print(
                f"Processing {index + 1}/{len(grouped)}: "
                f"{os.path.basename(audio_path)}"
            )

            features = extract_features(
                audio_path
            )

            X.append(features)
            y_labels.append(labels)

            valid_files += 1

        except Exception as error:

            print(
                "Feature extraction failed:",
                error
            )

    if not X:

        print("No training data found.")
        return

    X = np.asarray(X)

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    mlb = MultiLabelBinarizer()

    y = mlb.fit_transform(
        y_labels
    )

    print()
    print("Valid training tracks:", valid_files)
    print("Feature count:", X.shape[1])
    print("Classes:", list(mlb.classes_))

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print()
    print("Training model...")

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=mlb.classes_,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        mlb,
        LABEL_PATH
    )

    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print("Model:")
    print(MODEL_PATH)

    print("Labels:")
    print(LABEL_PATH)


if __name__ == "__main__":
    main()