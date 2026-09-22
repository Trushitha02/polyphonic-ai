import os
import joblib
import numpy as np
import librosa


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "instrument_model.joblib"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "instrument_labels.joblib"
)


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

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(float(np.mean(centroid)))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(float(np.mean(bandwidth)))

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features.extend(np.mean(contrast, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(float(np.mean(zcr)))

    rms = librosa.feature.rms(y=y)

    features.append(float(np.mean(rms)))

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


def main():

    print("=" * 60)
    print("BGM INSTRUMENT DETECTOR")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):

        print("Model not found!")
        print(
            "Run train_instrument_model.py first."
        )

        return

    # Automatically find separated BGM
    separated_dir = os.path.join(
        BASE_DIR,
        "separated"
    )

    bgm_files = []

    for root, _, files in os.walk(
        separated_dir
    ):

        for file in files:

            if file.lower() == "no_vocals.wav":

                bgm_files.append(
                    os.path.join(root, file)
                )

    if not bgm_files:

        print("No separated BGM found.")

        return

    print()
    print("Available BGM files:")

    for i, path in enumerate(
        bgm_files,
        start=1
    ):

        print(
            f"{i}. {path}"
        )

    print()

    selected = 0

    if len(bgm_files) > 1:

        while True:

            try:

                selected = int(
                    input(
                        f"Choose song 1-{len(bgm_files)}: "
                    )
                ) - 1

                if 0 <= selected < len(bgm_files):
                    break

            except ValueError:
                pass

            print("Invalid choice.")

    audio_path = bgm_files[selected]

    print()
    print("Selected:")
    print(audio_path)

    print()
    print("Loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    mlb = joblib.load(
        LABEL_PATH
    )

    print("Analyzing BGM...")

    features = extract_features(
        audio_path
    )

    X = features.reshape(
        1,
        -1
    )

    prediction = model.predict(
        X
    )[0]

    print()
    print("=" * 60)
    print("DETECTED INSTRUMENTS")
    print("=" * 60)

    detected = []

    for label, value in zip(
        mlb.classes_,
        prediction
    ):

        if value == 1:

            detected.append(
                label
            )

    if detected:

        for instrument in detected:
            print("✓", instrument)

    else:

        print(
            "No instrument class detected with the current model."
        )

    print()
    print("Analysis completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()