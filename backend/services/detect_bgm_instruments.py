import os
import sys
import librosa
import torch
from transformers import pipeline


# ============================================================
# FIND PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SEPARATED_DIR = os.path.join(
    BASE_DIR,
    "separated"
)


# ============================================================
# FIND BGM FILES
# ============================================================

def find_bgm_files():
    bgm_files = []

    if not os.path.exists(SEPARATED_DIR):
        return bgm_files

    for root, folders, files in os.walk(SEPARATED_DIR):

        for filename in files:

            if filename.lower() == "no_vocals.wav":

                full_path = os.path.abspath(
                    os.path.join(root, filename)
                )

                bgm_files.append(full_path)

    return bgm_files


# ============================================================
# LOAD AI MODEL
# ============================================================

def load_model():

    print()
    print("==============================================")
    print("Loading AI Instrument Recognition Model")
    print("==============================================")

    classifier = pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        device=-1
    )

    print("AI model loaded successfully!")

    return classifier


# ============================================================
# ANALYZE BGM
# ============================================================

def analyze_bgm(classifier, audio_path):

    print()
    print("==============================================")
    print("BGM INSTRUMENT ANALYSIS")
    print("==============================================")

    print("BGM file:")
    print(audio_path)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.isfile(audio_path):

        print()
        print("ERROR: Audio file does not exist.")
        return

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    print()
    print("Loading audio...")

    try:

        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

    except Exception as error:

        print()
        print("ERROR: Could not open audio file.")
        print(error)
        return

    duration = len(audio) / sample_rate

    print("Audio loaded successfully!")
    print("Sample rate:", sample_rate)
    print("Duration:", round(duration, 2), "seconds")

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    print()
    print("Analyzing BGM...")
    print("Please wait...")

    try:

        results = classifier(
            audio,
            top_k=20
        )

    except Exception as error:

        print()
        print("ERROR: Instrument analysis failed.")
        print(error)
        return

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("MODEL PREDICTIONS")
    print("==============================================")

    for result in results:

        label = result.get("label", "Unknown")

        score = float(
            result.get("score", 0)
        ) * 100

        print(
            f"{label:35s} {score:6.2f}%"
        )

    print("==============================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("POLYPHONIC BGM INSTRUMENT DETECTION")
    print("==============================================")

    # --------------------------------------------------------
    # FIND AVAILABLE BGM FILES
    # --------------------------------------------------------

    bgm_files = find_bgm_files()

    if not bgm_files:

        print()
        print("No no_vocals.wav files were found.")

        print()
        print("Expected location:")
        print(SEPARATED_DIR)

        print()
        print("First run Demucs on a song.")

        return

    # --------------------------------------------------------
    # DISPLAY FILES
    # --------------------------------------------------------

    print()
    print("Available separated BGM files:")
    print("----------------------------------------------")

    for index, path in enumerate(bgm_files, start=1):

        print(
            f"{index}. {path}"
        )

    print("----------------------------------------------")

    # --------------------------------------------------------
    # SELECT SONG
    # --------------------------------------------------------

    if len(bgm_files) == 1:

        selected_path = bgm_files[0]

        print()
        print("Only one BGM file found.")
        print("Automatically selecting it.")

    else:

        print()

        while True:

            choice = input(
                f"Select song number (1-{len(bgm_files)}): "
            )

            try:

                choice = int(choice)

                if 1 <= choice <= len(bgm_files):

                    selected_path = bgm_files[
                        choice - 1
                    ]

                    break

            except ValueError:

                pass

            print("Please enter a valid number.")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    classifier = load_model()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analyze_bgm(
        classifier,
        selected_path
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()