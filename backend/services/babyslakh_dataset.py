import os

# Find the backend folder
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset folder
DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets"
)


def dataset_exists():
    return os.path.exists(DATASET_PATH)


def get_dataset_files():
    files = []

    if not dataset_exists():
        return files

    for root, folders, filenames in os.walk(DATASET_PATH):

        for filename in filenames:
            files.append(
                os.path.join(root, filename)
            )

    return files


def get_audio_files():
    audio_files = []

    for file in get_dataset_files():

        if file.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg")
        ):
            audio_files.append(file)

    return audio_files


def get_midi_files():
    midi_files = []

    for file in get_dataset_files():

        if file.lower().endswith(
            (".mid", ".midi")
        ):
            midi_files.append(file)

    return midi_files


if __name__ == "__main__":

    print("==============================")
    print("BabySlakh Dataset")
    print("==============================")

    print("Dataset path:")
    print(DATASET_PATH)

    if dataset_exists():

        print("Dataset found!")

        audio = get_audio_files()
        midi = get_midi_files()

        print("Audio files:", len(audio))
        print("MIDI files:", len(midi))

    else:

        print("Dataset not found!")