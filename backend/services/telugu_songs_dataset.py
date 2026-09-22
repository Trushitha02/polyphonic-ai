import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "telugu_songs.csv"
)


def dataset_exists():
    return os.path.exists(CSV_PATH)


def load_songs():
    if not dataset_exists():
        return None

    return pd.read_csv(CSV_PATH)


def get_song_count():
    songs = load_songs()

    if songs is None:
        return 0

    return len(songs)


if __name__ == "__main__":

    print("==============================")
    print("Telugu Songs Dataset")
    print("==============================")

    print("Dataset path:")
    print(CSV_PATH)

    if dataset_exists():

        songs = load_songs()

        print("Dataset found!")
        print("Total songs:", len(songs))

        print("\nColumns:")
        print(list(songs.columns))

        print("\nFirst 5 songs:")
        print(songs.head())

    else:

        print("Dataset not found!")