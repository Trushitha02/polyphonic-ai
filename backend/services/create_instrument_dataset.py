import os
import csv
from collections import Counter

import pretty_midi


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "instrument_dataset.csv"
)


# ============================================================
# INSTRUMENT GROUPING
# ============================================================

def group_instrument(name, is_drum=False):

    if is_drum:
        return "Drums"

    name = name.lower()

    # Piano / keyboard
    if any(x in name for x in [
        "piano",
        "keyboard",
        "harpsichord",
        "clavinet"
    ]):
        return "Piano"

    # Guitar
    if "guitar" in name:
        return "Guitar"

    # Bass
    if "bass" in name:
        return "Bass"

    # Strings
    if any(x in name for x in [
        "violin",
        "viola",
        "cello",
        "contrabass",
        "fiddle",
        "string ensemble",
        "synth strings"
    ]):
        return "Strings"

    # Flute / woodwind
    if any(x in name for x in [
        "flute",
        "pan flute",
        "oboe",
        "harmonica"
    ]):
        if "harmonica" in name:
            return "Harmonica"
        return "Flute"

    # Brass
    if any(x in name for x in [
        "trumpet",
        "trombone",
        "horn",
        "brass"
    ]):
        return "Brass"

    # Saxophone
    if "sax" in name:
        return "Saxophone"

    # Organ
    if "organ" in name:
        return "Organ"

    # Harp
    if "harp" in name:
        return "Harp"

    # Percussion
    if any(x in name for x in [
        "vibraphone",
        "marimba",
        "xylophone",
        "glockenspiel",
        "timpani",
        "tubular bells",
        "woodblock",
        "dulcimer"
    ]):
        return "Percussion"

    # Voice
    if any(x in name for x in [
        "voice",
        "choir",
        "aahs",
        "oohs"
    ]):
        return "Voice"

    # Synth / electronic
    if any(x in name for x in [
        "synth",
        "pad",
        "lead"
    ]):
        return "Synth"

    # Other
    return "Other"


# ============================================================
# FIND MIDI FILES
# ============================================================

def find_midi_files():

    midi_files = []

    for root, _, files in os.walk(DATASET_DIR):

        for file in files:

            if file.lower().endswith(
                (".mid", ".midi")
            ):
                midi_files.append(
                    os.path.join(root, file)
                )

    return midi_files


# ============================================================
# CREATE DATASET
# ============================================================

def main():

    print("=" * 65)
    print("CREATING CLEAN INSTRUMENT DATASET")
    print("=" * 65)

    if not os.path.exists(DATASET_DIR):

        print("BabySlakh dataset not found!")
        print(DATASET_DIR)
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    midi_files = find_midi_files()

    print("MIDI files found:", len(midi_files))
    print()

    rows = []
    class_counter = Counter()

    for index, midi_path in enumerate(midi_files, start=1):

        try:

            midi = pretty_midi.PrettyMIDI(
                midi_path
            )

            found_classes = set()

            for instrument in midi.instruments:

                original_name = (
                    "Drums"
                    if instrument.is_drum
                    else pretty_midi.program_to_instrument_name(
                        instrument.program
                    )
                )

                grouped_class = group_instrument(
                    original_name,
                    instrument.is_drum
                )

                found_classes.add(
                    grouped_class
                )

            # One row per MIDI file + grouped classes
            for grouped_class in sorted(found_classes):

                rows.append({
                    "midi_file": os.path.abspath(midi_path),
                    "song_name": os.path.basename(
                        os.path.dirname(midi_path)
                    ),
                    "instrument_class": grouped_class
                })

                class_counter[grouped_class] += 1

        except Exception as error:

            print(
                "Could not read:",
                midi_path
            )
            print(
                "Error:",
                error
            )

        if index % 25 == 0:

            print(
                f"Processed {index}/{len(midi_files)}"
            )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "midi_file",
                "song_name",
                "instrument_class"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 65)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 65)

    print("Output:")
    print(OUTPUT_CSV)

    print()
    print("Total dataset rows:", len(rows))

    print()
    print("GROUPED INSTRUMENT CLASSES")
    print("-" * 65)

    for instrument, count in class_counter.most_common():

        print(
            f"{instrument:20s} {count}"
        )

    print()
    print(
        "Total grouped classes:",
        len(class_counter)
    )

    print("=" * 65)


if __name__ == "__main__":
    main()