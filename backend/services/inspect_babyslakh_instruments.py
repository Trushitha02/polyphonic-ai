import os
from collections import Counter

import pretty_midi


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "babyslakh_16k"
)


# ============================================================
# FIND MIDI FILES
# ============================================================

def find_midi_files():
    midi_files = []

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".mid", ".midi")):
                midi_files.append(
                    os.path.join(root, file)
                )

    return midi_files


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BABYSLakh INSTRUMENT INSPECTION")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print("Dataset not found!")
        print("Path:", DATASET_DIR)
        return

    midi_files = find_midi_files()

    print("Dataset path:")
    print(DATASET_DIR)

    print()
    print("Total MIDI files found:", len(midi_files))

    if not midi_files:
        print("No MIDI files found.")
        return

    instrument_counter = Counter()

    print()
    print("Reading MIDI files...")
    print()

    for index, midi_path in enumerate(midi_files, start=1):

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)

            for instrument in midi.instruments:

                # Ignore drums initially
                if instrument.is_drum:
                    name = "Drums"
                else:
                    name = pretty_midi.program_to_instrument_name(
                        instrument.program
                    )

                instrument_counter[name] += 1

        except Exception as error:
            print(
                f"Could not read: {midi_path}"
            )
            print("Error:", error)

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(midi_files)} MIDI files..."
            )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("INSTRUMENTS FOUND")
    print("=" * 60)

    for instrument, count in instrument_counter.most_common():

        print(
            f"{instrument:35s} {count}"
        )

    print()
    print("=" * 60)
    print("TOTAL UNIQUE INSTRUMENTS:", len(instrument_counter))
    print("=" * 60)


if __name__ == "__main__":
    main()