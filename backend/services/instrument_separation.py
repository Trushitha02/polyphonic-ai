import os
import subprocess
import sys


class SeparationInProgressError(RuntimeError):
    """Raised when another Demucs process already owns an audio job."""


def get_separation_python():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python
    backend_venv = os.path.join(os.path.dirname(__file__), "..", "venv", "Scripts", "python.exe")
    if os.path.isfile(backend_venv):
        return os.path.abspath(backend_venv)
    return sys.executable


def scan_existing_stems(output_folder):
    result = {
        "vocals": "",
        "drums": "",
        "bass": "",
        "other": "",
        "bgm": ""
    }

    if not os.path.isdir(output_folder):
        return result

    for root, dirs, files in os.walk(output_folder):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            lower = file.lower()

            if lower == "vocals.wav":
                result["vocals"] = full_path
            elif lower == "drums.wav":
                result["drums"] = full_path
            elif lower == "bass.wav":
                result["bass"] = full_path
            elif lower == "other.wav":
                result["other"] = full_path
            elif lower in ("bgm.wav", "no_vocals.wav"):
                result["bgm"] = full_path

    # Merge accompaniment stems into bgm.wav if bgm is missing
    if not result.get("bgm"):
        bgm_stems = [p for p in [result.get("drums"), result.get("bass"), result.get("other")] if p and os.path.isfile(p)]
        if bgm_stems:
            try:
                import numpy as np
                import soundfile as sf
                data_list = []
                sample_rate = None
                for stem in bgm_stems:
                    data, sr = sf.read(stem)
                    data_list.append(data)
                    sample_rate = sr

                max_len = max(len(d) for d in data_list)
                channels = data_list[0].shape[1] if data_list[0].ndim > 1 else 1
                bgm_data = np.zeros((max_len, channels), dtype=np.float32)
                for d in data_list:
                    if d.ndim == 1:
                        d = d[:, None]
                    bgm_data[:len(d)] += d

                max_val = np.max(np.abs(bgm_data))
                if max_val > 1.0:
                    bgm_data /= max_val

                stem_dir = os.path.dirname(bgm_stems[0])
                bgm_out = os.path.join(stem_dir, "bgm.wav")
                sf.write(bgm_out, bgm_data, sample_rate)
                result["bgm"] = os.path.abspath(bgm_out)
                print("Created composite BGM matching vocals length:", bgm_out)
            except Exception as e:
                print("Could not generate composite BGM audio:", e)

    return result


def separate_audio(audio_path, audio_id, force=False):

    base_folder = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "separated_audio"
        )
    )

    output_folder = os.path.join(
        base_folder,
        str(audio_id)
    )
    lock_path = os.path.join(output_folder, ".separation.lock")

    os.makedirs(output_folder, exist_ok=True)

    # Check if stems already exist and are valid
    if not force:
        existing = scan_existing_stems(output_folder)
        if existing.get("vocals") and (existing.get("bgm") or existing.get("other")):
            print(f"Using cached separation stems for audio {audio_id}")
            return existing

    try:
        lock_handle = open(lock_path, "x", encoding="ascii")
        lock_handle.write(str(os.getpid()))
        lock_handle.close()
    except FileExistsError:
        raise SeparationInProgressError(
            f"AI separation is already running for audio {audio_id}."
        )

    try:
        command = [
            get_separation_python(),
            "-m",
            "demucs.separate",
            "-d",
            "cpu",
            "-j",
            "4",
            "-o",
            output_folder,
            audio_path
        ]

        print("Running Demucs...")
        print("Audio:", audio_path)

        subprocess.run(command, check=True)

        result = scan_existing_stems(output_folder)

        print("Separation completed!")

        return result
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

