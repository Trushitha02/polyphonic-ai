import os
import shutil
import subprocess
import sys
import time

STALE_LOCK_SECONDS = 60 * 60


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
        "guitar": "",
        "piano": "",
        "bgm": ""
    }

    if not os.path.isdir(output_folder):
        return result

    for root, dirs, files in os.walk(output_folder):
        # Never treat our own isolated-instrument cache as Demucs stems.
        dirs[:] = [d for d in dirs if d.lower() != "instruments"]
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
            elif lower == "guitar.wav":
                result["guitar"] = full_path
            elif lower == "piano.wav":
                result["piano"] = full_path
            elif lower in ("bgm.wav", "no_vocals.wav"):
                result["bgm"] = full_path

    # Merge accompaniment stems into bgm.wav if bgm is missing
    if not result.get("bgm"):
        bgm_stems = [
            p for p in [
                result.get("drums"), result.get("bass"), result.get("other"),
                result.get("guitar"), result.get("piano"),
            ] if p and os.path.isfile(p)
        ]
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
            "separated"
        )
    )

    output_folder = os.path.join(
        base_folder,
        str(audio_id)
    )
    lock_path = os.path.join(output_folder, ".separation.lock")

    os.makedirs(output_folder, exist_ok=True)

    # A lock left behind by a crashed/killed run must not block forever.
    if os.path.isfile(lock_path):
        try:
            if time.time() - os.path.getmtime(lock_path) > STALE_LOCK_SECONDS:
                os.remove(lock_path)
                print(f"Removed stale separation lock for audio {audio_id}")
        except OSError:
            pass

    # Stems in this folder from a different song (reused audio_id) are stale.
    if not _folder_matches_song(output_folder, audio_path):
        force = True

    # Check if stems already exist and are valid
    if not force:
        existing = scan_existing_stems(output_folder)
        if existing.get("vocals") and (existing.get("bgm") or existing.get("other")):
            print(f"Using cached separation stems for audio {audio_id}")
            return existing
    elif not os.path.isfile(lock_path):
        # Re-run requested: clear old stems so fresh output is picked up.
        for entry in os.listdir(output_folder):
            target = os.path.join(output_folder, entry)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.remove(target)
                except OSError:
                    pass

    try:
        lock_handle = open(lock_path, "x", encoding="ascii")
        lock_handle.write(str(os.getpid()))
        lock_handle.close()
    except FileExistsError:
        raise SeparationInProgressError(
            f"AI separation is already running for audio {audio_id}."
        )

    _write_source_marker(output_folder, audio_path)

    try:
        python_exe = get_separation_python()
        if demucs_available(python_exe):
            command = [
                python_exe,
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

            try:
                subprocess.run(command, check=True)
                result = scan_existing_stems(output_folder)
                if _has_core_stems(result):
                    print("Separation completed!")
                    return result
                print("Demucs finished but produced no stems; using DSP fallback.")
            except Exception as error:
                print("Demucs failed, using DSP fallback separation:", error)
        else:
            print("Demucs/torch not installed for", python_exe, "- using DSP fallback separation.")

        result = dsp_separate(audio_path, os.path.join(output_folder, "dsp"))
        result = scan_existing_stems(output_folder)
        print("DSP fallback separation completed!")
        return result
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


_demucs_cache = {}


def demucs_available(python_exe=None):
    """True when the interpreter used for separation can import demucs + torch."""
    python_exe = python_exe or get_separation_python()
    if python_exe in _demucs_cache:
        return _demucs_cache[python_exe]
    try:
        check = subprocess.run(
            [python_exe, "-c", "import demucs.separate, torch"],
            capture_output=True,
            timeout=120,
        )
        ok = check.returncode == 0
    except Exception:
        ok = False
    _demucs_cache[python_exe] = ok
    return ok


def dsp_separate(audio_path, output_folder):
    """
    Lightweight signal-processing separation used when Demucs is not
    installed (e.g. on a small cloud server). Produces the same stem
    files as Demucs (vocals / drums / bass / other) so every page works:
      - drums  : percussive part (harmonic-percussive separation)
      - bass   : harmonic energy below ~180 Hz
      - vocals : centre-panned harmonic energy in the voice band
      - other  : remaining harmonic accompaniment
    The four soft masks sum to 1, so the stems add back up to the song.
    Quality is lower than Demucs, but it is fast and needs no torch.
    """
    import numpy as np
    import soundfile as sf
    import librosa

    os.makedirs(output_folder, exist_ok=True)
    sr = 22050
    n_fft = 2048
    hop = 512

    y, _ = librosa.load(audio_path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y[:2]
    left, right = y[0], y[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    L = librosa.stft(left, n_fft=n_fft, hop_length=hop)
    R = librosa.stft(right, n_fft=n_fft, hop_length=hop)
    M = np.abs(librosa.stft(mid, n_fft=n_fft, hop_length=hop))
    S = np.abs(librosa.stft(side, n_fft=n_fft, hop_length=hop))

    harmonic_mask, percussive_mask = librosa.decompose.hpss(M, mask=True)
    harmonic_mask = harmonic_mask.astype(np.float32)
    percussive_mask = percussive_mask.astype(np.float32)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[:, None]
    bass_band = (freqs < 180).astype(np.float32)
    voice_band = ((freqs >= 180) & (freqs <= 5000)).astype(np.float32)

    centre = np.clip(1.0 - S / (M + 1e-8), 0.0, 1.0) ** 2
    if float(np.mean(S)) < 1e-4 * max(float(np.mean(M)), 1e-8):
        # Mono recording: no stereo cue, use a gentler vocal estimate
        centre = np.full_like(M, 0.6, dtype=np.float32)

    masks = {
        "drums": percussive_mask,
        "bass": harmonic_mask * bass_band,
        "vocals": harmonic_mask * voice_band * centre,
    }
    masks["other"] = np.clip(harmonic_mask - masks["bass"] - masks["vocals"], 0.0, 1.0)

    length = y.shape[1]
    for name, mask in masks.items():
        stereo = np.stack([
            librosa.istft(L * mask, hop_length=hop, length=length),
            librosa.istft(R * mask, hop_length=hop, length=length),
        ], axis=1)
        sf.write(os.path.join(output_folder, f"{name}.wav"), stereo.astype(np.float32), sr)

    return scan_existing_stems(output_folder)


# ============================================================
# SHARED STEM LOOKUP
# ============================================================
# Every route (separation, instrument playback, transcription,
# performance) resolves stems through this one function, so an
# instrument can always be played no matter where its stems live:
#   1. separated/<audio_id>/...            (normal Demucs / fallback run)
#   2. the folder saved in separation_results  (database record)
#   3. separated/<model>/<song name>/      (Demucs run manually from CLI)

SEPARATED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "separated")
)


def _has_core_stems(stems):
    return bool(
        stems
        and stems.get("vocals")
        and os.path.isfile(stems["vocals"])
        and (stems.get("bgm") or stems.get("other"))
    )


def _name_key(value):
    return "".join(ch.lower() for ch in (value or "") if ch.isalnum())


def _stem_richness(stems):
    """6-stem (guitar/piano) > 4-stem (drums/bass/other) > 2-stem (no_vocals)."""
    return sum(1 for key in ("drums", "bass", "other", "guitar", "piano") if stems.get(key))


def find_stems_by_song_name(file_path, separated_root=None):
    """Find Demucs output that was generated manually for this song.

    If the song was separated with several models (e.g. htdemucs two-stem
    AND htdemucs_6s), use the folder with the most instrument stems, so
    drums/bass/guitar/piano get their real stems instead of a filtered BGM.
    """
    separated_root = separated_root or SEPARATED_ROOT
    expected_key = _name_key(os.path.splitext(os.path.basename(file_path or ""))[0])
    if not expected_key or not os.path.isdir(separated_root):
        return None

    best = None
    for root, dirs, files in os.walk(separated_root):
        dirs[:] = [d for d in dirs if d.lower() != "instruments"]
        if "vocals.wav" not in {name.lower() for name in files}:
            continue
        folder_key = _name_key(os.path.basename(root))
        if not folder_key or (expected_key not in folder_key and folder_key not in expected_key):
            continue
        stems = scan_existing_stems(root)
        if _has_core_stems(stems) and (best is None or _stem_richness(stems) > _stem_richness(best)):
            best = stems
    return best


def _audio_file_path(audio_id):
    try:
        from database.database import get_connection
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT file_path FROM audio_files WHERE audio_id = %s", (audio_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row.get("file_path") if row else None
    except Exception as error:
        print("Could not read audio file path:", error)
        return None


def find_stems_for_audio(audio_id, file_path=None, separated_root=None):
    """Return the stems dict for an audio_id, or None if not separated yet."""
    separated_root = separated_root or SEPARATED_ROOT

    # 1. Standard per-audio folder (ignored if it belongs to another song,
    #    e.g. after the database was reset and audio_ids started again)
    audio_folder = os.path.join(separated_root, str(audio_id))
    if file_path is None:
        file_path = _audio_file_path(audio_id)
    if _folder_matches_song(audio_folder, file_path):
        stems = scan_existing_stems(audio_folder)
        if _has_core_stems(stems):
            return stems

    # 2. Database record
    try:
        from database.database import get_connection
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT vocals_path, drums_path, bass_path, other_path
            FROM separation_results
            WHERE audio_id = %s
            ORDER BY separation_id DESC
            LIMIT 1
            """,
            (audio_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row.get("vocals_path") and os.path.isfile(row["vocals_path"]):
            stems = scan_existing_stems(os.path.dirname(row["vocals_path"]))
            for key in ("vocals", "drums", "bass", "other"):
                if not stems.get(key) and row.get(f"{key}_path") and os.path.isfile(row[f"{key}_path"]):
                    stems[key] = row[f"{key}_path"]
            if _has_core_stems(stems):
                return stems
    except Exception as error:
        print("Stem lookup (database) warning:", error)

    # 3. Manually generated Demucs folder matched by song name
    return find_stems_by_song_name(file_path, separated_root)


SOURCE_MARKER = "source.txt"


def _write_source_marker(output_folder, audio_path):
    try:
        with open(os.path.join(output_folder, SOURCE_MARKER), "w", encoding="utf-8") as marker:
            marker.write(os.path.abspath(audio_path or ""))
    except OSError:
        pass


def _folder_matches_song(folder, file_path):
    marker = os.path.join(folder, SOURCE_MARKER)
    if not os.path.isfile(marker) or not file_path:
        return True  # folders from older versions have no marker
    try:
        with open(marker, encoding="utf-8") as handle:
            recorded = handle.read().strip()
    except OSError:
        return True
    return os.path.normcase(recorded) == os.path.normcase(os.path.abspath(file_path))
