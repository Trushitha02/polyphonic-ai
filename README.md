# polyphonic-ai
AI-powered polyphonic instrument separation and music transcription system that separates instruments from audio and converts musical content into transcribed notes/MIDI.

## Run it on your computer (Windows)

```bat
cd Polyphonic
venv\Scripts\activate
cd backend
pip install -r requirements-local.txt   REM first time only (Flask, librosa, Demucs ...)
python app.py
```

Open **http://localhost:5000** in the browser, register, and log in.

* **Database:** if MySQL is running with the details in `backend/.env`, it is used.
  If MySQL is not running, the app automatically uses a local SQLite file
  (`backend/polyphonic.db`), so nothing breaks. Put `DB_TYPE=sqlite` in `.env` to always use SQLite.
* **Separation:** uses Demucs when it is installed in `venv`. If Demucs/torch is
  missing (for example on Render), a built-in signal-processing separator is used
  instead, so every page still works (quality is lower than Demucs).
* Demucs folders you created by hand (`backend/separated/htdemucs/<song name>/`,
  `htdemucs_6s/...`) are picked up automatically for the matching uploaded song.

## Workflow (every tab)

| Tab | What it does |
| --- | --- |
| Upload Music | Uploads a song and runs AI instrument detection on it |
| Instrument Separation | Splits Vocals / BGM (drums, bass, other, + guitar & piano with the 6-stem model), detects the instruments in the BGM and **plays each detected instrument** (▶ Play / ↓ Download) |
| Music Transcription | Converts each selected instrument into notes (drums → hits), plays the isolated instrument, **plays the transcribed notes**, and exports a **MIDI file** |
| Performance Analysis | Pick Vocals, BGM or any detected instrument as reference, upload your practice recording; it finds the matching section of the song and scores pitch / rhythm / timing / notes |
| Skill Level · Confidence · Progress · Dashboard · Profile | Built from the performances saved in the database |

## How detected instruments are played

| Detected instrument | Audio that is played |
| --- | --- |
| Drums, Percussion, Tabla, Mridangam, Dholak ... | `drums.wav` stem |
| Bass | `bass.wav` stem |
| Guitar / Piano | `guitar.wav` / `piano.wav` (6-stem Demucs) — otherwise filtered from `other.wav` |
| Other | `other.wav` stem |
| Flute, Strings, Brass, Synth, Organ, Violin, Sitar, Veena ... | isolated from `other.wav` with a harmonic filter + the instrument's frequency band, cached in `separated/<id>/instruments/` |

Isolated instrument audio is prepared in the background right after separation, so
the Play buttons start immediately.

## Render deployment

`render.yaml` installs `backend/requirements.txt` (no Demucs/torch, too big for the
free plan) and starts gunicorn with a long timeout, because separation and
transcription requests take longer than gunicorn's default 30 seconds.
Set `DATABASE_URL` (PostgreSQL) or the `DB_*` variables in the Render dashboard.
