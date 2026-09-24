# What was fixed (Sep 2026)

1. **Detected instruments now play.** "Play <instrument>" returned HTTP 400 whenever the stems were
   not in `separated/<audio_id>/` (e.g. Demucs output reused from `separated/htdemucs/<song>/`).
   All routes now use one shared stem lookup (`find_stems_for_audio`): per-song folder → database record → song-name match.
2. **Separation page broke after a real login.** `loadUser()` wrote to a missing `#userName` element,
   crashing page start-up (same bug on Transcription). Guarded.
3. **Guitar / Piano use the real 6-stem Demucs stems** when available; Drums/Percussion/Tabla → drums, Bass → bass, Other → other;
   other instruments are isolated from `other.wav` (harmonic filter + instrument band) and prepared in the background after separation.
4. **Stale audio fixed:** isolated-instrument cache files are fingerprinted to their source stem, and stem folders record their song,
   so a reset database can't play another song's audio.
5. **Works without Demucs/torch** (Render): built-in DSP separation into vocals / drums / bass / other.
6. **Works without MySQL:** automatic SQLite fallback (`backend/polyphonic.db`). DB password moved from code to `backend/.env`.
7. **Transcription:** drums → onset hits (kick/snare), pitched instruments → notes with instrument-specific pitch range,
   MIDI export (`/api/transcription/midi/...`), results saved in the `transcriptions` table, "Play transcribed notes" in the browser,
   max 2 instruments processed at a time.
8. **Performance analysis:** the practice take is aligned to the matching section of the reference; detected instruments can be chosen as reference;
   `separated_audio` path bug fixed; practice uploads no longer create song records.
9. **Confidence & Progress** now load from the database (same on every device); Chart.js guard.
10. Uploads: same-name different songs no longer overwrite each other; user_id saved from the Separation page upload.
11. Security: file streaming limited to audio files (the old route could download any project file, including the database).
12. `render.yaml`: gunicorn timeout 900 s (default 30 s killed separation/transcription requests).
13. **Different instruments no longer play the same BGM.** On Windows the 2-stem Demucs folder (vocals/no_vocals)
    was chosen before `htdemucs_6s`, so every instrument was a filtered copy of the whole BGM. Now the folder with the
    most stems is always used; without drum/bass stems, drums and bass are *extracted* (percussive part / low band);
    the remaining instruments each get their own non-overlapping frequency range of the "other" stem
    (similarity between instruments dropped from up to 0.93 to at most 0.43). Each card shows where its audio comes from.
14. **No more background vocals in instrument audio, and more difference between instruments.**
    All detected instruments of a song are now built together with competitive masking: every
    time-frequency point goes to the source that dominates it (vocals, guitar, piano, drums, bass, other),
    plus a noise gate for faint leakage. Instrument files are no longer boosted to full volume
    (that boost turned near-silent stems full of leaked singing into loud "same BGM" tracks).
    Measured on Mirchi: vocal leakage in Guitar 0.33 -> 0.05, Piano 0.34 -> 0.04, Other 0.32 -> 0.00.
    Each card now shows how present the instrument really is ("Clearly present" / "Quiet" /
    "Barely in this song — the AI detection may be wrong").
15. **"Login failed" on Render fixed.** If the configured MySQL/PostgreSQL (DB_HOST / DATABASE_URL) is not
    reachable, the app now falls back to SQLite instead of failing every login (set DB_STRICT=1 to disable).
    `/api/database/status` shows which database is used and why; login/register show the real error.
