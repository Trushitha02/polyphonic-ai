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
