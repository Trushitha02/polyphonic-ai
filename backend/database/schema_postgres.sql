-- Polyphonic Database Schema (PostgreSQL for Render)
-- All 8 tables matching production schema

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_files (
    audio_id SERIAL PRIMARY KEY,
    user_id INT DEFAULT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_audio_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_audio_user ON audio_files(user_id);

CREATE TABLE IF NOT EXISTS performances (
    performance_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    audio_id INT DEFAULT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    pitch_score NUMERIC(5,2) DEFAULT 0.00,
    rhythm_score NUMERIC(5,2) DEFAULT 0.00,
    timing_score NUMERIC(5,2) DEFAULT 0.00,
    note_score NUMERIC(5,2) DEFAULT 0.00,
    overall_score NUMERIC(5,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_perf_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_perf_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_perf_user ON performances(user_id);
CREATE INDEX IF NOT EXISTS idx_perf_audio ON performances(audio_id);

CREATE TABLE IF NOT EXISTS progress (
    progress_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    overall_score NUMERIC(5,2) DEFAULT 0.00,
    confidence_score NUMERIC(5,2) DEFAULT 0.00,
    songs_completed INT DEFAULT 0,
    performances_count INT DEFAULT 0,
    current_level VARCHAR(50) DEFAULT 'Musical Starter',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prog_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_prog_user ON progress(user_id);

CREATE TABLE IF NOT EXISTS instrument_detections (
    detection_id SERIAL PRIMARY KEY,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) NOT NULL,
    confidence NUMERIC(5,2) DEFAULT NULL,
    source_stem VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_det_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_det_audio ON instrument_detections(audio_id);

CREATE TABLE IF NOT EXISTS separation_results (
    separation_id SERIAL PRIMARY KEY,
    audio_id INT NOT NULL,
    vocals_path VARCHAR(500) DEFAULT NULL,
    drums_path VARCHAR(500) DEFAULT NULL,
    bass_path VARCHAR(500) DEFAULT NULL,
    other_path VARCHAR(500) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sep_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sep_audio ON separation_results(audio_id);

CREATE TABLE IF NOT EXISTS practice_recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    recommendation TEXT,
    difficulty_level VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rec_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rec_user ON practice_recommendations(user_id);

CREATE TABLE IF NOT EXISTS transcriptions (
    transcription_id SERIAL PRIMARY KEY,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    midi_path VARCHAR(500) DEFAULT NULL,
    notes_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_trans_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trans_audio ON transcriptions(audio_id);
