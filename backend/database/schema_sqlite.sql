-- Polyphonic Database Schema (SQLite - local zero-config fallback)
-- All 8 tables matching production and local schema

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_files (
    audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INT DEFAULT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS performances (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INT NOT NULL,
    audio_id INT DEFAULT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    pitch_score DECIMAL(5,2) DEFAULT 0.00,
    rhythm_score DECIMAL(5,2) DEFAULT 0.00,
    timing_score DECIMAL(5,2) DEFAULT 0.00,
    note_score DECIMAL(5,2) DEFAULT 0.00,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INT NOT NULL UNIQUE,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    confidence_score DECIMAL(5,2) DEFAULT 0.00,
    songs_completed INT DEFAULT 0,
    performances_count INT DEFAULT 0,
    current_level VARCHAR(50) DEFAULT 'Musical Starter',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS instrument_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,2) DEFAULT NULL,
    source_stem VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS separation_results (
    separation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id INT NOT NULL,
    vocals_path VARCHAR(500) DEFAULT NULL,
    drums_path VARCHAR(500) DEFAULT NULL,
    bass_path VARCHAR(500) DEFAULT NULL,
    other_path VARCHAR(500) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS practice_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    recommendation TEXT,
    difficulty_level VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transcriptions (
    transcription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    midi_path VARCHAR(500) DEFAULT NULL,
    notes_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
);
