-- Polyphonic Database Schema
-- Run this against the database named by DB_NAME.

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS audio_files (
    audio_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT DEFAULT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    uploaded_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audio_user (user_id),
    CONSTRAINT fk_audio_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS performances (
    performance_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    audio_id INT DEFAULT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    pitch_score DECIMAL(5,2) DEFAULT 0.00,
    rhythm_score DECIMAL(5,2) DEFAULT 0.00,
    timing_score DECIMAL(5,2) DEFAULT 0.00,
    note_score DECIMAL(5,2) DEFAULT 0.00,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_perf_user (user_id),
    INDEX idx_perf_audio (audio_id),
    CONSTRAINT fk_perf_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_perf_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS progress (
    progress_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    confidence_score DECIMAL(5,2) DEFAULT 0.00,
    songs_completed INT DEFAULT 0,
    performances_count INT DEFAULT 0,
    current_level VARCHAR(50) DEFAULT 'Musical Starter',
    updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_prog_user (user_id),
    CONSTRAINT fk_prog_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS instrument_detections (
    detection_id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,2) DEFAULT NULL,
    source_stem VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_det_audio (audio_id),
    CONSTRAINT fk_det_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS separation_results (
    separation_id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT NOT NULL,
    vocals_path VARCHAR(500) DEFAULT NULL,
    drums_path VARCHAR(500) DEFAULT NULL,
    bass_path VARCHAR(500) DEFAULT NULL,
    other_path VARCHAR(500) DEFAULT NULL,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sep_audio (audio_id),
    CONSTRAINT fk_sep_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS practice_recommendations (
    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    recommendation TEXT,
    difficulty_level VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rec_user (user_id),
    CONSTRAINT fk_rec_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS transcriptions (
    transcription_id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT NOT NULL,
    instrument_name VARCHAR(100) DEFAULT NULL,
    midi_path VARCHAR(500) DEFAULT NULL,
    notes_data LONGTEXT,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trans_audio (audio_id),
    CONSTRAINT fk_trans_audio FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
