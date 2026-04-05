CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE herds (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    drive_folder_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    herd_id INTEGER NOT NULL REFERENCES herds(id) ON DELETE CASCADE,
    drive_folder_id TEXT NOT NULL UNIQUE,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (herd_id, name)
);

CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    horse_id INTEGER NOT NULL REFERENCES horses(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    drive_file_id TEXT NOT NULL UNIQUE,
    drive_md5 TEXT,
    processing_status TEXT DEFAULT 'pending',
    detection_result TEXT,
    excluded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    photo_id INTEGER NOT NULL UNIQUE REFERENCES photos(id) ON DELETE CASCADE,
    horse_id INTEGER NOT NULL REFERENCES horses(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    extracted_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON features USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE TABLE sync_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',
    files_scanned INTEGER DEFAULT 0,
    files_added INTEGER DEFAULT 0,
    files_removed INTEGER DEFAULT 0,
    files_moved INTEGER DEFAULT 0
);
