CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id BIGINT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'quiz')),
    temperature NUMERIC(2,1) NOT NULL DEFAULT 0.3 CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)),
    generation BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chat_history_chat_id_id_idx ON chat_history (chat_id, id DESC);
