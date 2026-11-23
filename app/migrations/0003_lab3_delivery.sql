-- Destination is recorded only from a private Telegram update, never from a model.
CREATE TABLE notification_settings (
    owner_id BIGINT PRIMARY KEY REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    destination_chat_id BIGINT NOT NULL,
    briefing_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    city_name TEXT,
    city_country TEXT,
    city_region TEXT,
    latitude DOUBLE PRECISION CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION CHECK (longitude BETWEEN -180 AND 180),
    CHECK ((latitude IS NULL) = (longitude IS NULL))
);

-- event_key is derived by application code from a persisted reminder or local date.
CREATE TABLE notification_outbox (
    event_key TEXT PRIMARY KEY,
    trace_id UUID,
    kind TEXT NOT NULL CHECK (kind IN ('reminder', 'briefing')),
    owner_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    reminder_id BIGINT REFERENCES reminders(id) ON DELETE CASCADE,
    local_date DATE,
    timezone TEXT,
    message_text TEXT,
    next_offset INTEGER NOT NULL DEFAULT 0 CHECK (next_offset >= 0),
    due_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'missed', 'failed', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
    next_attempt_at TIMESTAMPTZ NOT NULL,
    error_code TEXT,
    sent_at TIMESTAMPTZ,
    CHECK ((kind = 'reminder' AND reminder_id IS NOT NULL AND local_date IS NULL AND timezone IS NULL)
        OR (kind = 'briefing' AND reminder_id IS NULL AND local_date IS NOT NULL AND timezone IS NOT NULL)),
    CHECK (message_text IS NOT NULL OR next_offset = 0),
    CHECK (message_text IS NULL OR next_offset <= char_length(message_text)),
    UNIQUE (reminder_id),
    UNIQUE (owner_id, kind, local_date)
);
CREATE INDEX notification_outbox_pending_idx ON notification_outbox (next_attempt_at)
    WHERE status = 'pending';
