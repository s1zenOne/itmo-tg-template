ALTER TABLE chat_settings ADD COLUMN timezone TEXT;

CREATE TABLE reminder_actions (
    action_id UUID PRIMARY KEY,
    owner_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    update_id BIGINT NOT NULL,
    text TEXT NOT NULL CHECK (char_length(text) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'confirmed', 'cancelled', 'executed')),
    UNIQUE (owner_id, update_id)
);

CREATE TABLE reminders (
    id BIGSERIAL PRIMARY KEY,
    action_id UUID NOT NULL UNIQUE REFERENCES reminder_actions(action_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE agent_audit (
    id BIGSERIAL PRIMARY KEY,
    owner_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    requested_at TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL,
    tool TEXT,
    arguments TEXT NOT NULL,
    validation TEXT NOT NULL,
    execution TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    reason TEXT NOT NULL
);

CREATE INDEX agent_audit_owner_requested_idx ON agent_audit (owner_id, requested_at DESC, id DESC);
