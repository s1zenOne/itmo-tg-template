CREATE TABLE pending_cities (
    action_id UUID PRIMARY KEY,
    owner_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
    destination_chat_id BIGINT NOT NULL,
    update_id BIGINT NOT NULL,
    city_name TEXT NOT NULL,
    city_country TEXT NOT NULL,
    city_region TEXT,
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    expires_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'cancelled')),
    UNIQUE (owner_id, update_id)
);
