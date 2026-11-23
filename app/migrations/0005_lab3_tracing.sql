CREATE TABLE lab3_traces (
    trace_id UUID PRIMARY KEY,
    request_id TEXT NOT NULL CHECK (request_id ~ '^([0-9a-f]{16}|[0-9a-f]{32})$'),
    owner_id BIGINT NOT NULL CHECK (owner_id > 0),
    route TEXT NOT NULL CHECK (route IN ('schedule', 'knowledge', 'lifestyle', 'clarify', 'general', 'briefing', 'study', 'translate', 'quiz', 'unknown')),
    started_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX lab3_traces_owner_started_idx ON lab3_traces (owner_id, started_at DESC, trace_id DESC);

CREATE TABLE lab3_trace_spans (
    span_id UUID PRIMARY KEY,
    trace_id UUID NOT NULL REFERENCES lab3_traces(trace_id) ON DELETE CASCADE,
    parent_span_id UUID REFERENCES lab3_trace_spans(span_id) ON DELETE CASCADE,
    component TEXT NOT NULL CHECK (component IN ('supervisor', 'router', 'schedule', 'knowledge', 'lifestyle', 'briefing', 'telegram', 'mcp')),
    operation TEXT NOT NULL CHECK (operation IN ('request', 'route', 'execute', 'tool', 'resource', 'prepare', 'send', 'confirm')),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'error', 'clarify', 'missed')),
    tool_name TEXT CHECK (tool_name IN ('get_schedule', 'add_reminder', 'get_weather', 'search_materials')),
    error_code TEXT CHECK (error_code IN ('invalid_route', 'invalid_result', 'unauthorized', 'unavailable', 'timeout', 'temporary', 'permanent', 'internal_error', 'not_found')),
    CHECK (finished_at IS NULL OR finished_at >= started_at),
    CHECK ((operation = 'tool') = (tool_name IS NOT NULL))
);

CREATE INDEX lab3_trace_spans_trace_started_idx ON lab3_trace_spans (trace_id, started_at, span_id);
