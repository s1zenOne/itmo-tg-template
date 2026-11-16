import hmac
import json
import re
from uuid import UUID

from mcp.types import RequestParamsMeta

from app.mcp_tools.contracts import ToolError

GRANT_KEY_ENV = "ITMO_MCP_GRANT_KEY"
GRANT_META = "itmo-tg/reminder-authorization"
SIGNATURE_PATTERN = re.compile(r"[0-9a-f]{64}")


def _signature(key: bytes, tool: str, arguments: dict, owner_id: int, action_id: UUID) -> str:
    data = json.dumps(
        [tool, arguments, owner_id, str(action_id)], sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hmac.digest(key, data, "sha256").hex()


def make_grant(
    key: bytes, tool: str, arguments: dict, owner_id: int, action_id: UUID
) -> RequestParamsMeta:
    return {
        GRANT_META: {
            "owner_id": owner_id,
            "action_id": str(action_id),
            "signature": _signature(key, tool, arguments, owner_id, action_id),
        }
    }


def verify_grant(
    key: bytes | None, tool: str, arguments: dict, meta: RequestParamsMeta | None
) -> tuple[int, UUID]:
    grant = meta.get(GRANT_META) if meta is not None else None
    if (
        key is None
        or not isinstance(grant, dict)
        or set(grant)
        != {
            "owner_id",
            "action_id",
            "signature",
        }
    ):
        raise ToolError("unauthorized")
    owner_id, action_id, signature = grant["owner_id"], grant["action_id"], grant["signature"]
    if (
        type(owner_id) is not int
        or not 0 < owner_id < 2**63
        or not isinstance(action_id, str)
        or len(action_id) != 36
        or not isinstance(signature, str)
        or not SIGNATURE_PATTERN.fullmatch(signature)
    ):
        raise ToolError("unauthorized")
    try:
        parsed = UUID(action_id)
    except ValueError:
        raise ToolError("unauthorized") from None
    if not hmac.compare_digest(signature, _signature(key, tool, arguments, owner_id, parsed)):
        raise ToolError("unauthorized")
    return owner_id, parsed
