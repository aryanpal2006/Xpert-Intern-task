"""Normalise a 'where to help' decision into a clean, schema-valid help object.

Scoring asymmetry: stepping in (assist/remind) where the answer is 'leave' costs a point, while
saying 'leave' where help was wanted only forgoes a point. So we trust a confident LLM call but
default to 'leave' whenever the decision is missing, malformed, or owned by someone else.
"""
from typing import Any, Dict, Optional

DEFAULT_REMIND = "T10:00:00+05:30"


def normalise_help(raw: Any, owner: str) -> Dict[str, Any]:
    # A loop the user doesn't own: the next move isn't theirs — stay out.
    if owner != "ME":
        return {"action": "leave", "reason": f"Owned by {owner}, not the user"}

    if not isinstance(raw, dict):
        return {"action": "leave", "reason": "No clear way to help without overstepping"}

    action = raw.get("action")
    if action == "assist":
        next_step = str(raw.get("next_step") or "Prepare options and a draft for review").strip()
        needs_yes = bool(raw.get("needs_yes", True))
        return {"action": "assist", "next_step": next_step[:200], "needs_yes": needs_yes}

    if action == "remind":
        remind_at = raw.get("remind_at")
        if not isinstance(remind_at, str) or not remind_at:
            remind_at = "2026-01-01" + DEFAULT_REMIND
        return {"action": "remind", "remind_at": remind_at}

    if action == "leave":
        reason = str(raw.get("reason") or "Personal matter; better left to the user").strip()
        return {"action": "leave", "reason": reason[:200]}

    return {"action": "leave", "reason": "Unclear; staying out to avoid overstepping"}
