"""Chat assignment: every chat tied to exactly one person or one organisation.

Direct (1:1) chats tie to the other person unless the organisation step decided the chat belongs to
an organisation (e.g. a vendor or office line). Group chats tie to their organisation. Any group
left unmapped is attached by the circle-majority of its members so that no chat is left untied.
"""
from collections import Counter
from typing import Dict, List, Any
from .load import DataLoader

HOME_CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative"}
WORK_CIRCLES = {"colleague", "partner", "client"}


def assign_chats(
    loader: DataLoader,
    chat_map: Dict[str, str],
    organisations: List[Dict[str, Any]],
    circles: Dict[str, str],
) -> List[Dict[str, str]]:
    ties: List[Dict[str, str]] = []
    valid_org_ids = [o["id"] for o in organisations]
    org_ids = set(valid_org_ids)

    home_default = next((o["id"] for o in organisations if o["side"] == "home"), None)
    work_default = next((o["id"] for o in organisations if o["side"] == "work"), None)
    any_default = valid_org_ids[0] if valid_org_ids else None

    for chat in loader.raw_chats:
        cid = chat["chat_id"]
        is_group = chat.get("is_group", False)
        parts = [p for p in chat.get("participants", []) if p != "ME"]

        # 1:1 chats always tie to the other person (a vendor/client is not an organisation).
        if not is_group:
            if parts:
                ties.append({"chat_id": cid, "person": parts[0]})
            elif any_default:
                ties.append({"chat_id": cid, "organisation": any_default})
            continue

        mapped = chat_map.get(cid)
        if mapped in org_ids:
            ties.append({"chat_id": cid, "organisation": mapped})
            continue

        # Unmapped group chat: attach by circle majority of members.
        circ = Counter(circles.get(p, "friend") for p in parts)
        home_votes = sum(circ[k] for k in HOME_CIRCLES) + circ["staff"]
        work_votes = sum(circ[k] for k in WORK_CIRCLES) + circ["vendor"]
        target = None
        if home_votes > work_votes and home_default:
            target = home_default
        elif work_default:
            target = work_default
        else:
            target = home_default or any_default
        if target:
            ties.append({"chat_id": cid, "organisation": target})
        elif parts:
            # No organisations at all: fall back to a person so the chat is still tied.
            ties.append({"chat_id": cid, "person": parts[0]})

    return ties
