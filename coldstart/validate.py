"""Schema and reference validation for output.json, matching check.py exactly."""
from typing import Dict, List, Any

CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
ORG_TYPES = {"company", "residence", "vacation home", "workspace", "society", "school", "other"}
ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
AREAS = {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"}
GOAL_TYPES = {"habit", "milestone", "experience"}
HORIZONS = {"3mo", "1yr", "3yr", "5yr"}
LOOP_TYPES = {"commitment", "followup", "deadline", "question"}
STATUSES = {"open", "done", "dropped"}
ACTIONS = {"assist", "remind", "leave"}


def validate_output(out: Dict[str, Any], chats: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> List[str]:
    """Returns a list of validation errors. Empty list means the output is 100% valid."""
    errs = []
    if not isinstance(out, dict):
        return ["The output must be a JSON object."]
    for key in ("people", "organisations", "chats", "goals", "open_loops"):
        if not isinstance(out.get(key), list):
            errs.append(f'"{key}" must be a list.')
    if errs:
        return errs
    extra = set(out) - {"people", "organisations", "chats", "goals", "open_loops"}
    if extra:
        errs.append(f"Unknown top-level keys: {sorted(extra)}.")

    chat_ids = {c["chat_id"] for c in chats}
    msg_ids = {m["id"] for m in messages}
    person_ids = {p for c in chats for p in c.get("participants", [])} - {"ME"}
    org_ids = set()
    for i, o in enumerate(out["organisations"]):
        where = f"organisations[{i}]"
        if not isinstance(o, dict) or not o.get("id"):
            errs.append(f"{where} needs an id.")
            continue
        if o["id"] in org_ids:
            errs.append(f'{where}: the id "{o["id"]}" is used twice.')
        org_ids.add(o["id"])
        if not o.get("name"):
            errs.append(f"{where} needs a name.")
        if o.get("type") not in ORG_TYPES:
            errs.append(f"{where}: type must be one of {sorted(ORG_TYPES)}.")
        if o.get("side") not in ("home", "work"):
            errs.append(f'{where}: side must be "home" or "work".')
        if o.get("role") not in ORG_ROLES:
            errs.append(f"{where}: role must be one of {sorted(ORG_ROLES)}.")

    seen_people = set()
    for i, p in enumerate(out["people"]):
        where = f"people[{i}]"
        pid = p.get("id") if isinstance(p, dict) else None
        if pid not in person_ids:
            errs.append(f'{where}: "{pid}" is not a person in the input (and ME is never listed).')
            continue
        if pid in seen_people:
            errs.append(f'{where}: "{pid}" is listed twice.')
        seen_people.add(pid)
        if p.get("circle") not in CIRCLES:
            errs.append(f"{where}: circle must be one of {sorted(CIRCLES)}.")
        for link in p.get("links", []) or []:
            if link.get("organisation") not in org_ids:
                errs.append(f'{where}: links to an unknown organisation "{link.get("organisation")}".')
            if link.get("link") not in ("runs", "supports"):
                errs.append(f'{where}: a link must be "runs" or "supports".')
        for other in p.get("looks_after", []) or []:
            if other not in person_ids:
                errs.append(f'{where}: looks_after names an unknown person "{other}".')

    tied = set()
    for i, c in enumerate(out["chats"]):
        where = f"chats[{i}]"
        cid = c.get("chat_id") if isinstance(c, dict) else None
        if cid not in chat_ids:
            errs.append(f'{where}: "{cid}" is not a chat in the input.')
            continue
        if cid in tied:
            errs.append(f'{where}: "{cid}" is tied twice. Tie each chat once.')
        tied.add(cid)
        has_p, has_o = "person" in c, "organisation" in c
        if has_p == has_o:
            errs.append(f"{where}: give exactly one of person or organisation.")
        elif has_p and c["person"] not in person_ids:
            errs.append(f'{where}: unknown person "{c["person"]}".')
        elif has_o and c["organisation"] not in org_ids:
            errs.append(f'{where}: unknown organisation "{c["organisation"]}".')
    missing = chat_ids - tied
    if missing:
        errs.append(f"{len(missing)} chat(s) are not tied to anyone, for example {sorted(missing)[:3]}.")

    if len(out["goals"]) > 10:
        errs.append("At most 10 goals.")
    for i, g in enumerate(out["goals"]):
        where = f"goals[{i}]"
        if not g.get("title"):
            errs.append(f"{where} needs a title.")
        if g.get("area") not in AREAS:
            errs.append(f"{where}: area must be one of {sorted(AREAS)}.")
        if g.get("type") not in GOAL_TYPES:
            errs.append(f"{where}: type must be one of {sorted(GOAL_TYPES)}.")
        if g.get("horizon") not in HORIZONS:
            errs.append(f"{where}: horizon must be one of {sorted(HORIZONS)}.")
        ev = g.get("evidence") or []
        if not ev:
            errs.append(f"{where}: give at least one evidence message.")
        for m in ev:
            if m not in msg_ids:
                errs.append(f'{where}: evidence "{m}" is not a message in the input.')

    for i, l in enumerate(out["open_loops"]):
        where = f"open_loops[{i}]"
        if not l.get("title"):
            errs.append(f"{where} needs a title.")
        if l.get("type") not in LOOP_TYPES:
            errs.append(f"{where}: type must be one of {sorted(LOOP_TYPES)}.")
        if l.get("owner") != "ME" and l.get("owner") not in person_ids:
            errs.append(f'{where}: owner must be "ME" or a person id.')
        if l.get("chat_id") not in chat_ids:
            errs.append(f"{where}: unknown chat_id.")
        if l.get("source_msg") not in msg_ids:
            errs.append(f"{where}: source_msg is not a message in the input.")
        if l.get("status") not in STATUSES:
            errs.append(f"{where}: status must be one of {sorted(STATUSES)}.")
        if l.get("closed_by") is not None and l.get("closed_by") not in msg_ids:
            errs.append(f"{where}: closed_by is not a message in the input.")
        help_ = l.get("help")
        if help_ is not None:
            if l.get("status") != "open":
                errs.append(f"{where}: help is only for open loops.")
            if not isinstance(help_, dict) or help_.get("action") not in ACTIONS:
                errs.append(f"{where}: help.action must be one of {sorted(ACTIONS)}.")
    return errs

