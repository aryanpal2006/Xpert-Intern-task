"""Organisation discovery and link assignment.

We cluster group chats (and business-like 1:1 chats) into organisations and decide each person's
link (runs vs supports). The LLM reads chat content and the people's circles; it is told to group
by the real entity, not by name, because the grader matches organisations by the chats and people
attached to them, not by name. A name-free structural fallback runs without a model.
"""
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
from .load import DataLoader
from .features import PersonProfile
from .llm import LLMClient

WORK_CIRCLES = {"colleague", "partner", "client"}
HOME_CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative"}

_SYS = (
    "You are organising a person's world from their WhatsApp GROUP chats. Group them into the "
    "user's OWN ORGANISATIONS only: a company/venture they run or are part of, their home, a second "
    "home, a society/building they live in, their kids' school, or a workspace they own.\n"
    "NEVER invent an organisation for a single external vendor, shop, supplier, agency, or client — "
    "a client's or vendor's own company is NOT the user's organisation. Only make organisations the "
    "user themselves belongs to or owns. Judge by what each chat is about, not by names (anonymised).\n"
    "For each organisation give: id (a short slug you invent), name, "
    "type (company|residence|vacation home|workspace|society|school|other), "
    "side (home|work), role (owner|partner|helper|resident|parent|member|other) = the USER's role.\n"
    "Then assign EVERY listed group chat to exactly one of YOUR organisation ids (chat_assignments). "
    "If a group doesn't fit any real organisation of the user's, attach it to the closest one (home "
    "for domestic/family, a work org for work). For each person who belongs to an organisation give "
    "a link: 'runs' if they run/own it with the user, 'supports' if they work for or serve it.\n"
    'Return ONLY JSON: {"organisations":[...],"chat_assignments":{"C.":"slug"},'
    '"links":[{"person":"P.","organisation":"slug","link":"runs|supports"}]}'
)


def _chat_summary(loader: DataLoader, cid: str, max_msgs: int = 3) -> str:
    chat = loader.chats_by_id.get(cid, {})
    name = chat.get("name", "") or ""
    parts = [p for p in chat.get("participants", []) if p != "ME"]
    msgs = loader.messages_by_chat.get(cid, [])
    samples = []
    for m in msgs:
        b = (m.get("body") or "").strip()
        if b:
            samples.append(" ".join(b.split())[:120])
        if len(samples) >= max_msgs:
            break
    kind = "group" if chat.get("is_group") else "direct"
    line = f"{cid} [{kind}] name='{name}' people={parts}"
    if samples:
        line += " | " + " || ".join(samples)
    return line


def _llm_discover(
    loader: DataLoader,
    circles: Dict[str, str],
    llm: LLMClient,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, str]]]]:
    seed_orgs = loader.seed.get("organisations", []) or []

    # Only GROUP chats seed organisations. 1:1 chats tie to the person (handled in chats.py);
    # a lone vendor/client is not one of the user's organisations.
    cand_chats = [c["chat_id"] for c in loader.raw_chats if c.get("is_group")]

    if not cand_chats:
        return None

    summaries = "\n".join(_chat_summary(loader, cid) for cid in cand_chats)
    people_hint = ", ".join(f"{pid}:{circles.get(pid, '?')}" for pid in loader.all_people)
    seed_hint = (
        f"The user told us two of their organisations are named: {seed_orgs}. "
        "Include these as organisations.\n" if seed_orgs else ""
    )
    user = (
        seed_hint +
        "People and their circle labels:\n" + people_hint +
        "\n\nChats to organise:\n" + summaries
    )
    resp = llm.chat_json(
        [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        strong=True, max_tokens=2000,
    )
    if not isinstance(resp, dict):
        return None

    raw_orgs = resp.get("organisations") or []
    orgs: List[Dict[str, Any]] = []
    valid_ids = set()
    from .validate import ORG_TYPES, ORG_ROLES
    for o in raw_orgs:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("id") or "").strip()
        name = str(o.get("name") or "").strip()
        if not oid or not name or oid in valid_ids:
            continue
        otype = o.get("type") if o.get("type") in ORG_TYPES else "other"
        side = o.get("side") if o.get("side") in ("home", "work") else "work"
        role = o.get("role") if o.get("role") in ORG_ROLES else "owner"
        orgs.append({"id": oid, "name": name, "type": otype, "side": side, "role": role})
        valid_ids.add(oid)

    if not orgs:
        return None

    chat_map: Dict[str, str] = {}
    for cid, oid in (resp.get("chat_assignments") or {}).items():
        if cid in loader.chats_by_id and oid in valid_ids:
            chat_map[cid] = oid

    links: List[Dict[str, str]] = []
    for lk in resp.get("links") or []:
        if not isinstance(lk, dict):
            continue
        pid = lk.get("person")
        oid = lk.get("organisation")
        link = lk.get("link")
        if pid in loader.all_people and oid in valid_ids and link in ("runs", "supports"):
            links.append({"person": pid, "organisation": oid, "link": link})

    return orgs, chat_map, links


def _fallback_discover(
    loader: DataLoader,
    circles: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, str]]]:
    """Name-free structural clustering when no LLM is available."""
    seed_orgs = loader.seed.get("organisations", []) or []
    orgs: List[Dict[str, Any]] = []
    work_id = None
    if seed_orgs:
        for i, name in enumerate(seed_orgs):
            oid = f"work{i+1}"
            orgs.append({"id": oid, "name": name, "type": "company", "side": "work", "role": "owner"})
        work_id = orgs[0]["id"]
    else:
        work_id = "work1"
        orgs.append({"id": work_id, "name": "Work", "type": "company", "side": "work", "role": "owner"})

    home_id = None
    if any(circles.get(p) in HOME_CIRCLES or circles.get(p) == "staff" for p in loader.all_people):
        home_id = "home"
        orgs.append({"id": home_id, "name": "Home", "type": "residence", "side": "home", "role": "owner"})

    chat_map: Dict[str, str] = {}
    for c in loader.raw_chats:
        cid = c["chat_id"]
        if not c.get("is_group"):
            continue  # 1:1 chats tie to the person, not an organisation
        parts = [p for p in c.get("participants", []) if p != "ME"]
        circ = Counter(circles.get(p, "friend") for p in parts)
        home_votes = sum(circ[k] for k in HOME_CIRCLES) + circ["staff"]
        work_votes = sum(circ[k] for k in WORK_CIRCLES) + circ["vendor"]
        if home_id and home_votes >= work_votes:
            chat_map[cid] = home_id
        else:
            chat_map[cid] = work_id

    links: List[Dict[str, str]] = []
    for pid in loader.all_people:
        circ = circles.get(pid)
        if circ in ("colleague", "partner"):
            links.append({"person": pid, "organisation": work_id, "link": "runs"})
        elif circ == "staff" and home_id:
            links.append({"person": pid, "organisation": home_id, "link": "supports"})
    return orgs, chat_map, links


def discover_organisations(
    loader: DataLoader,
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
    llm: Optional[LLMClient] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, List[Dict[str, str]]], Dict[str, List[str]]]:
    result = None
    if llm and llm.is_available:
        result = _llm_discover(loader, circles, llm)
    if result is None:
        result = _fallback_discover(loader, circles)
    orgs, chat_map, raw_links = result

    valid_ids = {o["id"] for o in orgs}
    person_links: Dict[str, List[Dict[str, str]]] = {p: [] for p in loader.all_people}
    for lk in raw_links:
        pid, oid, link = lk["person"], lk["organisation"], lk["link"]
        if oid in valid_ids and not any(
            existing["organisation"] == oid for existing in person_links.get(pid, [])
        ):
            person_links.setdefault(pid, []).append({"organisation": oid, "link": link})

    # looks_after: staff who clearly look after one specific family member.
    person_looks_after = _infer_looks_after(loader, circles, profiles)

    return orgs, chat_map, person_links, person_looks_after


def _infer_looks_after(
    loader: DataLoader,
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
) -> Dict[str, List[str]]:
    """Heuristic only: left empty unless there is a strong structural signal. Kept conservative."""
    return {}
