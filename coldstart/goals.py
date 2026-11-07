"""Goal extraction: the things the user cares about, drawn from what they actually discuss.

We shortlist aspiration-bearing messages with generic (name-free) cues, then the LLM turns them into
up to 10 goals, each citing real message ids as evidence. Evidence ids are validated against the
input; a goal with no valid evidence is dropped, because inventing goals costs points.
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from .load import DataLoader
from .llm import LLMClient
from .validate import AREAS, GOAL_TYPES, HORIZONS

# Generic aspiration / intent language (no personal names, no dev specifics).
ASPIRATION = re.compile(
    r"\b(want to|wanna|plan to|planning|goal|target|aim|aiming|hope to|hoping|dream|plan on|"
    r"this year|next year|by (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|year end|"
    r"end of)|plan is|resolution|plan for|save up|saving for|invest|learn|start|launch|grow|"
    r"scale|build|expand|get fit|lose weight|marathon|run a|training for|buy a|move to|shift to|"
    r"renovat|admission|get into|crack|milestone|long term|some day|someday|one day|eventually)\b",
    re.I,
)

_SYS = (
    "You read a person's WhatsApp and list the GOALS they genuinely care about — things they are "
    "working toward over months or years, shown repeatedly in what they talk about. Not one-off "
    "errands. For each goal give: title (short, concrete), "
    "area (self|spouse|kids|parents|work|friends|experiences|finances|health|home), "
    "type (habit|milestone|experience), horizon (3mo|1yr|3yr|5yr), and evidence = a list of 1-3 "
    "message ids (from the ids shown) that show it. Use only ids that appear in the input. "
    "Prefer fewer, well-evidenced goals over many weak ones. Return at most 10.\n"
    'Return ONLY JSON: {"goals":[{"title":...,"area":...,"type":...,"horizon":...,"evidence":["M.."]}]}'
)


def _candidates(loader: DataLoader, cap: int = 140) -> List[Tuple[str, str]]:
    cands: List[Tuple[str, str, int]] = []
    for m in loader.raw_messages:
        body = (m.get("body") or "").strip()
        if len(body) < 12 or len(body) > 400:
            continue
        if "forwarded" in body.lower():
            continue
        if ASPIRATION.search(body):
            # Prefer the user's own statements of intent.
            weight = 2 if m.get("sender_id") == "ME" else 1
            cands.append((m["id"], " ".join(body.split()), weight))
    cands.sort(key=lambda x: x[2], reverse=True)
    return [(mid, text) for mid, text, _ in cands[:cap]]


def extract_goals(loader: DataLoader, llm: Optional[LLMClient] = None) -> List[Dict[str, Any]]:
    cands = _candidates(loader)
    valid_ids = set(loader.messages_by_id.keys())
    if not cands or not (llm and llm.is_available):
        return []

    # Chunk candidates so each prompt stays small; merge afterwards.
    chunks = [cands[i:i + 70] for i in range(0, len(cands), 70)]
    payloads = []
    for ch in chunks:
        listing = "\n".join(f"{mid}: {text}" for mid, text in ch)
        payloads.append([
            {"role": "system", "content": _SYS},
            {"role": "user", "content": "Messages:\n" + listing},
        ])

    responses = llm.map_json(payloads, strong=True, max_tokens=1500)
    goals: List[Dict[str, Any]] = []
    seen_titles = set()
    for resp in responses:
        if not isinstance(resp, dict):
            continue
        for g in resp.get("goals") or []:
            if not isinstance(g, dict):
                continue
            title = str(g.get("title") or "").strip()
            area = g.get("area")
            gtype = g.get("type")
            horizon = g.get("horizon")
            ev = [e for e in (g.get("evidence") or []) if e in valid_ids]
            if not title or area not in AREAS or gtype not in GOAL_TYPES or horizon not in HORIZONS:
                continue
            if not ev:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            goals.append({
                "title": title, "area": area, "type": gtype,
                "horizon": horizon, "evidence": ev[:3],
            })
    return goals[:10]
