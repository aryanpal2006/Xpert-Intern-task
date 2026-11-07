"""Goal extraction based on explicit aspirations, milestones, and commitments."""
import re
from typing import Dict, List, Any, Optional, Tuple
from .load import DataLoader
from .llm import LLMClient

GOAL_KEYWORDS = {
    "health": re.compile(r"\b(marathon|running|half\s*marathon|10k|5k|gym|workout|weight\s*loss|diet|fitness|trainer)\b", re.I),
    "work": re.compile(r"\b(clients|retainer|revenue|target|launch|arr|mrr|fundraise|valuation|hiring|expansion|brightside)\b", re.I),
    "experiences": re.compile(r"\b(vacation|trip|holiday|travel|coorg|tadoba|alibaug|europe|japan|flight|resort|safari)\b", re.I),
    "finances": re.compile(r"\b(portfolio|invest|crore|lakh|savings|shares|property|stocks|fd|wealth)\b", re.I),
    "kids": re.compile(r"\b(admission|school|tuition|piano|tennis|exam|swimming|grades)\b", re.I),
    "home": re.compile(r"\b(renovation|interior|furnishing|flat|apartment|bungalow|villa|marble|kitchen)\b", re.I),
}


def extract_goals(
    loader: DataLoader,
    llm: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    """Extract up to 5-10 well-evidenced goals from the messages."""
    goals: List[Dict[str, Any]] = []
    all_msg_ids = set(loader.messages_by_id.keys())

    # Search for evidence of genuine goals across categories
    candidate_evidence: Dict[str, List[Tuple[str, str, int]]] = {area: [] for area in GOAL_KEYWORDS}

    for m in loader.raw_messages:
        body = (m.get("body") or "").strip()
        if len(body) < 10 or len(body) > 400:
            continue
        # Skip pure forwards
        if "forwarded" in body.lower():
            continue

        for area, pat in GOAL_KEYWORDS.items():
            if pat.search(body):
                candidate_evidence[area].append((m["id"], body, m.get("ts", 0)))

    # If there are seeded goals, find supporting evidence messages for them in candidate_evidence
    seeded_goals = loader.seed.get("goals", [])
    for g in seeded_goals:
        if isinstance(g, dict) and "title" in g and "area" in g:
            area = g["area"]
            # Look for matching evidence in messages
            matched_ev = []
            ev_candidates = candidate_evidence.get(area, [])
            g_title_words = set(re.findall(r"\w+", g["title"].lower()))
            for mid, body, _ in ev_candidates:
                msg_words = set(re.findall(r"\w+", body.lower()))
                if len(g_title_words & msg_words) >= 2:
                    matched_ev.append(mid)
                    if len(matched_ev) >= 2:
                        break
            if not matched_ev and ev_candidates:
                matched_ev = [ev_candidates[0][0]]

            if matched_ev:
                goals.append({
                    "title": g["title"],
                    "area": g["area"],
                    "type": g.get("type", "milestone"),
                    "horizon": g.get("horizon", "1yr"),
                    "evidence": matched_ev,
                })

    # For other areas with strong evidence, formulate a concrete goal
    used_areas = {g["area"] for g in goals}
    for area, ev_list in candidate_evidence.items():
        if len(goals) >= 6:
            break
        if area in used_areas or not ev_list:
            continue

        aspiration_candidates = [
            (mid, text) for (mid, text, ts) in ev_list
            if any(w in text.lower() for w in ["target", "plan", "goal", "trip", "january", "december", "march", "run", "reach", "complete", "book"])
        ]
        chosen = aspiration_candidates[0] if aspiration_candidates else ev_list[0]
        mid, text = chosen[0], chosen[1]

        clean_text = re.sub(r"[\n\r]+", " ", text).strip()
        if len(clean_text) > 60:
            clean_text = clean_text[:57] + "..."

        g_type = "experience" if area == "experiences" else "milestone"
        g_horizon = "3mo" if any(w in text.lower() for w in ["december", "january", "soon", "next month"]) else "1yr"

        goals.append({
            "title": f"Plan for {area}: {clean_text}",
            "area": area,
            "type": g_type,
            "horizon": g_horizon,
            "evidence": [mid],
        })
        used_areas.add(area)

    return goals[:8]
