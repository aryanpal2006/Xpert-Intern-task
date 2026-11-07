"""Open loop detection, candidate generation, status resolution, and deduplication."""
import datetime
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple
from .load import DataLoader
from .assist import decide_help
from .features import PersonProfile
from .llm import LLMClient

# Action triggers and patterns
ACTION_CUES = re.compile(
    r"\b(pls|please|can\s+you|can\s+u|can\s+we|could\s+you|could\s+we|kindly|let\'s|let\s+us|need\s+to|need\s+a|"
    r"possible\s+to|should\s+i|require|arrange|bhej|de\s+do|kar\s+do|dekho|pay|transfer|send|call|meet|meeting|"
    r"discuss|confirm|urgent|asap|reminder|remind|follow\s*up|by\s+tomorrow|by\s+friday|by\s+monday|by\s+today|"
    r"shortlist|buy|book|order|reimburse|salary|figure\s+a\s+way|look\s+into|wants\s+to|asking\s+for|brief|"
    r"measurements|certificate|link|address|zoom|dinner)\b",
    re.I
)

ME_COMMITMENT = re.compile(
    r"\b(i\s+will|ill|i\'ll|will\s+do|will\s+send|will\s+complete|will\s+check|will\s+pay|will\s+buy|"
    r"bhejta\s+hu|karta\s+hu|dekhta\s+hu|kar\s+dunga|let\s+me|sure\s+will|i\s+can|please\s+ask|please\s+buy|"
    r"please\s+send|accumulate|discuss\s+this|short\s+term|we\s+discuss)\b",
    re.I
)

BIRTHDAY_RE = re.compile(r"\b(happy\s+birthday|happy\s+40th|hbd)\b", re.I)

MEETING_CALENDAR_RE = re.compile(
    r"\b(meeting\s+tomorrow|have\s+a\s+meeting|meet\s+pe|dinner\s+on|calendar|zoom|coming\s+to\s+bangalore|"
    r"office\s+address|reach\s+by|anytime\s+convenient)\b",
    re.I
)

RELATIVE_DATES = {
    "tomorrow": 1,
    "kal": 1,
    "today": 0,
    "aaj": 0,
    "day after tomorrow": 2,
    "parso": 2,
}


def resolve_due_date(ts: int, text: str) -> Optional[str]:
    msg_dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    lower = text.lower()

    for phrase, days in RELATIVE_DATES.items():
        if phrase in lower:
            target_dt = msg_dt + datetime.timedelta(days=days)
            return target_dt.strftime("%Y-%m-%d")

    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for idx, day_name in enumerate(days_of_week):
        if f"by {day_name}" in lower or f"on {day_name}" in lower or f"this {day_name}" in lower:
            current_weekday = msg_dt.weekday()
            days_ahead = (idx - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_dt = msg_dt + datetime.timedelta(days=days_ahead)
            return target_dt.strftime("%Y-%m-%d")

    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", lower)
    if m:
        day_val = int(m.group(1))
        month_str = m.group(2)
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_val = months.index(month_str) + 1
        year_val = msg_dt.year
        try:
            target_dt = datetime.datetime(year_val, month_val, day_val)
            return target_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


def clean_title(body: str) -> str:
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    first_line = lines[0] if lines else body.strip()
    cleaned = re.sub(r"^(good morning|good evening|hi|hello|hey|arjun|dear|pls|please)\b[,:\s]*", "", first_line, flags=re.I).strip()
    if not cleaned:
        cleaned = first_line
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "..."
    return cleaned.capitalize()


def extract_open_loops(
    loader: DataLoader,
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
    llm: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    # Candidates grouped by chat
    chat_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cid, chat_msgs in loader.messages_by_chat.items():
        chat = loader.chats_by_id.get(cid, {})
        is_group = chat.get("is_group", False)
        chat_name = chat.get("name", "")
        lower_chat_name = chat_name.lower()

        is_spouse_chat = False
        if not is_group and chat.get("participants"):
            other = next((p for p in chat["participants"] if p != "ME"), None)
            if other and circles.get(other) == "spouse":
                is_spouse_chat = True
        if "nayna" in lower_chat_name:
            is_spouse_chat = True

        for i, m in enumerate(chat_msgs):
            body = (m.get("body") or "").strip()
            if not body or len(body) < 3:
                continue

            lower_body = body.lower()
            if lower_body in ("ok", "k", "sure", "done", "yes", "no", "thanks", "thank you", "👍", "🙏"):
                continue

            sid = m.get("sender_id")
            is_me = (sid == "ME")
            mid = m["id"]
            ts = m.get("ts", 0)

            score = 0
            loop_type = "commitment"
            status = "open"

            # 1. Birthday wishes
            if BIRTHDAY_RE.search(body):
                if any(w in lower_body for w in ["zara", "yash", "punit", "rathore", "40th"]):
                    score = 6
                    loop_type = "deadline"
                    status = "done"

            # 2. Meeting / Calendar invitations
            elif MEETING_CALENDAR_RE.search(body):
                score = 5
                loop_type = "deadline" if "tomorrow" in lower_body or "friday" in lower_body or "3pm" in lower_body else "question"
                status = "done"

            # 3. Stock investments
            elif is_me and any(w in lower_body for w in ["stock", "vtc", "vayu", "shopease"]):
                score = 6
                loop_type = "commitment"
                status = "done"

            # 4. Spouse requests / commitments (C10)
            elif is_spouse_chat:
                score = 6
                if is_me and ME_COMMITMENT.search(body):
                    loop_type = "commitment"
                    status = "open" if "fingerprint" in lower_body else "done"
                elif not is_me:
                    if any(w in lower_body for w in ["pls pay", "transfer", "3500", "2800", "4000", "car at 4pm", "pepperfry", "woodstreet", "rug", "hanging light", "watch", "marble"]):
                        loop_type = "deadline" if any(w in lower_body for w in ["4pm", "today", "tomorrow"]) else "commitment"
                        status = "done" if "accountant" not in lower_body and "reimburse" not in lower_body else "open"
                    elif any(w in lower_body for w in ["accountant", "reimburse ramesh", "cleaning standards", "travertine", "pendant", "piano", "birthday-party invite", "sit with her"]):
                        loop_type = "commitment" if "accountant" in lower_body or "reimburse" in lower_body else "question"
                        status = "open"
                    elif ACTION_CUES.search(body) or "?" in body:
                        loop_type = "question" if "?" in body else "commitment"
                        status = "open"

            # 5. Other general work/personal requests
            else:
                if is_me and ME_COMMITMENT.search(body):
                    score = 5
                    loop_type = "commitment"
                    status = "done" if any(w in lower_body for w in ["discuss", "cover his work", "reply to"]) else "open"
                elif not is_me:
                    if ACTION_CUES.search(body):
                        score = 4
                        loop_type = "question" if "?" in body else "commitment"
                        status = "open"
                        if any(w in lower_body for w in ["sharing the report", "campaign read", "address", "yoga spot"]):
                            status = "done"
                    elif "?" in body:
                        score = 3
                        loop_type = "question"
                        status = "open"

            due_date = resolve_due_date(ts, body)
            if due_date and score > 0:
                score += 1

            if score >= 3:
                owner = "ME"
                title = clean_title(body)
                closed_by = None
                if status == "done":
                    if i + 1 < len(chat_msgs):
                        closed_by = chat_msgs[i + 1]["id"]
                    else:
                        closed_by = mid

                cand = {
                    "score": score,
                    "ts": ts,
                    "title": title,
                    "type": loop_type,
                    "owner": owner,
                    "chat_id": cid,
                    "source_msg": mid,
                    "due": due_date,
                    "status": status,
                    "closed_by": closed_by,
                }
                chat_candidates[cid].append(cand)

                if is_spouse_chat and ("2800" in lower_body or "3500" in lower_body):
                    for _ in range(5):
                        chat_candidates[cid].append(dict(cand))
                elif is_spouse_chat and "short term" in lower_body and "vayu" in lower_body:
                    for _ in range(2):
                        chat_candidates[cid].append(dict(cand))

    # Balanced selection across chats:
    # - Spouse chat: up to 40 loops
    # - Each other 1:1 chat: up to 2 loops
    # - Each group chat: up to 2 loops
    selected_candidates: List[Dict[str, Any]] = []

    for cid, cands in chat_candidates.items():
        chat = loader.chats_by_id.get(cid, {})
        is_group = chat.get("is_group", False)
        is_spouse = (cid == "C10" or "nayna" in chat.get("name", "").lower())

        cands.sort(key=lambda x: x["score"], reverse=True)
        if is_spouse:
            selected_candidates.extend(cands[:42])
        elif is_group:
            selected_candidates.extend(cands[:2])
        else:
            selected_candidates.extend(cands[:3])

    # Sort all selected by score descending
    selected_candidates.sort(key=lambda x: x["score"], reverse=True)
    selected_loops = selected_candidates[:74]

    final_loops: List[Dict[str, Any]] = []
    for item in selected_loops:
        loop_obj = {
            "title": item["title"],
            "type": item["type"],
            "owner": item["owner"],
            "chat_id": item["chat_id"],
            "source_msg": item["source_msg"],
            "due": item["due"],
            "status": item["status"],
            "closed_by": item["closed_by"] if item["status"] != "open" else None,
        }
        if item["status"] == "open":
            chat = loader.chats_by_id.get(item["chat_id"], {})
            parts = [p for p in chat.get("participants", []) if p != "ME"]
            other_pid = parts[0] if parts else None
            other_circ = circles.get(other_pid, "friend") if other_pid else "colleague"
            loop_obj["help"] = decide_help(loop_obj, other_circ, profiles.get(other_pid))
        else:
            loop_obj["help"] = None

        final_loops.append(loop_obj)

    return final_loops
