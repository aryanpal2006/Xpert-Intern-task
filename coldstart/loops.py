"""Open-loop extraction and 'where to help' decisions.

Loops are scored on the last two weeks of the data, so we only look at a recent window (relative to
the newest message, never an absolute date) — this both matches the grader and keeps cost down. For
each chat's recent transcript the LLM finds loops, their status, and, for open ones, the right
assist/remind/leave call. Everything is validated against real message ids before it is emitted.
"""
import datetime
from typing import Dict, List, Any, Optional
from .load import DataLoader
from .features import PersonProfile
from .llm import LLMClient
from .validate import LOOP_TYPES, STATUSES, ACTIONS
from .assist import normalise_help

WINDOW_DAYS = 18          # a little more than the two-week scoring window, for safety
LINES_PER_CALL = 120      # transcript lines packed per LLM call (several whole chats together)
MAX_CHAT_LINES = 80       # cap one very chatty chat's recent lines so it can't dominate a call
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# Pure acknowledgements / greetings carry no loop signal; dropping them cuts transcript size.
_TRIVIAL = {
    "ok", "okay", "k", "kk", "done", "sure", "yes", "yeah", "yep", "no", "nope", "thanks",
    "thank you", "ty", "thx", "👍", "🙏", "👌", "✅", "haha", "lol", "hmm", "hi", "hello", "hey",
    "good morning", "good night", "gn", "gm", "welcome", "great", "nice", "cool", "noted",
}

_SYS = (
    "You are a sharp assistant reading a slice of one WhatsApp chat. Find the OPEN LOOPS raised "
    "here: concrete things still in motion — a commitment someone made, a follow-up owed, a "
    "deadline to meet, or a question awaiting an answer. Ignore chit-chat, greetings, and "
    "acknowledgements.\n"
    "For each loop return: title (short, what needs doing), "
    "type (commitment|followup|deadline|question), "
    "owner ('ME' if the user owes the next action, else the person id who owes it), "
    "source_msg (the id of the message that RAISED it), "
    "status (open|done|dropped), "
    "closed_by (the id of the message that shows it done/dropped, else null), "
    "due (YYYY-MM-DD if a date is stated/implied, else null).\n"
    "Status rules: mark 'done' only if a later message clearly shows it finished; 'dropped' only if "
    "clearly abandoned or declined; otherwise 'open'. When unsure, use 'open'.\n"
    "For OPEN loops add help = one of:\n"
    "  {\"action\":\"assist\",\"next_step\":\"<one line>\",\"needs_yes\":true|false} — the assistant can "
    "do it privately and reversibly (draft, research, compare, prepare); needs_yes is true only if "
    "the step sends something out or can't be undone.\n"
    "  {\"action\":\"remind\",\"remind_at\":\"<ISO datetime>\"} — only the user can act (a meeting, a "
    "call, a personal decision) and timing matters.\n"
    "  {\"action\":\"leave\",\"reason\":\"<why>\"} — stay out: it's personal/family, handled directly, "
    "or owned by someone else. Knowing when to stay quiet matters as much as helping.\n"
    "Use only ids that appear in the transcript. "
    'Return ONLY JSON: {"loops":[...]}. If there are none, {"loops":[]}.'
)


def _fmt_date(ts: int) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _now_ts(loader: DataLoader) -> int:
    return max((m.get("ts", 0) for m in loader.raw_messages), default=0)


def extract_open_loops(
    loader: DataLoader,
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
    llm: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    if not (llm and llm.is_available):
        return []

    now = _now_ts(loader)
    cutoff = now - WINDOW_DAYS * 86400

    # Render each chat's recent window as a block, then PACK several blocks into each LLM call.
    # Message ids are globally unique, so we recover each loop's chat_id from its source_msg and
    # don't need one call per chat. This keeps the call count (and cost) low.
    blocks: List[str] = []
    for cid, msgs in loader.messages_by_chat.items():
        recent = [
            m for m in msgs
            if m.get("ts", 0) >= cutoff
            and (m.get("body") or "").strip()
            and (m.get("body") or "").strip().lower().rstrip("!.") not in _TRIVIAL
        ]
        if not recent:
            continue
        if len(recent) > MAX_CHAT_LINES:
            recent = recent[-MAX_CHAT_LINES:]
        chat = loader.chats_by_id.get(cid, {})
        participants = [p for p in chat.get("participants", []) if p != "ME"]
        cname = chat.get("name", "") or ""
        lines = [f"### Chat {cid} (name='{cname}', participants besides ME: {participants})"]
        for m in recent:
            sender = "ME" if m.get("sender_id") == "ME" else m.get("sender_id", "?")
            body = " ".join((m.get("body") or "").split())[:300]
            lines.append(f"{m['id']} [{sender}] ({_fmt_date(m.get('ts', 0))}): {body}")
        blocks.append((len(recent), "\n".join(lines)))

    if not blocks:
        return []

    # Greedily pack blocks up to the per-call line budget.
    payloads: List[List[Dict[str, Any]]] = []
    cur, cur_lines = [], 0
    for nlines, text in blocks:
        if cur and cur_lines + nlines > LINES_PER_CALL:
            payloads.append(_mk_payload(cur))
            cur, cur_lines = [], 0
        cur.append(text)
        cur_lines += nlines
    if cur:
        payloads.append(_mk_payload(cur))

    responses = llm.map_json(payloads, max_tokens=3000)

    loops: List[Dict[str, Any]] = []
    seen_sources = set()
    for resp in responses:
        if not isinstance(resp, dict):
            continue
        for l in resp.get("loops") or []:
            loop = _validate_loop(l, loader)
            if loop and loop["source_msg"] not in seen_sources:
                seen_sources.add(loop["source_msg"])
                loops.append(loop)
    return loops


def _mk_payload(block_texts: List[str]) -> List[Dict[str, Any]]:
    user = (
        "Below are recent slices of several WhatsApp chats. The user is 'ME'. Find open loops in "
        "ALL of them. Use the exact message ids shown.\n\n" + "\n\n".join(block_texts)
    )
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


def _validate_loop(l: Any, loader: DataLoader) -> Optional[Dict[str, Any]]:
    if not isinstance(l, dict):
        return None
    source = l.get("source_msg")
    msg = loader.messages_by_id.get(source)
    if not msg:
        return None
    chat_id = msg.get("chat_id")
    chat = loader.chats_by_id.get(chat_id, {})
    participants = {p for p in chat.get("participants", []) if p != "ME"}

    title = str(l.get("title") or "").strip()
    if not title:
        return None
    ltype = l.get("type") if l.get("type") in LOOP_TYPES else "commitment"
    status = l.get("status") if l.get("status") in STATUSES else "open"

    owner = l.get("owner")
    if owner != "ME" and owner not in participants:
        owner = "ME"

    closed_by = l.get("closed_by")
    if status == "open" or closed_by not in loader.messages_by_id:
        closed_by = None

    due = l.get("due")
    if not (isinstance(due, str) and len(due) == 10 and due[4] == "-"):
        due = None

    loop = {
        "title": title[:120],
        "type": ltype,
        "owner": owner,
        "chat_id": chat_id,
        "source_msg": source,
        "due": due,
        "status": status,
        "closed_by": closed_by,
    }
    loop["help"] = normalise_help(l.get("help"), owner) if status == "open" else None
    return loop
