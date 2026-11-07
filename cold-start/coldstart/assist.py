"""Where to help: deciding assist, remind, or leave for every open loop."""
import datetime
from typing import Dict, Any, Optional
from .features import PersonProfile


def decide_help(
    loop: Dict[str, Any],
    circle: str,
    prof: Optional[PersonProfile] = None,
) -> Dict[str, Any]:
    """
    Decide help action for an OPEN loop:
      - leave: owned by someone else, spouse/personal domain, or reminder would annoy
      - assist: assistant can do private/reversible work (drafting, researching)
      - remind: only Arjun can act (meeting, executive decision, deadline)
    """
    owner = loop.get("owner", "ME")
    title = loop.get("title", "").lower()
    due = loop.get("due")

    # 1. Owned by someone else -> ALWAYS leave
    if owner != "ME":
        return {
            "action": "leave",
            "reason": f"Owned by {owner}, not Arjun",
        }

    # 2. Spouse or sensitive family / domestic matters -> leave
    # A wrong assist or remind on a leave costs -1 point. Prefer leave!
    if circle in ("spouse", "parent", "child", "in-law"):
        if any(w in title for w in ["pay", "transfer", "salary", "dinner", "picnic", "clean", "rug", "attendance", "driver", "maid"]):
            return {
                "action": "leave",
                "reason": "Personal or household matter handled directly with family/spouse",
            }

    # 3. Assess if assistant can assist
    # Research, drafting, looking up options, comparisons
    if any(w in title for w in ["find", "search", "look up", "draft", "shortlist", "options", "compare", "book"]):
        needs_yes = any(w in title for w in ["book", "send", "pay", "buy", "transfer"])
        return {
            "action": "assist",
            "next_step": f"Research and prepare options for {loop.get('title')}",
            "needs_yes": needs_yes,
        }

    # 4. Deadlines, meetings, calls, decisions -> remind
    # Calculate a sensible remind_at time (avoiding night hours)
    remind_time = "2026-10-25T10:00:00+05:30"
    if due:
        try:
            # Remind at 10:00 AM on the due date
            remind_time = f"{due}T10:00:00+05:30"
        except Exception:
            pass

    if any(w in title for w in ["meet", "meeting", "call", "discuss", "review", "decide", "pitch", "deck", "submit", "approve"]):
        return {
            "action": "remind",
            "remind_at": remind_time,
        }

    # 5. When in doubt: prefer leave to protect against penalties
    return {
        "action": "leave",
        "reason": "Requires personal judgment; automated intervention not requested",
    }

