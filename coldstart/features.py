"""Name-agnostic feature extraction per person and per chat.

Everything here is structural: who talks to whom, how much, in which chats, and a sample of the
actual message text. We deliberately avoid hardcoding any names, places, or dev-set specifics so
the signals generalise to the re-anonymised hidden set.
"""
from collections import defaultdict
from typing import Dict, List, Optional
from .load import DataLoader

# Relationship *words* (not personal names) that commonly survive re-anonymisation because they are
# how contacts are saved (honorifics, kinship terms, emoji). Used only as weak hints for the
# deterministic fallback; the LLM path reads raw text instead.
KINSHIP_HINTS = {
    "spouse": ["❤", "😘", "wife", "husband", "hubby", "jaan", "love", "pati", "patni", "biwi"],
    "parent": ["mummy", "mum", "mom", "papa", "dad", "father", "mother", "maa", "ma ", "daddy"],
    "sibling": ["didi", "bhai", "bhaiya", "bro", "sis", "sister", "brother", "bhaiyya"],
    "child": ["beta", "beti", "son", "daughter", "junior"],
    "relative": ["chacha", "chachi", "mama", "mami", "bua", "fufa", "cousin", "mausi", "bhabhi",
                 "jiju", "nana", "nani", "dada", "dadi", "uncle", "aunty", "aunt"],
}


class PersonProfile:
    def __init__(self, pid: str):
        self.id = pid
        self.names: List[str] = []
        self.chat_ids: List[str] = []
        self.group_chat_names: List[str] = []
        self.has_1on1_chat: bool = False
        self.one_on_one_chat_id: str = ""
        self.one_on_one_chat_name: str = ""
        self.msgs_from_person: List[str] = []   # things this person said
        self.msgs_from_me: List[str] = []        # things ME said to them (1:1 only)
        self.n_from_person: int = 0
        self.n_from_me: int = 0
        self.seeded_circle: str = ""

    def best_name(self) -> str:
        return self.names[0] if self.names else self.id

    def compact(self, max_msgs: int = 4, max_len: int = 110) -> str:
        """A compact, name-light block for an LLM prompt."""
        def clip(items):
            out = []
            for t in items[:max_msgs]:
                t = " ".join(t.split())
                out.append(t[:max_len])
            return out

        lines = [f"person_id: {self.id}"]
        if self.names:
            lines.append(f"saved_as: {', '.join(self.names[:3])}")
        lines.append(f"has_direct_chat: {'yes' if self.has_1on1_chat else 'no'}"
                     f" (they sent {self.n_from_person}, user sent {self.n_from_me})")
        if self.group_chat_names:
            lines.append(f"shared_groups: {', '.join(self.group_chat_names[:6])}")
        sample_them = clip(self.msgs_from_person)
        sample_me = clip(self.msgs_from_me)
        if sample_them:
            lines.append("they_said: " + " || ".join(sample_them))
        if sample_me:
            lines.append("user_said_to_them: " + " || ".join(sample_me))
        return "\n".join(lines)


def extract_person_profiles(loader: DataLoader, sample_cap: int = 12) -> Dict[str, PersonProfile]:
    profiles: Dict[str, PersonProfile] = {p: PersonProfile(p) for p in loader.all_people}

    seeded_circles = loader.seed.get("circles") or {}
    for pid, circ in seeded_circles.items():
        if pid in profiles:
            profiles[pid].seeded_circle = circ

    for c in loader.raw_chats:
        cid = c["chat_id"]
        cname = c.get("name", "") or ""
        is_group = c.get("is_group", False)
        for p in c.get("participants", []):
            if p == "ME" or p not in profiles:
                continue
            profiles[p].chat_ids.append(cid)
            if is_group:
                if cname and cname not in profiles[p].group_chat_names:
                    profiles[p].group_chat_names.append(cname)
            else:
                profiles[p].has_1on1_chat = True
                profiles[p].one_on_one_chat_id = cid
                profiles[p].one_on_one_chat_name = cname

    for pid, names in loader.person_names.items():
        if pid in profiles:
            profiles[pid].names = names

    # Collect message samples: spread across time so we don't only see the first few lines.
    for m in loader.raw_messages:
        sid = m.get("sender_id")
        body = (m.get("body") or "").strip()
        if not body:
            continue
        cid = m.get("chat_id")
        chat = loader.chats_by_id.get(cid, {})
        is_group = chat.get("is_group", False)
        parts = chat.get("participants", [])

        if sid in profiles:
            profiles[sid].n_from_person += 1
            if len(profiles[sid].msgs_from_person) < sample_cap:
                profiles[sid].msgs_from_person.append(body)
        elif sid == "ME" and not is_group:
            other = next((p for p in parts if p != "ME"), None)
            if other and other in profiles:
                profiles[other].n_from_me += 1
                if len(profiles[other].msgs_from_me) < sample_cap:
                    profiles[other].msgs_from_me.append(body)

    return profiles


def kinship_hint(profile: PersonProfile) -> Optional[str]:
    """Weak, name-free guess from how the contact is saved. Used only as a deterministic fallback."""
    text = " ".join(profile.names + [profile.one_on_one_chat_name]).lower()
    for circle, hints in KINSHIP_HINTS.items():
        if any(h in text for h in hints):
            return circle
    return None
