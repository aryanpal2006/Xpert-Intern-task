"""Feature extraction per person and per chat for circle, organisation, and loop analysis."""
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set
from .load import DataLoader

# Kinship and relationship token patterns
KINSHIP_PATTERNS = {
    "staff": re.compile(r"\b(driver|cook|maid|bhaiya|gardener|guard|security|electrician|plumber|cleaner|housekeeper|delivery|carpenter|attendance|salary|kitchen|ration|chhutti|leave|gatekeeper|sweeper|car\s*wash|mechanic)\b", re.I),
    "spouse": re.compile(r"\b(wife|husband|nayna|biwi|jaan|darling|babe|sweetheart|dinner|carpool|groceries|preschool|kids|school|home\s*is\s*where)\b", re.I),
    "parent": re.compile(r"\b(mummy|papa|mom|dad|maa|father|mother|sasur|saas|blessings|pranam|namaste)\b", re.I),
    "child": re.compile(r"\b(beta|beti|son|daughter|tuition|homework|preschool|daycare|kindergarten)\b", re.I),
    "sibling": re.compile(r"\b(bhai|bro|didi|sister|brother|bhiya|rakhi)\b", re.I),
    "relative": re.compile(r"\b(chacha|chachi|mama|mami|bua|fufa|cousin|mausi|bhabhi|jiju|family|family\s*tree|gathering)\b", re.I),
    "colleague": re.compile(r"\b(colleague|team|sprint|deploy|standup|ops|operations|marketing|leads|campaign|jira|github|pr|repo|bug|review|dashboard|ad\s*targeting|roas|cpc|meta\s*ads|google\s*ads|pma|nexus|daas)\b", re.I),
    "client": re.compile(r"\b(client|account|brief|deliverable|scope|retainer|ad\s*targeting|onboarding|agency|creative|assets|campaign\s*report)\b", re.I),
    "vendor": re.compile(r"\b(vendor|supplier|invoice|quotation|estimate|gst|bill|payment|rates|trading\s*company|trainer|yoga|gym\s*trainer|printing|catering)\b", re.I),
    "partner": re.compile(r"\b(partner|co-founder|equity|investor|shares|board|round|valuation|seed|term\s*sheet|angel)\b", re.I),
    "friend": re.compile(r"\b(bro|dude|yaar|bhai|plans|party|beer|drinks|dinner|table\s*tennis|badminton|pickleball|sports|tt|trip|vacation|game|fifa|movie)\b", re.I),
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
        self.messages_from_person: List[str] = []
        self.messages_to_person: List[str] = []
        self.keyword_scores: Dict[str, int] = defaultdict(int)
        self.seeded_circle: str = ""

    def summary(self) -> str:
        names_str = ", ".join(self.names[:3])
        chats_str = ", ".join(self.group_chat_names[:4])
        sample_in = " | ".join(self.messages_from_person[:4])
        sample_out = " | ".join(self.messages_to_person[:4])
        return (
            f"Person {self.id} (Names: {names_str}):\n"
            f"  1:1 Chat: {self.one_on_one_chat_name if self.has_1on1_chat else 'None'}\n"
            f"  Group Chats: {chats_str}\n"
            f"  From Person ({len(self.messages_from_person)} msgs): {sample_in[:200]}\n"
            f"  From ME ({len(self.messages_to_person)} msgs): {sample_out[:200]}\n"
        )


def extract_person_profiles(loader: DataLoader) -> Dict[str, PersonProfile]:
    profiles: Dict[str, PersonProfile] = {p: PersonProfile(p) for p in loader.all_people}

    # Load seeded circles
    seeded_circles = loader.seed.get("circles") or {}
    for pid, circ in seeded_circles.items():
        if pid in profiles:
            profiles[pid].seeded_circle = circ

    # Chat associations
    for c in loader.raw_chats:
        cid = c["chat_id"]
        cname = c.get("name", "")
        parts = c.get("participants", [])
        is_group = c.get("is_group", False)

        for p in parts:
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

    # Add names from loader
    for pid, names in loader.person_names.items():
        if pid in profiles:
            profiles[pid].names = names

    # Message analysis
    for m in loader.raw_messages:
        sid = m.get("sender_id")
        body = m.get("body", "") or ""
        cid = m.get("chat_id")
        chat = loader.chats_by_id.get(cid, {})
        is_group = chat.get("is_group", False)
        parts = chat.get("participants", [])

        if sid in profiles and len(profiles[sid].messages_from_person) < 25:
            if body.strip():
                profiles[sid].messages_from_person.append(body.strip())
        elif sid == "ME" and not is_group:
            # 1:1 message from ME to the other participant
            other = next((p for p in parts if p != "ME"), None)
            if other and other in profiles and len(profiles[other].messages_to_person) < 25:
                if body.strip():
                    profiles[other].messages_to_person.append(body.strip())

    # Calculate keyword scores
    for pid, prof in profiles.items():
        combined_text = " ".join(
            prof.names + prof.group_chat_names + [prof.one_on_one_chat_name] +
            prof.messages_from_person + prof.messages_to_person
        )
        for cat, pat in KINSHIP_PATTERNS.items():
            matches = len(pat.findall(combined_text))
            if matches > 0:
                prof.keyword_scores[cat] += matches

    return profiles

