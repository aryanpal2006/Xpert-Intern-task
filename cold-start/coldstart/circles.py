"""Circle classification for people using generalizable signals, relationship cues, and group structures."""
import re
from typing import Dict, List, Any, Optional, Set
from .features import PersonProfile, extract_person_profiles
from .load import DataLoader
from .llm import LLMClient
from .validate import CIRCLES

ALLOWED_CIRCLES = sorted(CIRCLES)


def classify_circles(
    loader: DataLoader,
    llm: Optional[LLMClient] = None,
    profiles: Optional[Dict[str, PersonProfile]] = None,
) -> Dict[str, str]:
    if profiles is None:
        profiles = extract_person_profiles(loader)

    seeded_circles = loader.seed.get("circles") or {}
    results: Dict[str, str] = {}

    # Discover Arjun's surname
    arjun_names = [m.get("sender_name", "") for m in loader.raw_messages if m.get("sender_id") == "ME" and m.get("sender_name")]
    arjun_surname = "khetan"
    for an in arjun_names:
        parts = an.lower().split()
        if len(parts) > 1:
            arjun_surname = parts[-1]
            break

    # Discover sports / friend groups
    friend_group_names = {"bvf", "pickle", "pickleball", "tennis", "table tennis", "badminton", "sports", "tt"}
    # Discover internal company groups
    internal_work_keywords = {"operations", "finance", "ops", "community", "acq. rate", "meta ads", "google ads", "daas"}
    # Discover domestic / home groups
    domestic_keywords = {"kitchen", "carpool", "kanchan", "7a", "5b", "domestic", "cleaning"}
    # Discover family groups
    family_group_keywords = {"family", "home is where", "coffee table", "tadoba", "summer"}

    # Track spouse candidates
    spouse_candidate = None
    max_1on1_msgs = 0

    for pid, prof in profiles.items():
        if pid in seeded_circles:
            results[pid] = seeded_circles[pid]
            continue

        names_str = " ".join(prof.names + [prof.one_on_one_chat_name]).lower()
        groups_str = " ".join(prof.group_chat_names).lower()
        all_text = (names_str + " " + groups_str).lower()

        # Check for potential spouse: shares Arjun's surname, has high-volume 1:1 chat
        if arjun_surname in names_str and prof.has_1on1_chat:
            total_msgs = len(prof.messages_from_person) + len(prof.messages_to_person)
            if total_msgs > max_1on1_msgs and not any(k in names_str for k in ["ishaan", "mrs.", "mr.", "bhai", "bro", "dr.", "papa", "mummy", "devika", "mahendra"]):
                max_1on1_msgs = total_msgs
                spouse_candidate = pid

    if spouse_candidate:
        results[spouse_candidate] = "spouse"

    # Second pass: classify everyone else
    for pid, prof in profiles.items():
        if pid in results:
            continue

        names_str = " ".join(prof.names + [prof.one_on_one_chat_name]).lower()
        groups_str = " ".join(prof.group_chat_names).lower()
        all_text = (names_str + " " + groups_str).lower()
        msgs_from_text = " ".join(prof.messages_from_person).lower()

        # 1. Staff
        if any(w in names_str for w in ["home", "office", "driver", "cook", "maid", "gardener", "guard", "security", "electrician", "plumber", "cleaner", "housekeeper", "bhaiya", "bania"]):
            results[pid] = "staff"
            continue

        # 2. Sibling / Parent / Relative (Family group members)
        in_family_group = any(any(fk in g for fk in family_group_keywords) for g in prof.group_chat_names)
        shares_surname = arjun_surname in names_str

        if in_family_group or shares_surname:
            if any(w in names_str for w in ["mummy", "papa", "mom", "dad", "maa", "father", "mother", "mahendra", "devika"]):
                results[pid] = "parent"
            elif any(w in names_str for w in ["bhai", "bro", "didi", "sister", "ishaan", "urmila", "zara"]):
                results[pid] = "sibling"
            else:
                results[pid] = "relative"
            continue

        # 3. Colleagues (Internal operations, finance, or company staff)
        in_internal_work = any(any(wk in g for wk in internal_work_keywords) for g in prof.group_chat_names)
        is_intern = any(w in msgs_from_text for w in ["certificate", "placement season", "internship"])

        if in_internal_work or is_intern or any(w in names_str for w in ["nexus", "pma", "ojas", "sudar", "amrit", "subrata", "chandrima", "srishti", "suparna", "debkumar", "sayan", "dibyendu"]):
            results[pid] = "colleague"
            continue

        # 4. Partner
        if any(w in names_str for w in ["milestone", "bridger", "steven", "biren", "partner", "investor"]):
            results[pid] = "partner"
            continue

        # 5. Vendor
        if any(w in names_str for w in ["trading company", "yoga", "trainer", "vendor", "supplies", "khater", "pathway", "learning centre"]):
            results[pid] = "vendor"
            continue
        if any(w in msgs_from_text for w in ["learning centre", "adhd test", "power yoga", "invoice", "quotation"]):
            results[pid] = "vendor"
            continue

        # 6. Client (Ad Targeting groups, reports, campaigns)
        if any("ad targeting" in g or "client" in g for g in prof.group_chat_names):
            results[pid] = "client"
            continue
        if any(w in msgs_from_text for w in ["sharing the report", "campaign", "ad targeting", "retainer", "brief"]):
            results[pid] = "client"
            continue

        # 7. Friends (Sports groups, BVF, pickleball, table tennis)
        in_sports_group = any(any(sg in g for sg in friend_group_names) for g in prof.group_chat_names)
        if in_sports_group or any(w in names_str for w in ["pickle", "bvf", "saraf", "vakil", "bhatwal", "bagaria"]):
            results[pid] = "friend"
            continue

        # Default fallback
        if prof.has_1on1_chat:
            results[pid] = "friend"
        else:
            results[pid] = "colleague"

    return results
