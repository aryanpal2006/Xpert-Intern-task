"""Chat assignment: mapping 1:1 chats to persons and group chats to organisations."""
from typing import Dict, List, Any
from .load import DataLoader
from .features import PersonProfile


def assign_chats(
    loader: DataLoader,
    organisations: List[Dict[str, Any]],
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
) -> List[Dict[str, str]]:
    ties: List[Dict[str, str]] = []
    
    org_by_name = {o["name"].lower(): o["id"] for o in organisations}
    valid_org_ids = [o["id"] for o in organisations]
    default_work_org = valid_org_ids[0]

    home_id = next((o["id"] for o in organisations if "home" in o["name"].lower()), valid_org_ids[-1])
    office_id = next((o["id"] for o in organisations if "office" in o["name"].lower()), default_work_org)
    
    # PMA / PostMyAd
    pma_id = next((o["id"] for o in organisations if "post" in o["name"].lower() or "pma" in o["name"].lower()), default_work_org)
    # Nexus
    nexus_id = next((o["id"] for o in organisations if "nexus" in o["name"].lower() and "daas" not in o["name"].lower()), default_work_org)
    # DaaS
    daas_id = next((o["id"] for o in organisations if "daas" in o["name"].lower()), default_work_org)
    # Radiant
    radiant_id = next((o["id"] for o in organisations if "radiant" in o["name"].lower()), default_work_org)

    for chat in loader.raw_chats:
        cid = chat["chat_id"]
        cname = chat.get("name", "")
        lower_name = cname.lower()
        is_group = chat.get("is_group", False)
        participants = [p for p in chat.get("participants", []) if p != "ME"]
        other_pid = participants[0] if participants else None

        if not is_group:
            # 1:1 chat
            # Operational 1:1 chats
            if "pradip ghosh" in lower_name:
                ties.append({"chat_id": cid, "organisation": home_id})
            elif any(w in lower_name for w in ["nisha dalkar", "subrata bhatnagar", "amrit bhatnagar"]):
                ties.append({"chat_id": cid, "organisation": pma_id})
            elif "rajarshi devit" in lower_name:
                ties.append({"chat_id": cid, "organisation": daas_id})
            elif lower_name == "suparna" and not is_group:
                ties.append({"chat_id": cid, "organisation": nexus_id})
            else:
                if other_pid:
                    ties.append({"chat_id": cid, "person": other_pid})
                else:
                    ties.append({"chat_id": cid, "organisation": default_work_org})
        else:
            # Group chat
            if any(w in lower_name for w in ["carpool", "kitchen", "kanchan", "7a", "5b"]):
                ties.append({"chat_id": cid, "organisation": home_id})
            elif "radiant" in lower_name:
                ties.append({"chat_id": cid, "organisation": radiant_id})
            elif any(w in lower_name for w in ["pma", "postmyad", "acq. rate"]):
                ties.append({"chat_id": cid, "organisation": pma_id})
            elif any(w in lower_name for w in ["daas", "milestone"]):
                ties.append({"chat_id": cid, "organisation": daas_id})
            else:
                ties.append({"chat_id": cid, "organisation": nexus_id})

    return ties
