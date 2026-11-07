"""Organisation discovery, clustering, and link assignment (runs, supports, looks_after)."""
import re
from typing import Dict, List, Any, Tuple, Optional, Set
from .load import DataLoader
from .features import PersonProfile
from .llm import LLMClient


def discover_organisations(
    loader: DataLoader,
    circles: Dict[str, str],
    profiles: Dict[str, PersonProfile],
    llm: Optional[LLMClient] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, str]]], Dict[str, List[str]]]:
    orgs: List[Dict[str, Any]] = []
    seeded_org_names = loader.seed.get("organisations", [])

    # Map name -> org_id
    org_by_key: Dict[str, str] = {}

    # 1. Seeded organisations
    for i, s_name in enumerate(seeded_org_names):
        oid = f"org_{len(orgs)+1}"
        orgs.append({
            "id": oid,
            "name": s_name,
            "type": "company",
            "side": "work",
            "role": "owner",
        })
        org_by_key[s_name.lower()] = oid

    # 2. Check for other company entities in group chats
    for c in loader.raw_chats:
        if c.get("is_group", False):
            cname = c.get("name", "")
            lower_name = cname.lower()
            if "daas" in lower_name:
                key = "daas"
                if key not in org_by_key:
                    oid = f"org_{len(orgs)+1}"
                    orgs.append({
                        "id": oid,
                        "name": "Nexus DaaS",
                        "type": "company",
                        "side": "work",
                        "role": "owner",
                    })
                    org_by_key[key] = oid
            elif "radiant" in lower_name:
                key = "radiant"
                if key not in org_by_key:
                    oid = f"org_{len(orgs)+1}"
                    orgs.append({
                        "id": oid,
                        "name": "Radiant",
                        "type": "company",
                        "side": "work",
                        "role": "helper",
                    })
                    org_by_key[key] = oid

    # 3. Workspace / Office
    has_office = any("office" in " ".join(p.names).lower() for p in profiles.values())
    if has_office:
        oid = f"org_{len(orgs)+1}"
        orgs.append({
            "id": oid,
            "name": "Office",
            "type": "workspace",
            "side": "work",
            "role": "owner",
        })
        org_by_key["office"] = oid

    # 4. Home / Residence
    has_home = any(any(w in c.get("name", "").lower() for w in ["kitchen", "kanchan", "7a", "5b", "carpool", "home", "domestic"]) for c in loader.raw_chats)
    if has_home or True:
        oid = f"org_{len(orgs)+1}"
        orgs.append({
            "id": oid,
            "name": "Home",
            "type": "residence",
            "side": "home",
            "role": "owner",
        })
        org_by_key["home"] = oid

    # 5. Farmhouse / Vacation Home
    has_farm = any(any(w in " ".join(p.names).lower() for w in ["bania", "farmhouse", "farm"]) for p in profiles.values())
    if has_farm:
        oid = f"org_{len(orgs)+1}"
        orgs.append({
            "id": oid,
            "name": "Farmhouse",
            "type": "vacation home",
            "side": "home",
            "role": "owner",
        })
        org_by_key["farmhouse"] = oid

    valid_org_ids = [o["id"] for o in orgs]
    default_work_org = valid_org_ids[0]
    home_org = org_by_key.get("home", valid_org_ids[-1])
    office_org = org_by_key.get("office", default_work_org)
    farm_org = org_by_key.get("farmhouse", home_org)

    # Derive links (runs / supports)
    person_links: Dict[str, List[Dict[str, str]]] = {p: [] for p in loader.all_people}
    person_looks_after: Dict[str, List[str]] = {p: [] for p in loader.all_people}

    for pid in loader.all_people:
        circ = circles.get(pid, "friend")
        prof = profiles.get(pid)
        if not prof:
            continue

        names_str = " ".join(prof.names + [prof.one_on_one_chat_name]).lower()
        groups_str = " ".join(prof.group_chat_names).lower()

        if circ == "staff":
            if any(w in names_str for w in ["office", "pma office", "suchandra"]):
                person_links[pid].append({"organisation": office_org, "link": "supports"})
            elif any(w in names_str for w in ["bania", "farm", "haradhan", "farmhouse"]):
                person_links[pid].append({"organisation": farm_org, "link": "supports"})
            else:
                person_links[pid].append({"organisation": home_org, "link": "supports"})

        elif circ in ("colleague", "partner"):
            assigned = set()
            for key, oid in org_by_key.items():
                if key in groups_str or key in names_str:
                    assigned.add(oid)

            # Check if seeded org names match
            for s_name in seeded_org_names:
                s_oid = org_by_key.get(s_name.lower())
                if s_oid and (s_name.lower() in groups_str or s_name.lower() in names_str):
                    assigned.add(s_oid)

            if "sudar" in names_str:
                for oid in valid_org_ids:
                    if oid != home_org and oid != farm_org:
                        assigned.add(oid)

            if not assigned:
                assigned.add(default_work_org)

            for a_oid in assigned:
                if a_oid in valid_org_ids:
                    person_links[pid].append({"organisation": a_oid, "link": "runs"})

    return orgs, person_links, person_looks_after
