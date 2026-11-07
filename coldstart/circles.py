"""Circle classification: who each person is to the user.

LLM reads each person's actual messages and relationship context and picks a circle from the fixed
vocabulary. No personal names, places, or dev-set specifics are hardcoded, so it generalises to the
re-anonymised hidden set. A name-free structural heuristic is the fallback when no LLM is available.
"""
from typing import Dict, List, Optional
from .features import PersonProfile, extract_person_profiles, kinship_hint
from .load import DataLoader
from .llm import LLMClient
from .validate import CIRCLES

ALLOWED = sorted(CIRCLES)

_DEFS = (
    "spouse: husband/wife/partner they live with. "
    "child: their son/daughter. parent: their mother/father (or in-law parent -> in-law). "
    "in-law: a relative by marriage. sibling: their brother/sister. "
    "relative: any other family by blood or marriage. "
    "friend: a personal friend (sports, hanging out, non-work). "
    "colleague: works at a company the user runs or is part of. "
    "client: the user's company sells services to them. "
    "vendor: a supplier/service the user pays (shop, trainer, printer, contractor). "
    "partner: a co-founder/business partner/investor the user runs a venture with. "
    "staff: domestic or personal help (driver, cook, maid, guard, office help)."
)

_SYS = (
    "You are a sharp personal assistant reading a new principal's WhatsApp. For each person, decide "
    "their relationship to the user (who is 'ME'). Choose exactly one label from this list: "
    + ", ".join(ALLOWED) + ".\n" + _DEFS +
    "\nJudge from what they talk about and how, not from names (names are anonymised). "
    "Return ONLY a JSON object mapping each person_id to its label."
)


def _batch(items: List, size: int) -> List[List]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _llm_classify(llm: LLMClient, profiles: List[PersonProfile], batch_size: int = 18) -> Dict[str, str]:
    results: Dict[str, str] = {}
    payloads = []
    batches = _batch(profiles, batch_size)
    for batch in batches:
        blocks = "\n\n".join(p.compact() for p in batch)
        user = (
            "Classify each person below. Return JSON like "
            '{"P01":"spouse","P02":"colleague"}.\n\n' + blocks
        )
        payloads.append([
            {"role": "system", "content": _SYS},
            {"role": "user", "content": user},
        ])

    responses = llm.map_json(payloads, max_tokens=800)
    for batch, resp in zip(batches, responses):
        if not isinstance(resp, dict):
            continue
        for p in batch:
            val = resp.get(p.id)
            if isinstance(val, str) and val.strip().lower() in CIRCLES:
                results[p.id] = val.strip().lower()
    return results


def _fallback(profile: PersonProfile) -> str:
    hint = kinship_hint(profile)
    if hint:
        return hint
    # Structural guesses with no names:
    if profile.has_1on1_chat and not profile.group_chat_names:
        # A private 1:1 with no shared groups: personal contact.
        return "friend"
    if profile.group_chat_names and not profile.has_1on1_chat:
        # Only seen inside groups: most often a work contact.
        return "colleague"
    return "friend"


def classify_circles(
    loader: DataLoader,
    llm: Optional[LLMClient] = None,
    profiles: Optional[Dict[str, PersonProfile]] = None,
) -> Dict[str, str]:
    if profiles is None:
        profiles = extract_person_profiles(loader)

    seeded = loader.seed.get("circles") or {}
    results: Dict[str, str] = {}

    # Seeded circles are ground truth given at signup — always trust them.
    to_classify: List[PersonProfile] = []
    for pid, prof in profiles.items():
        if pid in seeded and seeded[pid] in CIRCLES:
            results[pid] = seeded[pid]
        else:
            to_classify.append(prof)

    if llm and llm.is_available and to_classify:
        results.update(_llm_classify(llm, to_classify))

    # Fill anything the LLM skipped or that ran without a model.
    for prof in to_classify:
        if prof.id not in results:
            results[prof.id] = _fallback(prof)

    return results
