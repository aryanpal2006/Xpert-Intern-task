# Cold Start Solution Write-up

## 1. What I Focused On, and Why

The solution is architected strictly around the challenge's scoring weights and loss asymmetry:
- **Open Loops & Where to Help (38 points total: 28 for loops + 10 for help):** This is the decisive scoring block. In `check.py`, calling an open loop "done" or "dropped" when it is actually open penalizes by subtracting 1 point, and saying `assist` or `remind` on a loop that should be `leave` also penalizes by subtracting 1 point. I focused heavily on high-precision candidate extraction, asymmetric conservative status resolution (defaulting to `open`), and strict `leave`-first assistance rubrics.
- **Circles, Organisations, and Chats (28 points combined: 12 circles, 8 orgs, 8 chats):** Deterministic relational cues, group membership graph topology, and kinship linguistic markers provide massive point yield at near-zero token cost. Tying 1:1 chats to their respective counterparts and group chats to parent organisations achieves near-perfect chat accuracy (~0.98 on dev) and strong organisation Jaccard overlap.
- **Cost (12 points) and Runtime (7 points):** Designed from day one to execute in under one second on standard Python 3.9+ without third-party dependencies, featuring on-disk caching, graceful `--no-llm` deterministic fallbacks, and multi-tier LLM integration.
- **Goals (8 points):** Tuned for strict precision over recall. Because hallucinated or unbacked goals penalize the F1 score, the system only outputs goals anchored by verified message IDs.

---

## 2. My Approach

The pipeline executes in six modular stages:

```
                  ┌──────────────────────┐
                  │ Raw Chats & Messages │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Feature Extraction  │ (Profiles, message counts, group overlaps)
                  └──────────┬───────────┘
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼─────────┐┌────────▼─────────┐┌────────▼─────────┐
│ Circle Classifier││   Org Clustering ││  Goal Extractor  │
│ (Kinship, Staff, ││ (Seed, Home, Farm││ (Health, Travel, │
│  Colleagues, etc)││  Derive Runs/Sup)││  Work Milestones)│
└────────┬─────────┘└────────┬─────────┘└────────┬─────────┘
         │                   │                   │
         └─────────┬─────────┴───────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Chat Tie Engine  │ (1:1 -> Person/Org, Groups -> Org)
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Open Loops Engine  │ (Spouse requests, commitments, relative dates)
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   Where to Help    │ (Strict leave-first rubric, assist, remind)
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │ Validated output   │
         └────────────────────┘
```

1. **Loader & Profiler (`load.py`, `features.py`):** Ingests JSONL data, normalizes timestamps into IST (+05:30), indexes bidirectional communication, and extracts kinship, internal operations, client, and staff tokens.
2. **Circle Classification (`circles.py`):** Identifies Arjun's household surname and single spouse via interaction volume, isolates domestic/office staff via operational keywords (`driver`, `maid`, `office boy`), separates internal colleagues running operations from external clients, and clusters friends via sports/recreational clubs (pickleball, table tennis).
3. **Organisations & Links (`orgs.py`):** Anchors on `seed.json` companies (`PostMyAd`, `NEXUS`), discovers internal ventures (`Nexus DaaS`, `Radiant`), workspaces (`Office`), and domestic residences (`Home`, `Farmhouse`). Automatically derives `runs` for colleagues/partners and `supports` for staff.
4. **Chat Ties (`chats.py`):** Maps 1:1 personal chats to participants and operational 1:1s to respective orgs; clusters group chats to their corresponding parent organisation.
5. **Open Loops & Status (`loops.py`):** Scans for explicit ME commitments ("I will", "bhejta hu"), spouse action items, financial disbursements, and deadline markers. Resolves relative dates against message timestamps. Applies conservative status resolution: only marks `done` when an unambiguous confirmation message appears, completely avoiding false-completion penalties.
6. **Where to Help (`assist.py`):** Categorizes open loops using a weighted cost rubric: defaults to `leave` on spouse/family personal matters and items owned by others; assigns `assist` with `needs_yes` for research and bookings; assigns `remind` for executive decisions and meetings.

---

## 3. What It Costs

| Mode | Model | Prompt Tokens | Completion Tokens | Total Cost / 1k msgs | Dominant Stage |
|---|---|---|---|---|---|
| **Deterministic (`--no-llm`)** | None | 0 | 0 | **$0.00** | Plain Python (< 0.5s) |
| **LLM Tier (Default)** | `gemini-2.5-flash-lite` | ~18k | ~2.5k | **~$0.005** | Ambiguous Circle & Loop boundary calls |
| **LLM Tier (Strong)** | `gemini-2.5-flash` | ~25k | ~3k | **~$0.02** | Goal and multi-turn open loop extraction |

The pipeline disk-caches every API call in `.cache_llm/` hashed by prompt and parameters, meaning subsequent runs cost $0.00. Token spend is 85% concentrated on open loop boundary validation and 15% on ambiguous group-only circle classification.

---

## 4. Where It's Wrong & Calibration

Based on dev-set error analysis (4,502 messages, 138 chats), here are the expected error modes and predicted performance shifts on the hidden set:

1. **Circle Classification Errors (~15–22% expected error rate):**
   - *Failure mode:* Group-only participants with zero direct 1:1 messages lack conversational signals. In dev, relatives sharing group chats with colleagues or sports friends without common surnames can be misclassified.
   - *Hidden set prediction:* Will slightly degrade if new family members or peripheral vendors lack typical Indian kinship/honorific cues.
2. **Open Loops Status Asymmetry (~15–20% precision / ~25–35% recall):**
   - *Failure mode:* Multilingual Hinglish idioms ("de do", "dekh liyo", "ho gaya samjho") and implicit conversational completions that span multiple days without an explicit "done".
   - *Calibration choice:* By design, the engine intentionally prefers false negatives over false "done" calls because `check.py` subtracts 1 full point for marking an open loop as done. This sacrifice protects the total score.
3. **Where to Help Errors (~20–30% error rate):**
   - *Failure mode:* Boundary between `leave` and `remind`. Arjun occasionally appreciates reminders for domestic appointments that look like "leave" territory.
   - *Calibration choice:* Uncertain loops are routed to `leave`, as false `assist`/`remind` calls on `leave` ground truths carry a -1 penalty.
4. **Hidden Set Name Re-anonymization:**
   - The solution uses zero hardcoded personal names or static IDs. All features derive dynamically from lexical patterns, graph topology, and relative timestamps, ensuring robust generalization to the hidden 6-week evaluation set.
