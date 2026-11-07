# Cold Start — Write-up

## 1. Focus, and why

The hidden set is **re-anonymised**: every name and id changes. So the one thing that matters is
that nothing keys off a specific name. I treat every chat as structure + content, and let an LLM
read the *content* to decide relationships — never a lookup of known names.

I spent effort where the points and the loss-asymmetry are:

- **Open loops + where to help (38 pts):** the biggest block. `check.py` subtracts a point for
  calling an open loop `done`/`dropped`, and subtracts a point for `assist`/`remind` where the
  answer is `leave`. So status defaults to `open` when unsure, and help defaults to `leave` when
  unsure.
- **Circles / organisations / chats (28 pts):** chats are almost free from structure (1:1 → person,
  group → organisation). Circles and org-clustering are done by the LLM from message content.
- **Goals (8 pts):** precision over recall — every goal must cite a real message id, or it is
  dropped, because an invented goal costs F1.
- **Cost (12 pts):** calls are batched (many chats per prompt), the loop pass only reads the recent
  window that is actually scored, trivial acks are stripped, and every call is disk-cached.

## 2. Approach

Pipeline (`coldstart/`), one command via `run.sh <input_dir> <output.json>`:

1. **load.py / features.py** — index chats, messages, people; build a *name-light* profile per
   person (direct vs group, message volume, a few sample lines each way). No hardcoded names.
2. **circles.py** — the LLM reads each person's profile in batches and picks a circle from the
   fixed vocabulary. Seeded circles are trusted as-is. Fallback: a generic kinship/structure
   heuristic when no model is available.
3. **orgs.py** — the LLM clusters **group chats** into the user's *own* organisations (home,
   companies, workspace, second home, society, school), assigns each group to one, and labels each
   member `runs`/`supports`. A vendor's or client's own company is explicitly **not** an
   organisation.
4. **chats.py** — 1:1 chats tie to the person; group chats tie to their organisation; any leftover
   group is attached by the circle-majority of its members so no chat is ever untied.
5. **loops.py** — only the **last ~18 days** (the scored window is the last two weeks). Recent
   transcripts from several chats are packed into each prompt; the LLM returns each loop's title,
   type, owner, `source_msg`, status, `closed_by`, and — for open loops — the help call. Everything
   is validated against real message ids (a loop's chat is recovered from its `source_msg`, since
   ids are globally unique).
6. **assist.py** — normalises each help object; defaults to `leave` for anything missing, malformed,
   or owned by someone else.

**LLM client (`coldstart/llm.py`)** reads `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`, runs calls
concurrently, disk-caches by prompt hash, logs tokens to `usage.jsonl`, retries with backoff
(honouring `Retry-After`), sends a normal `User-Agent` (some gateways block the default), and — for
reasoning models that choke on strict `json_object` on large outputs — transparently retries without
`response_format` and parses the JSON out of the reply.

## 3. Cost

Measured on the example (6 chats, 29 messages) and the dev set (138 chats, 4,502 messages).

| | model | calls | prompt toks | completion toks |
|---|---|---|---|---|
| Example (complete) | gemini-2.5-flash-lite | ~9 | ~5.3k | ~1.5k |
| Example (complete) | openai/gpt-oss-120b | ~5 | ~3.4k | ~5.0k |
| Dev, ~4.5k msgs (full attempt) | openai/gpt-oss-120b | ~40 | ~45k | ~28k |

So a full dev pass is on the order of ~70–90k tokens for ~4,500 messages. At flash-lite / gpt-oss
pricing (~$0.15–0.60 / 1M tokens) that is well under **$0.05 per 1,000 messages**. Money goes mostly
to the loop pass (recent-window transcripts); circles/orgs/goals are a handful of calls. It is kept
down by the loop window, cross-chat packing, ack-stripping, and disk caching (a repeat run is free).

**Model to run it with:** `google/gemini-2.5-flash-lite` (best accuracy/cost in my tests);
`openai/gpt-oss-120b` also works. Any OpenAI-compatible model is fine — nothing depends on `seed` or
on `json_object`.

> Note on testing: my own Groq/Gemini keys are free-tier, where the token-per-minute cap throttles a
> full 4,502-message dev run (so some calls fall back mid-run). The grader's OpenRouter key is not
> so limited; the design stays within the 30-minute budget there. The example scores below are on a
> fully-completed run.

## 4. Where it's wrong

Example score (fully completed): circles 1.00, orgs 0.80, chats 1.00, goals 1.00, loops-found 0.76,
loops-closed 0.80, help 0.60 — **≈62/74** (gemini-2.5-flash-lite); old overfit code scored 17/74 on
this same unseen example.

Expected mistakes on the hidden set:

1. **Loops — recall ~25–40% miss, status ~15–25% off.** Hinglish/implicit completions ("ho gaya",
   "de diya") and loops that close days later across a chunk boundary. By design it under-reports
   `done` to avoid the −1 penalty, so misses lean toward false-negatives, not false "done".
2. **Where to help ~20–35% off.** The `leave` vs `remind` line is genuinely subjective; I bias to
   `leave` because stepping in wrongly is the only move that costs points.
3. **Organisations — under/over-clustering.** The grader matches by shared chats+people, so merging
   two real ventures or splitting one lowers the score; the fuzzier the group content, the worse.
4. **Circles for peripheral people** seen only inside a mixed group with almost no messages — little
   signal, so these fall back to a structural guess.
5. **Generalisation:** there are **no** hardcoded names, ids, places, or amounts anywhere; all
   signal is message content, chat structure, and relative timestamps.
