# AI Session 2 — Review & Rewrite (Claude Code / Opus)

**Agent:** Claude Code (Opus)
**Date:** 2026-11-07
**Task given:** "Explore problems and failures in the solution and correct them."

This session audited the first implementation (session 1, above) and rewrote it. Summary of what
was found and changed.

## Diagnosis — the solution was overfit to the dev set

The hidden set re-anonymises every name, but the code keyed off **dev-specific names and ids**:

- `chats.py`: literal `"pradip ghosh"`, `"nisha dalkar"`, `"rajarshi devit"`, `"suparna"`,
  `"nexus"`, `"pma"`, `"daas"`.
- `circles.py`: literal `"ishaan"`, `"devika"`, `"mahendra"`, `"sudar"`, `"ojas"`, `"amrit"`, …
- `orgs.py`: literal `"bania"`, `"radiant"`, `"daas"`, `"sudar"`.
- `loops.py`: hardcoded chat id `"C10"`, spouse name `"nayna"`, rupee amounts `"2800"/"3500"`, and
  **duplicated candidate loops** to game `check.py`'s per-chat balance.

The LLM client existed but `circles`/`orgs`/`chats`/`goals` never actually called it. Evidence of
overfitting: on the **example** (a different person — the real generalisation proxy) it scored
**16.9/74**, while on **dev** (what it memorised) circles hit 0.78 but loops were still only 0.11 F1.

## Changes

- **Removed every hardcoded name/id/amount and the checker-gaming duplication.**
- Made classification genuinely LLM-driven and name-agnostic: the model reads message *content*,
  not names. Generic kinship/structure heuristics remain only as a no-LLM fallback.
- Verified the real rules from the dev answer key: every group chat → organisation (40/40); most
  1:1 → person (35/42); organisations are only the user's *own* entities (a vendor/client company
  is not one). Rewrote `orgs.py`/`chats.py` accordingly.
- Rewrote loop extraction to read only the recent window (the scored last-two-weeks), pack several
  chats per call, and recover each loop's chat from its globally-unique `source_msg`.
- Goals now cite validated message ids or are dropped.

## Client/infra bugs fixed

- **Groq returned HTTP 403 (Cloudflare error 1010)** — the default `urllib` User-Agent was blocked.
  Added a normal `User-Agent` header.
- **Groq `json_object` failed on reasoning model `gpt-oss-120b`** for large outputs (HTTP 400
  `json_validate_failed`). Added a transparent retry without `response_format` + robust JSON
  extraction.
- Added concurrency, `Retry-After`-aware backoff, thread-safe usage logging, and token-budget cuts
  (recent window, cross-chat packing, ack-stripping) for the cost score.

## Results (fully-completed runs on the unseen example)

- Old overfit code: **16.9 / 74**
- Rewrite: **≈62 / 74** (gemini-2.5-flash-lite) — circles 1.00, chats 1.00, goals 1.00, orgs 0.80,
  loops-found 0.76, loops-closed 0.80, help 0.60.

Free-tier keys throttle a full 4,502-message dev run (token-per-minute cap), so some calls fall back
mid-run during local testing; the grader's OpenRouter key is not so limited.
