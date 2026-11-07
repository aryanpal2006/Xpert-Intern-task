"""Evaluation and detailed error analysis wrapper."""
import json
import sys
from pathlib import Path

# Add coldstart to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coldstart.load import DataLoader
from coldstart.validate import validate_output
from check import score, match_orgs, POINTS


def run_evaluation(input_dir: str, output_path: str):
    loader = DataLoader(input_dir)
    chats = loader.raw_chats
    messages = loader.raw_messages
    seed = loader.seed
    answers = loader.answers

    out = json.loads(Path(output_path).read_text(encoding="utf-8"))
    errs = validate_output(out, chats, messages)
    if errs:
        print(f"Validation errors ({len(errs)}):")
        for e in errs[:10]:
            print("  -", e)
        return

    if not answers:
        print("No answers found in", input_dir)
        return

    s = score(out, answers, seed, messages)
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS FOR: {input_dir}")
    print("=" * 50)
    for k, pts in POINTS.items():
        val = s.get(k)
        if val is None:
            print(f"{k:20}: not labelled")
        else:
            print(f"{k:20}: {val:6.2f}  ({val*pts:5.1f} / {pts} pts)")

    # 1. Circle Error Analysis
    print("\n--- CIRCLES ANALYSIS ---")
    seeded_pids = set((seed.get("circles") or {}).keys())
    out_circles = {p["id"]: p.get("circle") for p in out["people"]}
    circle_mistakes = []
    for p in answers["people"]:
        pid = p["id"]
        if pid in seeded_pids:
            continue
        ans_c = p.get("circle")
        pred_c = out_circles.get(pid)
        if ans_c != pred_c:
            names = loader.person_names.get(pid, [])
            circle_mistakes.append((pid, names, ans_c, pred_c))

    print(f"Total unseeded people: {len(answers['people']) - len(seeded_pids)}, Misclassified: {len(circle_mistakes)}")
    for pid, names, ans_c, pred_c in circle_mistakes[:15]:
        print(f"  {pid:4} ({names[:1]}): expected '{ans_c}', got '{pred_c}'")

    # 2. Organisations Analysis
    print("\n--- ORGANISATIONS ANALYSIS ---")
    matched_orgs, n_ours, n_yours = match_orgs(out, answers)
    print(f"Matched orgs ({len(matched_orgs)}/{max(n_ours, n_yours)}):")
    for ans_oid, (pred_oid, score_val) in matched_orgs.items():
        ans_name = next(o["name"] for o in answers["organisations"] if o["id"] == ans_oid)
        pred_name = next(o["name"] for o in out["organisations"] if o["id"] == pred_oid)
        print(f"  Ans '{ans_name}' ({ans_oid}) <-> Pred '{pred_name}' ({pred_oid}): overlap={score_val:.2f}")

    # 3. Loops Analysis
    print("\n--- OPEN LOOPS ANALYSIS ---")
    loops_out = out["open_loops"]
    ans_loops = answers["open_loops"]
    since = answers.get("loops_scored_from")
    if since is not None:
        ts = {m["id"]: m["ts"] for m in messages}
        loops_out = [l for l in loops_out if ts.get(l.get("source_msg"), 0) >= since]
        ans_loops = [a for a in ans_loops if min(ts.get(x, 0) for x in a["source_msgs"]) >= since]

    matched_pairs = []
    used_ans = set()
    for l in loops_out:
        for j, a in enumerate(ans_loops):
            if j not in used_ans and l.get("source_msg") in a["source_msgs"]:
                used_ans.add(j)
                matched_pairs.append((l, a))
                break

    print(f"Predicted loops: {len(loops_out)}, Actual loops: {len(ans_loops)}, Matched: {len(matched_pairs)}")
    if loops_out and ans_loops:
        prec = len(matched_pairs) / len(loops_out)
        rec = len(matched_pairs) / len(ans_loops)
        print(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {2*prec*rec/(prec+rec) if prec+rec else 0:.2f}")

    # Status accuracy on matched
    correct_status = sum(1 for l, a in matched_pairs if l.get("status") == a["status"])
    penalized = sum(1 for l, a in matched_pairs if l.get("status") != a["status"] and a["status"] == "open")
    print(f"Matched loops with right status: {correct_status}/{len(matched_pairs)}")
    print(f"Matched loops penalized (said done when open): {penalized}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_eval.py <input_dir> <output.json>")
        sys.exit(1)
    run_evaluation(sys.argv[1], sys.argv[2])

