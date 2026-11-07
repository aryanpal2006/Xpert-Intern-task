"""Main pipeline entrypoint for Cold Start challenge."""
import argparse
import json
import sys
import time
from pathlib import Path

from .load import DataLoader
from .features import extract_person_profiles
from .circles import classify_circles
from .orgs import discover_organisations
from .chats import assign_chats
from .loops import extract_open_loops
from .goals import extract_goals
from .validate import validate_output
from .llm import LLMClient


def build_solution(input_dir: str, no_llm: bool = False) -> dict:
    started_time = time.time()
    loader = DataLoader(input_dir)
    llm = LLMClient(no_llm=no_llm)

    # 1. Feature extraction
    profiles = extract_person_profiles(loader)

    # 2. Circle classification
    circles = classify_circles(loader, llm=llm, profiles=profiles)

    # 3. Organisations and links
    organisations, person_links, person_looks_after = discover_organisations(
        loader, circles=circles, profiles=profiles, llm=llm
    )

    # 4. Chat assignments
    chats = assign_chats(loader, organisations=organisations, circles=circles, profiles=profiles)

    # 5. Open loops and help
    open_loops = extract_open_loops(loader, circles=circles, profiles=profiles, llm=llm)

    # 6. Goals
    goals = extract_goals(loader, llm=llm)

    # Build people array
    people_out = []
    for pid in loader.all_people:
        p_obj = {
            "id": pid,
            "circle": circles.get(pid, "friend"),
        }
        links = person_links.get(pid, [])
        if links:
            p_obj["links"] = links
        looks = person_looks_after.get(pid, [])
        if looks:
            p_obj["looks_after"] = looks
        people_out.append(p_obj)

    out = {
        "people": people_out,
        "organisations": organisations,
        "chats": chats,
        "goals": goals,
        "open_loops": open_loops,
    }

    # Validate schema
    errs = validate_output(out, loader.raw_chats, loader.raw_messages)
    if errs:
        print(f"Warning: Output validation had {len(errs)} issue(s):", file=sys.stderr)
        for e in errs[:10]:
            print(f"  - {e}", file=sys.stderr)

    return out


def main():
    parser = argparse.ArgumentParser(description="Cold Start Challenge Solution Pipeline")
    parser.add_argument("input_dir", help="Path to input directory containing chats.jsonl, messages.jsonl, seed.json")
    parser.add_argument("output_path", help="Path to write output.json")
    parser.add_argument("--no-llm", action="store_true", help="Run deterministic heuristic pipeline without LLM API")
    args = parser.parse_args()

    started = time.time()
    out = build_solution(args.input_dir, no_llm=args.no_llm)
    
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    
    elapsed = round(time.time() - started, 2)
    print(f"Solution generated in {elapsed}s and saved to {args.output_path}")


if __name__ == "__main__":
    main()

