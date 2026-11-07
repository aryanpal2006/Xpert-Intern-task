"""Data loading and preprocessing utilities for Cold Start challenge."""
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file with UTF-8 encoding."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [json.loads(line) for line in f if line.strip()]


class DataLoader:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.chats_path = self.input_dir / "chats.jsonl"
        self.messages_path = self.input_dir / "messages.jsonl"
        self.seed_path = self.input_dir / "seed.json"
        self.answers_path = self.input_dir / "answers" / "answers.json"

        # Raw data
        self.raw_chats: List[Dict[str, Any]] = read_jsonl(self.chats_path)
        self.raw_messages: List[Dict[str, Any]] = read_jsonl(self.messages_path)
        
        self.seed: Dict[str, Any] = {}
        if self.seed_path.exists():
            try:
                self.seed = json.loads(self.seed_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                self.seed = {}

        self.answers: Optional[Dict[str, Any]] = None
        if self.answers_path.exists():
            try:
                self.answers = json.loads(self.answers_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                self.answers = None

        # Build indexes
        self.chats_by_id: Dict[str, Dict[str, Any]] = {c["chat_id"]: c for c in self.raw_chats}
        self.messages_by_id: Dict[str, Dict[str, Any]] = {m["id"]: m for m in self.raw_messages}

        # Index messages by chat_id, sorted by timestamp
        self.messages_by_chat: Dict[str, List[Dict[str, Any]]] = {}
        for m in sorted(self.raw_messages, key=lambda x: (x.get("ts", 0), x.get("id", ""))):
            cid = m.get("chat_id")
            if cid:
                if cid not in self.messages_by_chat:
                    self.messages_by_chat[cid] = []
                self.messages_by_chat[cid].append(m)

        # Discover all persons (excluding ME)
        self.all_people = sorted(
            {p for c in self.raw_chats for p in c.get("participants", []) if p != "ME"}
        )

        # Map person to their known sender names
        self.person_names: Dict[str, List[str]] = {p: [] for p in self.all_people}
        for m in self.raw_messages:
            sid = m.get("sender_id")
            sname = m.get("sender_name")
            if sid and sid in self.person_names and sname:
                if sname not in self.person_names[sid]:
                    self.person_names[sid].append(sname)

        # If a person has a 1:1 chat, chat name is often their saved name
        for c in self.raw_chats:
            if not c.get("is_group", False):
                parts = [p for p in c.get("participants", []) if p != "ME"]
                if parts:
                    pid = parts[0]
                    cname = c.get("name")
                    if cname and pid in self.person_names:
                        if cname not in self.person_names[pid]:
                            self.person_names[pid].insert(0, cname)

