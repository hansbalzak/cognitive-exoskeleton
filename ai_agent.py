#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --- Optional sandbox tools (no-crash fallback) ---
try:
    from agent_work.sandbox import read_file, write_file, list_dir, run_shell  # type: ignore
    SANDBOX_AVAILABLE = True
except Exception:
    SANDBOX_AVAILABLE = False

    def read_file(path: str) -> str:
        return "ERROR: sandbox tools not available (missing agent_work/sandbox.py)"

    def write_file(path: str, content: str) -> str:
        return "ERROR: sandbox tools not available (missing agent_work/sandbox.py)"

    def list_dir(path: str) -> str:
        return "ERROR: sandbox tools not available (missing agent_work/sandbox.py)"

    def run_shell(command_name: str, args: List[str]) -> str:
        return "ERROR: sandbox tools not available (missing agent_work/sandbox.py)"


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _ensure_file(path: Path, content: str = "") -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


class SimpleAI:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.model = model

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.repo_root = Path(".").resolve()

        # Files
        self.personality_path = self.repo_root / "personality.txt"
        self.profile_path = self.repo_root / "profile.json"
        self.facts_path = self.repo_root / "facts.jsonl"
        self.conversation_path = self.repo_root / "conversation.json"

        self.ensure_personality_file()
        self.ensure_profile_and_facts()
        self.conversation: List[Dict[str, str]] = self.load_conversation()

        # Repo search index (lazy)
        self._index_built = False
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    # --------- Persistence ---------
    def ensure_personality_file(self) -> None:
        if not self.personality_path.exists():
            self.personality_path.write_text(
                "You are Xero, a friendly chatting coding bot but can also just have friendly conversations.",
                encoding="utf-8",
            )

    def ensure_profile_and_facts(self) -> None:
        if not self.profile_path.exists():
            self.profile_path.write_text(json.dumps({"name": None}, indent=2), encoding="utf-8")
        if not self.facts_path.exists():
            self.facts_path.write_text("", encoding="utf-8")

    def load_conversation(self) -> List[Dict[str, str]]:
        data = _safe_load_json(self.conversation_path, default=[])
        if isinstance(data, list):
            # ensure shape
            cleaned = []
            for m in data:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    cleaned.append({"role": str(m["role"]), "content": str(m["content"])})
            return cleaned
        return []

    def save_conversation(self) -> None:
        self.conversation_path.write_text(json.dumps(self.conversation, indent=2), encoding="utf-8")

    def clear_conversation(self) -> str:
        self.conversation = []
        self.save_conversation()
        return "Conversation cleared."

    def load_profile(self) -> Dict[str, Any]:
        prof = _safe_load_json(self.profile_path, default={"name": None})
        if not isinstance(prof, dict):
            return {"name": None}
        return prof

    def save_profile(self, profile: Dict[str, Any]) -> None:
        self.profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def load_facts(self) -> List[Dict[str, Any]]:
        if not self.facts_path.exists():
            return []
        facts: List[Dict[str, Any]] = []
        for line in self.facts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    facts.append(obj)
            except Exception:
                continue
        return facts

    def save_facts(self, facts: List[Dict[str, Any]]) -> None:
        with self.facts_path.open("w", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    # --------- LLM chat ---------
    def chat(self, user_text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        personality = self.personality_path.read_text(encoding="utf-8", errors="ignore").strip()

        messages: List[Dict[str, str]] = [{"role": "system", "content": personality}]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 400,
            "stream": False,
        }

        r = self.session.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            try:
                return f"HTTP {r.status_code}: {r.json()}"
            except Exception:
                return f"HTTP {r.status_code}: {r.text}"

        data = r.json()
        assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(assistant, str):
            assistant = str(assistant)

        # Update memory
        self.conversation.append({"role": "user", "content": user_text})
        self.conversation.append({"role": "assistant", "content": assistant})
        self.save_conversation()

        # Optional: keep "name" in profile if user says "my name is X"
        m = re.search(r"\bmy name is\s+([A-Za-z0-9 _-]{1,60})\b", user_text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            prof = self.load_profile()
            prof["name"] = name
            self.save_profile(prof)

        return assistant

    # --------- Facts / Profile ---------
    def remember_fact(self, fact: str) -> str:
        facts = self.load_facts()
        new_fact = {
            "id": (max([f.get("id", 0) for f in facts], default=0) + 1),
            "fact": fact,
            "confidence": 1.0,
            "source": "user",
            "timestamp": datetime.now().isoformat(),
        }
        facts.append(new_fact)
        self.save_facts(facts)
        return f"Fact remembered (id={new_fact['id']})."

    def forget_fact(self, fact_id: int) -> str:
        facts = self.load_facts()
        new_facts = [f for f in facts if int(f.get("id", -1)) != fact_id]
        self.save_facts(new_facts)
        return f"Fact forgotten (id={fact_id})."

    def show_profile(self) -> str:
        return json.dumps(self.load_profile(), indent=2)

    # --------- Repo search (lightweight) ---------
    def build_index(self) -> None:
        if self._index_built:
            return
        skip_dirs = {"venv", ".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
        max_bytes = 300_000  # skip huge files

        for root, dirs, files in os.walk(self.repo_root):
            # prune dirs
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fn in files:
                fp = Path(root) / fn
                try:
                    if fp.stat().st_size > max_bytes:
                        continue
                except Exception:
                    continue

                # only text-ish files
                if fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gguf", ".bin"}:
                    continue

                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                rel = str(fp.relative_to(self.repo_root))
                for i, line in enumerate(text.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    # index tokens, not full lines
                    for tok in re.findall(r"[A-Za-z0-9_]{3,}", line):
                        self.index[tok.lower()].append((rel, i))

        self._index_built = True

    def search_repo(self, keyword: str) -> List[Tuple[str, int, str]]:
        self.build_index()
        key = keyword.strip().lower()
        if not key:
            return []

        hits = self.index.get(key, [])
        results: List[Tuple[str, int, str]] = []

        for rel_path, line_no in hits[:50]:
            fp = self.repo_root / rel_path
            try:
                lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            start = max(0, line_no - 3)
            end = min(len(lines), line_no + 2)
            context = "\n".join(lines[start:end]).strip()
            results.append((rel_path, line_no, context))

        return results

    # --------- Commands ---------
    def help(self) -> str:
        return "\n".join(
            [
                "Commands:",
                "  /help                 Show this help",
                "  /exit, /quit          Exit",
                "  /clear                Clear conversation memory",
                "  /profile              Show profile.json",
                "  /remember <fact>      Save a fact to facts.jsonl",
                "  /forget <id>          Remove fact by id",
                "  /askrepo <keyword>    Search repo for keyword (indexes lazily; skips venv/.git)",
                "  /read <path>          Read file via sandbox tool (if available)",
                "  /write <path> <text>  Write file via sandbox tool (if available) [asks approval]",
                "  /list <dir>           List dir via sandbox tool (if available)",
                "  /run <cmd> [args...]  Run allowlisted shell via sandbox tool (if available) [asks approval]",
            ]
        )

    def handle_command(self, command: str) -> str:
        cmd = command.strip()

        if cmd in ("/exit", "/quit"):
            return "__EXIT__"
        if cmd == "/help":
            return self.help()
        if cmd == "/clear":
            return self.clear_conversation()
        if cmd == "/profile":
            return self.show_profile()

        if cmd.startswith("/remember "):
            fact = cmd[len("/remember ") :].strip()
            if not fact:
                return "Usage: /remember <fact>"
            return self.remember_fact(fact)

        if cmd.startswith("/forget "):
            raw = cmd[len("/forget ") :].strip()
            try:
                return self.forget_fact(int(raw))
            except Exception:
                return "Usage: /forget <id>"

        if cmd.startswith("/askrepo "):
            keyword = cmd[len("/askrepo ") :].strip()
            results = self.search_repo(keyword)
            if not results:
                return "No results found."
            out = ["Relevant snippets:"]
            for rel_path, line_no, ctx in results[:10]:
                out.append(f"\nFile: {rel_path}  Line: {line_no}\n{ctx}")
            return "\n".join(out)

        # Sandbox tools
        if cmd.startswith("/read "):
            path = cmd[len("/read ") :].strip()
            return read_file(path)

        if cmd.startswith("/list "):
            path = cmd[len("/list ") :].strip()
            return list_dir(path)

        if cmd.startswith("/write "):
            rest = cmd[len("/write ") :].strip()
            parts = rest.split(" ", 1)
            if len(parts) != 2:
                return "Usage: /write <path> <content>"
            path, content = parts[0], parts[1]
            if not self._approve(f"write_file({path}, <{len(content)} chars>)"):
                return "Canceled."
            return write_file(path, content)

        if cmd.startswith("/run "):
            rest = cmd[len("/run ") :].strip()
            if not rest:
                return "Usage: /run <cmd> [args...]"
            parts = rest.split()
            command_name, args = parts[0], parts[1:]
            if not self._approve(f"run_shell({command_name} {' '.join(args)})"):
                return "Canceled."
            return run_shell(command_name, args)

        return "Unknown command. Type /help."

    def _approve(self, label: str) -> bool:
        # Only ask approval for risky actions; if sandbox not available, no need
        if not SANDBOX_AVAILABLE:
            return True
        ans = input(f"About to run {label}. Proceed? (y/n) ").strip().lower()
        return ans in ("y", "yes")


def main() -> None:
    ai = SimpleAI()
    print("Type /help for commands. /exit to quit.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            response = ai.handle_command(user_input)
            if response == "__EXIT__":
                print("goodbye see you soon!")
                break
            print(f"AI: {response}")
        else:
            response = ai.chat(user_input)
            print(f"AI: {response}")


if __name__ == "__main__":
    main()
