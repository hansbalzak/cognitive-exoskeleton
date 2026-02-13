#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

handler = RotatingFileHandler('logs/agent.log', maxBytes=1024*1024*5, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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


def _atomic_write_json(path: Path, data: Any) -> None:
    temp_path = path.with_suffix('.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    temp_path.rename(path)


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_suffix('.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        f.write(content)
    temp_path.rename(path)


class SimpleAI:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8080/v1"
        self.model = "Qwen2.5-7B-Instruct"
        self.temperature = 0.7
        self.max_tokens = 100

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
        self.identity_path = self.repo_root / "identity.json"
        self.self_reflection_log_path = self.repo_root / "self_reflection.log"
        self.lockfile_path = self.repo_root / "agent.lock"

        # Check for lockfile
        if self.lockfile_path.exists():
            print("Agent is already running. Exiting.")
            sys.exit(1)

        # Create lockfile
        self.lockfile_path.touch()

        self.ensure_personality_file()
        self.ensure_profile_and_facts()
        self.conversation: List[Dict[str, str]] = self.load_conversation()
        self.identity = self.load_identity()

        # Repo search index (lazy)
        self._index_built = False
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Hardware acceleration setup
        self._setup_hardware_acceleration()

        # Message counter
        self.user_message_count = 0

        # Circuit breaker
        self.consecutive_failures = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_end_time = 0

    # --------- Persistence ---------
    def ensure_personality_file(self) -> None:
        if not self.personality_path.exists():
            self.personality_path.write_text(
                "You are Xero, a friendly and approachable AI assistant. Your primary function is to assist with coding tasks, answer questions, and engage in friendly conversations. Here are some guidelines to ensure a positive and productive interaction:\n\n1. **Respect Boundaries**: Do not engage in discussions about sensitive topics such as politics, religion, or personal information unless explicitly asked.\n2. **Stay On Topic**: Focus on the task at hand and avoid drifting into unrelated topics.\n3. **Be Polite and Professional**: Always use respectful and professional language.\n4. **Provide Clear and Concise Answers**: Aim to be as clear and concise as possible in your responses.\n5. **Encourage Learning**: Feel free to suggest resources or further reading if you think it will help.\n\nRemember, your goal is to assist and provide value to the user. Let's get started!",
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
        _atomic_write_json(self.conversation_path, self.conversation)

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
        _atomic_write_json(self.profile_path, profile)

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

    def load_identity(self) -> Dict[str, Any]:
        identity = _safe_load_json(self.identity_path, default={
            "style": "friendly and approachable",
            "values": ["respect boundaries", "stay on topic", "be polite and professional", "provide clear and concise answers", "encourage learning"],
            "preferences": {
                "language": "English",
                "timezone": "UTC"
            }
        })
        if not isinstance(identity, dict):
            identity = {
                "style": "friendly and approachable",
                "values": ["respect boundaries", "stay on topic", "be polite and professional", "provide clear and concise answers", "encourage learning"],
                "preferences": {
                    "language": "English",
                   , "timezone": "UTC"
                }
            }
        return identity

    def save_identity(self, identity: Dict[str, Any]) -> None:
        _atomic_write_json(self.identity_path, identity)

    def validate_identity(self, identity: Dict[str, Any]) -> bool:
        return isinstance(identity, dict) and "style" in identity and "values" in identity and "preferences" in identity

    # --------- LLM chat ---------
    def _post_chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int, stream: bool = False) -> Tuple[bool, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=(10, 120))
            if response.status_code == 200:
                return True, response.json()
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return False, response.text
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return False, str(e)

    def chat(self, user_text: str, stream: bool = False) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        personality = self.personality_path.read_text(encoding="utf-8", errors="ignore").strip()
        identity = json.dumps(self.identity, indent=2)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": f"{personality}\n\nIdentity:\n{identity}"}
        ]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        if stream:
            ok, response = self._post_chat(messages, self.temperature, self.max_tokens, stream=True)
            if not ok:
                return response

            assistant = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                assistant += content
                                sys.stdout.write(content)
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        continue
            # Update conversation after stream completes
            self.conversation.append({"role": "user", "content": user_text})
            self.conversation.append({"role": "assistant", "content": assistant})
            self.trim_conversation()
            self.save_conversation()
            return assistant
        else:
            ok, response = self._post_chat(messages, self.temperature, self.max_tokens)
            if not ok:
                # Retry once after 1s on network failure
                time.sleep(1)
                ok, response = self._post_chat(messages, self.temperature, self.max_tokens)
                if not ok:
                    return response

            data = response
            assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(assistant, str):
                assistant = str(assistant)

            # Update memory
            self.conversation.append({"role": "user", "content": user_text})
            self.conversation.append({"role": "assistant", "content": assistant})
            self.trim_conversation()
            self.save_conversation()

            # Optional: keep "name" in profile if user says "my name is X"
            m = re.search(r"\bmy name is\s+([A-Za-z0-9 _-]{1,60})\b", user_text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                prof = self.load_profile()
                prof["name"] = name
                self.save_profile(prof)

            # Update identity every 20 user messages
            self.user_message_count += 1
            if self.user_message_count >= 20:
                self.update_identity_from_conversation()
                self.user_message_count = 0

            return assistant

    # --------- Conversation Trimming ---------
    def trim_conversation(self) -> None:
        if len(self.conversation) > 40:
            summary = self.summarize_conversation()
            self.conversation = [
                {"role": "system", "content": summary},
                *self.conversation[-10:]
            ]
            self.save_identity(self.identity)

    def summarize_conversation(self) -> str:
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active. Skipping summarize_conversation.")
            return ""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Summarize the following conversation into a compact summary with the following sections:\n\n1. User goals\n2. Important decisions\n3. Open tasks/questions\n4. Stable user prefs\n\nEnsure the summary stays under ~800 characters."},
            {"role": "user", "content": json.dumps(self.conversation)}
        ]

        ok, response = self._post_chat(messages, 0.0, 200, stream=False)
        if not ok:
            logger.error(f"Failed to summarize conversation: {response}")
            return ""

        data = response
        summary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(summary, str):
            summary = str(summary)

        return summary

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
                "  /summarize            Summarize the conversation and keep the last 10 messages",
                "  /decay_facts          Decay confidence of facts and remove low-confidence ones",
                "  /health               Check agent health",
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
        if not SANDBOX_AVAILABLE:
            return "ERROR: Sandbox tools are not available."

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
            if command_name not in {"ls", "cat", "git", "python3", "pip", "grep"}:
                return "ERROR: Command not allowed."
            if not self._approve(f"run_shell({command_name} {' '.join(args)})"):
                return "Canceled."
            return run_shell(command_name, args)

        if cmd == "/summarize":
            summary = self.summarize_conversation()
            self.conversation = [
                {"role": "system", "content": summary},
                *self.conversation[-10:]
            ]
            self.save_conversation()
            return "Conversation summarized and updated."

        if cmd == "/decay_facts":
            return self.decay_facts()

        if cmd == "/health":
            return self.health_check()

        return "Unknown command. Type /help."

    def _approve(self, label: str) -> bool:
        # Only ask approval for risky actions; if sandbox not available, no need
        if not SANDBOX_AVAILABLE:
            return True
        ans = input(f"About to run {label}. Proceed? (y/n) ").strip().lower()
        return ans in ("y", "yes")

    def _setup_hardware_acceleration(self) -> None:
        # Placeholder for hardware acceleration setup
        # This can be expanded based on specific hardware acceleration needs
        print("Hardware acceleration setup complete.")

    def decay_facts(self) -> str:
        facts = self.load_facts()
        updated_facts = []

        for fact in facts:
            timestamp = fact.get("timestamp", "")
            try:
                fact_date = datetime.fromisoformat(timestamp)
                age_days = (datetime.now() - fact_date).days
                fact["confidence"] = max(0.0, fact["confidence"] - (age_days * 0.01))
            except (ValueError, TypeError):
                fact["confidence"] = max(0.0, fact["confidence"] - 0.1)

            if fact["confidence"] >= 0.2:
                updated_facts.append(fact)

        self.save_facts(updated_facts)
        return "Facts decayed and low-confidence ones removed."

    def self_reflect(self, assistant_reply: str) -> None:
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active. Skipping self_reflect.")
            return

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Reflect on the assistant's reply and provide a JSON response with the following structure: {quality:1-5, uncertainty:'low|med|high', memory_suggestion:''}."},
            {"role": "user", "content": assistant_reply}
        ]

        ok, response = self._post_chat(messages, 0.0, 120, stream=False)
        if not ok:
            logger.error(f"Failed to self-reflect: {response}")
            return

        data = response
        assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(assistant, str):
            assistant = str(assistant)

        try:
            reflection = json.loads(assistant)
            if isinstance(reflection, dict) and "quality" in reflection and "uncertainty" in reflection and "memory_suggestion" in reflection:
                quality = reflection["quality"]
                uncertainty = reflection["uncertainty"]
                memory_suggestion = reflection["memory_suggestion"]
            else:
                print("Invalid reflection JSON received.")
                return
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from reflection response.")
            return

        # Log the reflection
        reflection_entry = f"Timestamp: {datetime.now().isoformat()}\n"
        reflection_entry += f"Assistant Reply: {assistant_reply}\n"
        reflection_entry += f"Quality: {quality}\n"
        reflection_entry += f"Uncertainty: {uncertainty}\n"
        reflection_entry += f"Memory Suggestions: {memory_suggestion}\n\n"

        with self.self_reflection_log_path.open("a", encoding="utf-8") as f:
            f.write(reflection_entry)

        # If memory_suggestion is non-empty, call remember_fact
        if memory_suggestion:
            self.remember_fact(memory_suggestion)

    def fallback_reflect(self, assistant_reply: str) -> None:
        quality = "Good"
        uncertainty = "Low"
        memory_suggestions = []

        # Simple reflection logic
        if len(assistant_reply) < 10:
            quality = "Poor"
        if "I'm not sure" in assistant_reply:
            uncertainty = "High"
        if "I don't know" in assistant_reply:
            uncertainty = "High"
        if "I can't help with that" in assistant_reply:
            uncertainty = "High"
        if "I don't understand" in assistant_reply:
            uncertainty = "High"
        if "I'm sorry" in assistant_reply:
            uncertainty = "High"
        if "I'm unable to" in assistant_reply:
            uncertainty = "High"
        if "I'm not capable of" in assistant_reply:
            uncertainty = "High"
        if "I'm not equipped to" in assistant_reply:
            uncertainty = "High"
        if "I'm not programmed to" in assistant_reply:
            uncertainty = "High"
        if "I'm not designed to" in assistant_reply:
            uncertainty = "High"
        if "I'm not built to" in assistant_reply:
            uncertainty = "High"

        # Log the reflection
        reflection_entry = f"Timestamp: {datetime.now().isoformat()}\n"
        reflection_entry += f"Assistant Reply: {assistant_reply}\n"
        reflection_entry += f"Quality: {quality}\n"
        reflection_entry += f"Uncertainty: {uncertainty}\n"
        reflection_entry += f"Memory Suggestions: {memory_suggestions}\n\n"

        with self.self_reflection_log_path.open("a", encoding="utf-8") as f:
            f.write(reflection_entry)

    def update_identity_from_conversation(self) -> None:
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active. Skipping update_identity_from_conversation.")
            return

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Return ONLY valid JSON with keys: style (string), values (list), preferences (object). Keep concise. Do not add new keys."},
            {"role": "user", "content": json.dumps(self.conversation[-20:])}
        ]

        ok, response = self._post_chat(messages, self.temperature, 200, stream=False)
        if not ok:
            logger.error(f"Failed to update identity: {response}")
            return

        data = response
        assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(assistant, str):
            assistant = str(assistant)

        try:
            new_identity = json.loads(assistant)
            if self.validate_identity(new_identity):
                self.identity = new_identity
                self.save_identity(new_identity)
            else:
                logger.warning("Invalid identity JSON received.")
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from identity update response.")

    def health_check(self) -> str:
        url = f"{self.base_url}/models"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        try:
            response = self.session.get(url, headers=headers, timeout=(10, 120))
            if response.status_code == 200:
                return "OK"
            else:
                return f"FAIL: HTTP {response.status_code}"
        except requests.RequestException as e:
            return f"FAIL: {e}"

    def handle_exit(self) -> None:
        self.lockfile_path.unlink()

def main():
    ai = SimpleAI()

    try:
        while True:
            user_input = input("User: ")
            if user_input in ("/exit", "/quit"):
                break
            response = ai.chat(user_input)
            print(f"Assistant: {response}")
            ai.self_reflect(response)
    finally:
        ai.handle_exit()

if __name__ == "__main__":
    main()
