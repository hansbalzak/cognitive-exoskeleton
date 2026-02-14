#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import threading
import time
import curses
import signal
import uuid
import logging
from logging.handlers import RotatingFileHandler
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib

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

# Ensure traces directory exists
traces_dir = Path("traces")
traces_dir.mkdir(parents=True, exist_ok=True)

# Ensure events directory exists
events_dir = Path("events")
events_dir.mkdir(parents=True, exist_ok=True)

# Ensure failures directory exists
failures_dir = Path("failures")
failures_dir.mkdir(parents=True, exist_ok=True)

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
        logger.warning(f"Failed to load {path}. Renaming to *.corrupt-{datetime.now().isoformat()}")
        path.rename(path.with_suffix(f'.corrupt-{datetime.now().isoformat()}'))
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


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


class SimpleAI:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, privacy_mode: bool, supervisor: bool):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.privacy_mode = privacy_mode
        self.supervisor = supervisor

        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})
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
        self.claims_path = self.repo_root / "claims.jsonl"
        self.config_path = self.repo_root / "config.json"
        self.eval_prompts_path = self.repo_root / "eval/prompts.jsonl"
        self.eval_results_path = self.repo_root / "eval/results.jsonl"
        self.events_path = self.repo_root / "events.jsonl"
        self.failures_path = self.repo_root / "failures.jsonl"
        self.state_path = self.repo_root / "state.json"
        self.identity_freeze_path = self.repo_root / "identity_freeze.json"

        self.ensure_personality_file()
        self.ensure_profile_and_facts()
        self.conversation: List[Dict[str, str]] = self.load_conversation()
        self.identity = self.load_identity()
        self.claims = self.load_claims()
        self.config = self.load_config()
        self.identity_freeze = self.load_identity_freeze()

        # Repo search index (lazy)
        self._index_built = False
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Circuit breaker
        self.consecutive_failures = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_end_time = 0

        # Reflect sampling
        self.reflect_counter = 0
        self.reflect_interval = 5

        # Supervisor loop
        self.supervisor_thread = None
        self.supervisor_interval = 30
        self.degraded_mode = False

        # Register signal handler to remove lockfile on exit
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

        # Initialize user message count
        self.user_message_count = 0

        # Load state
        self.load_state()

        # Start supervisor if enabled
        if self.supervisor:
            self.start_supervisor()

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
        self.log_event("conversation_saved", {"length": len(self.conversation)})

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
        self.log_event("facts_saved", {"length": len(facts)})

    def load_identity(self) -> Dict[str, Any]:
        identity = _safe_load_json(self.identity_path, default={
            "style": "friendly and approachable",
            "values": ["respect boundaries", "stay on topic", "be polite and professional", "provide clear and concise answers", "encourage learning"],
            "preferences": {
                "language": "English",
                "timezone": "UTC"
            },
            "name": "Xero"
        })
        if not isinstance(identity, dict):
            identity = {
                "style": "friendly and approachable",
                "values": ["respect boundaries", "stay on topic", "be polite and professional", "provide clear and concise answers", "encourage learning"],
                "preferences": {
                    "language": "English",
                    "timezone": "UTC"
                },
                "name": "Xero"
            }
        return identity

    def save_identity(self, identity: Dict[str, Any]) -> None:
        _atomic_write_json(self.identity_path, identity)
        self.log_event("identity_saved", identity)

    def validate_identity(self, identity: Dict[str, Any]) -> bool:
        return isinstance(identity, dict) and "style" in identity and "values" in identity and "preferences" in identity and "name" in identity

    def load_claims(self) -> List[Dict[str, Any]]:
        if not self.claims_path.exists():
            return []
        claims: List[Dict[str, Any]] = []
        for line in self.claims_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    claims.append(obj)
            except Exception:
                continue
        return claims

    def append_claims(self, claims: List[Dict[str, Any]]) -> None:
        with self.claims_path.open("a", encoding='utf-8') as f:
            for claim in claims:
                f.write(json.dumps(claim, ensure_ascii=False) + "\n")
        self.log_event("claims_added", {"length": len(claims)})

    def rewrite_claims(self, claims: List[Dict[str, Any]]) -> None:
        temp_path = self.claims_path.with_suffix('.tmp')
        with temp_path.open('w', encoding='utf-8') as f:
            for claim in claims:
                f.write(json.dumps(claim, ensure_ascii=False) + "\n")
        temp_path.rename(self.claims_path)
        self.log_event("claims_rewritten", {"length": len(claims)})

    def load_config(self) -> Dict[str, Any]:
        config = _safe_load_json(self.config_path, default={
            "allow_write": False,
            "allow_run": False,
            "allow_network_readonly": True,
            "privacy_mode": True
        })
        if not isinstance(config, dict):
            config = {
                "allow_write": False,
                "allow_run": False,
                "allow_network_readonly": True,
                "privacy_mode": True
            }
        return config

    def load_identity_freeze(self) -> Dict[str, Any]:
        identity_freeze = _safe_load_json(self.identity_freeze_path, default={
            "freeze_values": True,
            "freeze_name": True,
            "freeze_core_preferences": True
        })
        if not isinstance(identity_freeze, dict):
            identity_freeze = {
                "freeze_values": True,
                "freeze_name": True,
                "freeze_core_preferences": True
            }
        return identity_freeze

    def load_state(self) -> None:
        state = _safe_load_json(self.state_path, default={
            "last_reflect_ts": None,
            "message_count_since_reflect": 0
        })
        if isinstance(state, dict):
            self.last_reflect_ts = state.get("last_reflect_ts")
            self.message_count_since_reflect = state.get("message_count_since_reflect", 0)

    def save_state(self) -> None:
        state = {
            "last_reflect_ts": self.last_reflect_ts,
            "message_count_since_reflect": self.message_count_since_reflect
        }
        _atomic_write_json(self.state_path, state)
        self.log_event("state_saved", state)

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
                self.consecutive_failures = 0
                self.circuit_breaker_active = False
                return True, response.json()
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self.circuit_breaker_active = True
                    self.circuit_breaker_end_time = time.time() + 30
                logger.error(f"HTTP {response.status_code}: {response.text}")
                self.log_event("error", {"subsystem": "LLM", "error_type": "HTTP", "message": response.text})
                return False, response.text
        except requests.RequestException as e:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.circuit_breaker_active = True
                self.circuit_breaker_end_time = time.time() + 30
            logger.error(f"Request failed: {e}")
            self.log_event("error", {"subsystem": "LLM", "error_type": "RequestException", "message": str(e)})
            return False, str(e)

    def _post_chat_stream(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        return self.session.post(url, headers=headers, json=payload, stream=True)

    def chat(self, user_text: str, stream: bool = False) -> str:
        self._update_circuit_breaker_state()
        if self.circuit_breaker_active:
            return "Circuit breaker active. Please try again later."

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        personality = self.personality_path.read_text(encoding="utf-8", errors="ignore").strip()
        identity = self.identity
        identity_str = f"Style: {identity['style']}\nValues: {', '.join(identity['values'][:3])}\nPrefs: language={identity['preferences']['language']}, timezone={identity['preferences']['timezone']}\nName: {identity['name']}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": f"{personality}\n\nIdentity:\n{identity_str}"}
        ]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_text})

        if stream:
            response = self._post_chat_stream(messages, self.temperature, self.max_tokens)
            assistant = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8").split("data: ")[1])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                assistant += content
                                sys.stdout.write(content)
                                sys.stdout.flush()
                    except (json.JSONDecodeError, IndexError):
                        continue
            # Update conversation after stream completes
            self.conversation.append({"role": "user", "content": user_text})
            self.conversation.append({"role": "assistant", "content": assistant})
            self.trim_conversation()
            self.save_conversation()
            self.log_event("assistant_response_generated", {"length": len(assistant), "hash": _hash_content(assistant)})
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

            # Reflect sampling
            if self.should_reflect():
                self.self_reflect(assistant)

            # Extract and store claims
            new_claims = self.extract_claims(assistant)
            if new_claims:
                self.append_claims(new_claims)
                self.rewrite_claims(self.claims[-500:])
                print(f"(logged {len(new_claims)} claims; /claims to view)")

            self.log_event("assistant_response_generated", {"length": len(assistant), "hash": _hash_content(assistant)})
            return assistant

    def _contains_uncertainty_markers(self, text: str) -> bool:
        uncertainty_markers = [
            "I'm not sure",
            "I don't know",
            "I can't help with that",
            "I don't understand",
            "I'm sorry",
            "I'm unable to",
            "I'm not capable of",
            "I'm not equipped to",
            "I'm not programmed to",
            "I'm not designed to",
            "I'm not built to"
        ]
        return any(marker in text for marker in uncertainty_markers)

    # --------- Conversation Trimming ---------
    def trim_conversation(self) -> None:
        if len(self.conversation) > 40:
            summary = self.summarize_conversation()
            self.conversation = [
                {"role": "system", "content": summary},
                *self.conversation[-10:]
            ]
            self.save_identity(self.identity)
            self.log_event("conversation_summarized", {"length": len(self.conversation)})

    def summarize_conversation(self) -> str:
        self._update_circuit_breaker_state()
        if self.circuit_breaker_active:
            return ""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Summarize the following conversation into a compact summary with the following sections:\n\n1. User goals\n2. Important decisions\n3. Open tasks/questions\n4. Stable user prefs\n\nEnsure the summary stays under ~800 characters."},
            {"role": "user", "content": json.dumps(self.conversation)}
        ]

        ok, response = self._post_chat(messages, 0.0, 200, stream=False)
        if not ok:
            logger.error(f"Failed to summarize conversation: {response}")
            self.log_event("error", {"subsystem": "summarization", "error_type": "HTTP", "message": response})
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
        self.log_event("fact_added", {"id": new_fact["id"], "fact": fact})
        return f"Fact remembered (id={new_fact['id']})."

    def forget_fact(self, fact_id: int) -> str:
        facts = self.load_facts()
        new_facts = [f for f in facts if int(f.get("id", -1)) != fact_id]
        self.save_facts(new_facts)
        self.log_event("fact_removed", {"id": fact_id})
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
                "  /claims [n]           List last n claims with id/status/confidence/topic",
                "  /claim <id>           Show a single claim with history",
                "  /correct <id> <note>  Mark claim corrected; store note; reduce confidence strongly",
                "  /retract <id> <note>  Mark claim retracted; confidence to 0",
                "  /verify <id>          Ask the LLM to self-check the claim using only repo snippets (from /askrepo) + known facts; then update confidence/status accordingly (soft-fail if not enough evidence)",
                "  /eval                 Run evaluation prompts and store results",
                "  /supervisor           Enable or disable supervisor mode",
                "  /privacy_mode         Enable or disable privacy mode",
                "  /identity_freeze      Enable or disable identity freeze",
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
                return "Usage: /write <path> <text>"
            path, content = parts[0], parts[1]
            if not self.config["allow_write"]:
                return "ERROR: Writing is not allowed."
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
            if not self.config["allow_run"]:
                return "ERROR: Running commands is not allowed."
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

        if cmd.startswith("/claims "):
            n = 10
            if cmd[len("/claims ") :].strip():
                try:
                    n = int(cmd[len("/claims ") :].strip())
                except ValueError:
                    return "Usage: /claims [n]"
            return self.list_claims(n)

        if cmd.startswith("/claim "):
            raw = cmd[len("/claim ") :].strip()
            try:
                return self.show_claim(int(raw))
            except Exception:
                return "Usage: /claim <id>"

        if cmd.startswith("/correct "):
            rest = cmd[len("/correct ") :].strip()
            parts = rest.split(" ", 1)
            if len(parts) != 2:
                return "Usage: /correct <id> <note>"
            try:
                claim_id = int(parts[0])
                note = parts[1]
                return self.correct_claim(claim_id, note)
            except Exception:
                return "Usage: /correct <id> <note>"

        if cmd.startswith("/retract "):
            rest = cmd[len("/retract ") :].strip()
            parts = rest.split(" ", 1)
            if len(parts) != 2:
                return "Usage: /retract <id> <note>"
            try:
                claim_id = int(parts[0])
                note = parts[1]
                return self.retract_claim(claim_id, note)
            except Exception:
                return "Usage: /retract <id> <note>"

        if cmd.startswith("/verify "):
            raw = cmd[len("/verify ") :].strip()
            try:
                return self.verify_claim(int(raw))
            except Exception:
                return "Usage: /verify <id>"

        if cmd == "/eval":
            return self.eval_prompts()

        if cmd == "/supervisor":
            self.supervisor = not self.supervisor
            if self.supervisor:
                self.start_supervisor()
            else:
                self.stop_supervisor()
            return f"Supervisor mode {'enabled' if self.supervisor else 'disabled'}."

        if cmd == "/privacy_mode":
            self.config["privacy_mode"] = not self.config["privacy_mode"]
            self.save_config()
            return f"Privacy mode {'enabled' if self.config['privacy_mode'] else 'disabled'}."

        if cmd == "/identity_freeze":
            self.identity_freeze = not self.identity_freeze
            self.save_identity_freeze()
            return f"Identity freeze {'enabled' if self.identity_freeze else 'disabled'}."

        return "Unknown command. Type /help."

    def _approve(self, label: str) -> bool:
        # Only ask approval for risky actions; if sandbox not available, no need
        if not SANDBOX_AVAILABLE:
            return True
        ans = input(f"About to run {label}. Proceed? (y/n) ").strip().lower()
        return ans in ("y", "yes")

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
        self.log_event("facts_decayed", {"length": len(updated_facts)})
        return "Facts decayed and low-confidence ones removed."

    def self_reflect(self, assistant_reply: str) -> None:
        self._update_circuit_breaker_state()
        if self.circuit_breaker_active or self.degraded_mode:
            logger.warning("Circuit breaker active or degraded mode. Skipping self_reflect.")
            self.log_event("self_reflect_skipped", {"reason": "circuit_breaker_active or degraded_mode"})
            return

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Reflect on the assistant's reply and provide a JSON response with the following structure: {quality:1-5, uncertainty:'low|med|high', memory_suggestion:''}."},
            {"role": "user", "content": assistant_reply}
        ]

        ok, response = self._post_chat(messages, 0.0, 120, stream=False)
        if not ok:
            logger.error(f"Failed to self-reflect: {response}")
            self.log_event("error", {"subsystem": "self_reflect", "error_type": "HTTP", "message": response})
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
                self.log_event("invalid_reflection", {"assistant_reply": assistant_reply})
                return
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from reflection response.")
            self.log_event("error", {"subsystem": "self_reflect", "error_type": "JSONDecodeError", "message": "Failed to decode JSON from reflection response."})
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

        self.log_event("self_reflected", {"quality": quality, "uncertainty": uncertainty, "memory_suggestion": memory_suggestion})

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
        self._update_circuit_breaker_state()
        if self.circuit_breaker_active or self.degraded_mode:
            logger.warning("Circuit breaker active or degraded mode. Skipping identity update.")
            self.log_event("identity_update_skipped", {"reason": "circuit_breaker_active or degraded_mode"})
            return

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Return ONLY valid JSON with keys: style (string), values (list), preferences (object), name (string). Keep concise. Do not add new keys."},
            {"role": "user", "content": json.dumps(self.conversation[-20:])}
        ]

        ok, response = self._post_chat(messages, 0.2, 200, stream=False)
        if not ok:
            logger.error(f"Failed to update identity: {response}")
            self.log_event("error", {"subsystem": "identity_update", "error_type": "HTTP", "message": response})
            return

        data = response
        assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(assistant, str):
            assistant = str(assistant)

        try:
            new_identity = json.loads(assistant)
            if self.validate_identity(new_identity):
                if self.identity_freeze["freeze_values"]:
                    new_identity["values"] = self.identity["values"]
                if self.identity_freeze["freeze_name"]:
                    new_identity["name"] = self.identity["name"]
                if self.identity_freeze["freeze_core_preferences"]:
                    new_identity["preferences"] = self.identity["preferences"]

                if new_identity != self.identity:
                    self.identity = new_identity
                    self.save_identity(new_identity)
                    self.log_event("identity_updated", {"diff": {k: (self.identity[k], new_identity[k]) for k in self.identity if self.identity[k] != new_identity[k]}})
                else:
                    self.log_event("identity_update_skipped", {"reason": "no_changes"})
            else:
                logger.warning("Invalid identity JSON received.")
                self.log_event("invalid_identity", {"assistant_response": assistant})
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from identity update response.")
            self.log_event("error", {"subsystem": "identity_update", "error_type": "JSONDecodeError", "message": "Failed to decode JSON from identity update response."})

    def health_check(self) -> str:
        url = f"{self.base_url}/models"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        try:
            response = self.session.get(url, headers=headers, timeout=(10, 120))
            if response.status_code == 200:
                self.degraded_mode = False
                self.log_event("supervisor_health_check", {"status": "OK"})
                return "OK"
            else:
                self.degraded_mode = True
                self.log_event("supervisor_health_check", {"status": "FAIL", "http_status": response.status_code})
                return f"FAIL: HTTP {response.status_code}"
        except requests.RequestException as e:
            self.degraded_mode = True
            self.log_event("supervisor_health_check", {"status": "FAIL", "error": str(e)})
            return f"FAIL: {e}"

    def handle_exit(self, *args) -> None:
        if self.supervisor_thread:
            self.supervisor_thread.join()

    def _update_circuit_breaker_state(self) -> None:
        if self.circuit_breaker_active and time.time() > self.circuit_breaker_end_time:
            self.circuit_breaker_active = False
            self.consecutive_failures = 0
            self.log_event("circuit_breaker_recovered", {})

    def extract_claims(self, assistant_reply: str) -> List[Dict[str, Any]]:
        self._update_circuit_breaker_state()
        if self.circuit_breaker_active or self.degraded_mode:
            logger.warning("Circuit breaker active or degraded mode. Skipping claim extraction.")
            self.log_event("claim_extraction_skipped", {"reason": "circuit_breaker_active or degraded_mode"})
            return []

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Extract 1-3 atomic factual claims from the following assistant reply. Return ONLY valid JSON list of claims with the following structure: [{claim: '...', topic: '...'}]. Keep the claims short and to the point. Do not include any other text or explanations."},
            {"role": "user", "content": assistant_reply}
        ]

        ok, response = self._post_chat(messages, 0.0, 100, stream=False)
        if not ok:
            logger.error(f"Failed to extract claims: {response}")
            self.log_event("error", {"subsystem": "claim_extraction", "error_type": "HTTP", "message": response})
            return []

        data = response
        assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(assistant, str):
            assistant = str(assistant)

        try:
            claims = json.loads(assistant)
            if isinstance(claims, list) and all(isinstance(c, dict) and "claim" in c and "topic" in c for c in claims):
                new_claims = []
                for claim in claims:
                    new_claim = {
                        "id": (max([c.get("id", 0) for c in self.claims], default=0) + 1),
                        "claim": claim["claim"],
                        "topic": claim["topic"],
                        "confidence": 0.6,
                        "status": "active",
                        "created_at": datetime.now().isoformat(),
                        "last_seen_at": datetime.now().isoformat(),
                        "source": "assistant",
                        "corrections": []
                    }
                    new_claims.append(new_claim)
                self.claims.extend(new_claims)
                self.rewrite_claims(self.claims[-500:])
                self.log_event("claims_extracted", {"length": len(new_claims)})
                return new_claims
            else:
                logger.warning("Invalid claims JSON received.")
                self.log_event("invalid_claims", {"assistant_reply": assistant_reply})
                return []
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from claims extraction response.")
            self.log_event("error", {"subsystem": "claim_extraction", "error_type": "JSONDecodeError", "message": "Failed to decode JSON from claims extraction response."})
            return []

    def list_claims(self, n: int = 10) -> str:
        claims = self.claims[-n:]
        if not claims:
            return "No claims found."
        out = ["Claims:"]
        for claim in claims:
            out.append(f"ID: {claim['id']}, Status: {claim['status']}, Confidence: {claim['confidence']:.2f}, Topic: {claim['topic']}")
        return "\n".join(out)

    def show_claim(self, claim_id: int) -> str:
        for claim in self.claims:
            if claim["id"] == claim_id:
                out = [
                    f"ID: {claim['id']}",
                    f"Claim: {claim['claim']}",
                    f"Topic: {claim['topic']}",
                    f"Confidence: {claim['confidence']:.2f}",
                    f"Status: {claim['status']}",
                    f"Created At: {claim['created_at']}",
                    f"Last Seen At: {claim['last_seen_at']}",
                    f"Source: {claim['source']}",
                    "Corrections:"
                ]
                for correction in claim.get("corrections", []):
                    out.append(f"  At: {correction['at']}, Note: {correction['note']}, New Status: {correction['new_status']}")
                return "\n".join(out)
        return "Claim not found."

    def correct_claim(self, claim_id: int, note: str) -> str:
        for claim in self.claims:
            if claim["id"] == claim_id:
                if claim["status"] == "active":
                    claim["status"] = "corrected"
                    claim["confidence"] = max(0.0, claim["confidence"] - 0.5)
                    claim["corrections"].append({
                        "at": datetime.now().isoformat(),
                        "note": note,
                        "new_status": "corrected"
                    })
                    self.rewrite_claims(self.claims)
                    self.log_event("claim_corrected", {"id": claim_id, "note": note})
                    return f"Claim {claim_id} corrected. Confidence reduced to {claim['confidence']:.2f}."
                else:
                    return f"Claim {claim_id} is already {claim['status']}."
        return "Claim not found."

    def retract_claim(self, claim_id: int, note: str) -> str:
        for claim in self.claims:
            if claim["id"] == claim_id:
                if claim["status"] == "active":
                    claim["status"] = "retracted"
                    claim["confidence"] = 0.0
                    claim["corrections"].append({
                        "at": datetime.now().isoformat(),
                        "note": note,
                        "new_status": "retracted"
                    })
                    self.rewrite_claims(self.claims)
                    self.log_event("claim_retracted", {"id": claim_id, "note": note})
                    return f"Claim {claim_id} retracted. Confidence set to 0.0."
                else:
                    return f"Claim {claim_id} is already {claim['status']}."
        return "Claim not found."

    def verify_claim(self, claim_id: int) -> str:
        for claim in self.claims:
            if claim["id"] == claim_id:
                if claim["status"] == "corrected" or claim["status"] == "retracted":
                    return f"Claim {claim_id} is already {claim['status']}. No need to verify."

                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": "Verify the following claim using only the provided repo snippets and known facts. Return ONLY valid JSON with keys: verified (bool), note (str). Do not include any other text or explanations."},
                    {"role": "user", "content": claim["claim"]},
                    {"role": "user", "content": "Repo snippets:\n" + "\n".join([f"File: {rel_path}, Line: {line_no}, Context: {context}" for rel_path, line_no, context in self.search_repo(claim["claim"])[:10]])}
                ]

                ok, response = self._post_chat(messages, 0.0, 200, stream=False)
                if not ok:
                    logger.error(f"Failed to verify claim: {response}")
                    self.log_event("error", {"subsystem": "claim_verification", "error_type": "HTTP", "message": response})
                    return f"Failed to verify claim {claim_id}."

                data = response
                assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not isinstance(assistant, str):
                    assistant = str(assistant)

                try:
                    verification = json.loads(assistant)
                    if isinstance(verification, dict) and "verified" in verification and "note" in verification:
                        verified = verification["verified"]
                        note = verification["note"]
                        if verified:
                            claim["status"] = "active"
                            claim["confidence"] = min(1.0, claim["confidence"] + 0.2)
                        else:
                            claim["status"] = "retracted"
                            claim["confidence"] = 0.0
                        claim["corrections"].append({
                            "at": datetime.now().isoformat(),
                            "note": note,
                            "new_status": claim["status"]
                        })
                        self.rewrite_claims(self.claims)
                        self.log_event("claim_verified", {"id": claim_id, "verified": verified, "note": note})
                        return f"Claim {claim_id} verified. Status: {claim['status']}, Confidence: {claim['confidence']:.2f}."
                    else:
                        logger.warning("Invalid verification JSON received.")
                        self.log_event("invalid_verification", {"claim_id": claim_id, "assistant_response": assistant})
                        return f"Invalid verification response for claim {claim_id}."
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from verification response.")
                    self.log_event("error", {"subsystem": "claim_verification", "error_type": "JSONDecodeError", "message": "Failed to decode JSON from verification response."})
                    return f"Failed to decode JSON for claim {claim_id}."
        return "Claim not found."

    def inject_known_corrections(self, user_text: str) -> str:
        relevant_claims = []
        for claim in self.claims:
            if claim["status"] in ["corrected", "retracted"]:
                for keyword in claim["claim"].split():
                    if keyword.lower() in user_text.lower():
                        relevant_claims.append(claim)
                        break
        if not relevant_claims:
            return ""

        corrections = "\n".join([
            f"Corrected/Retracted: {claim['claim']} (Topic: {claim['topic']}, Status: {claim['status']})"
            for claim in relevant_claims[:10]
        ])
        return f"Known Corrections:\n{corrections}\n"

    def eval_prompts(self) -> str:
        if not self.eval_prompts_path.exists():
            return "No eval/prompts.jsonl found."

        prompts = []
        for line in self.eval_prompts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "prompt" in obj:
                    prompts.append(obj["prompt"])
            except Exception:
                continue

        if not prompts:
            return "No valid prompts found."

        results = []
        for prompt in prompts:
            start_time = time.time()
            response = self.chat(prompt, stream=False)
            end_time = time.time()
            duration = end_time - start_time

            # Simple self-score rubric
            score = 0
            if "correct" in response.lower():
                score += 1
            if "helpful" in response.lower():
                score += 1
            if "clear" in response.lower():
                score += 1

            result = {
                "trace_id": self.trace_id,
                "prompt": prompt,
                "response": response,
                "score": score,
                "duration": duration
            }
            results.append(result)

        with self.eval_results_path.open("a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        self.log_event("eval_completed", {"length": len(results)})
        return "Evaluation completed. Results stored in eval/results.jsonl."

    def log_event(self, event_type: str, payload: Dict[str, Any], trace_id: str = None) -> None:
        event = {
            "ts": datetime.now().isoformat(),
            "trace_id": trace_id,
            "event_type": event_type,
            "payload": payload
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_failure(self, subsystem: str, error_type: str, message: str, note: str = "", mitigation: str = "") -> None:
        failure = {
            "ts": datetime.now().isoformat(),
            "subsystem": subsystem,
            "error_type": error_type,
            "message": message,
            "note": note,
            "mitigation": mitigation
        }
        with self.failures_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def should_reflect(self) -> bool:
        if self.user_message_count >= 8 or (time.time() - self.last_reflect_ts) >= 60 * 60:
            self.last_reflect_ts = time.time()
            self.message_count_since_reflect = 0
            self.save_state()
            return True
        return False

    def start_supervisor(self) -> None:
        self.supervisor_thread = threading.Thread(target=self.supervisor_loop, daemon=True)
        self.supervisor_thread.start()

    def stop_supervisor(self) -> None:
        if self.supervisor_thread:
            self.supervisor_thread.join()
            self.supervisor_thread = None

    def supervisor_loop(self) -> None:
        while self.supervisor:
            self.health_check()
            self.sanity_check()
            time.sleep(self.supervisor_interval)

    def sanity_check(self) -> None:
        try:
            self.validate_json_state_files()
        except Exception as e:
            self.log_event("sanity_check_failed", {"error": str(e)})

    def validate_json_state_files(self) -> None:
        state_files = [
            self.conversation_path,
            self.facts_path,
            self.claims_path,
            self.identity_path
        ]

        for file_path in state_files:
            try:
                _safe_load_json(file_path, default={})
            except Exception as e:
                self.log_event("file_corruption_detected", {"file_path": str(file_path), "error": str(e)})
                self.recover_corrupted_file(file_path)

    def recover_corrupted_file(self, file_path: Path) -> None:
        backup_path = file_path.with_suffix('.corrupt-' + datetime.now().isoformat())
        file_path.rename(backup_path)
        self.log_event("file_recovered", {"original_path": str(file_path), "backup_path": str(backup_path)})
        self.create_minimal_safe_default(file_path)

    def create_minimal_safe_default(self, file_path: Path) -> None:
        if file_path == self.conversation_path:
            _ensure_file(file_path, json.dumps([], indent=2))
        elif file_path == self.facts_path:
            _ensure_file(file_path, "")
        elif file_path == self.claims_path:
            _ensure_file(file_path, "")
        elif file_path == self.identity_path:
            _ensure_file(file_path, json.dumps({
                "style": "friendly and approachable",
                "values": ["respect boundaries", "stay on topic", "be polite and professional", "provide clear and concise answers", "encourage learning"],
                "preferences": {
                    "language": "English",
                    "timezone": "UTC"
                },
                "name": "Xero"
            }, indent=2))

    def save_config(self) -> None:
        _atomic_write_json(self.config_path, self.config)
        self.log_event("config_saved", self.config)

    def save_identity_freeze(self) -> None:
        _atomic_write_json(self.identity_freeze_path, self.identity_freeze)
        self.log_event("identity_freeze_saved", self.identity_freeze)

def main():
    parser = argparse.ArgumentParser(description="Run the AI agent.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8080/v1"), help="Base URL for the LLM API")
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen2.5-7B-Instruct"), help="Model to use for the LLM")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")), help="Temperature for the LLM")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "100")), help="Max tokens for the LLM")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--no-tui", action="store_true", help="Disable TUI mode")
    parser.add_argument("--privacy-mode", action="store_true", default=True, help="Enable privacy mode (log only hashes/lengths)")
    parser.add_argument("--supervisor", action="store_true", default=False, help="Enable supervisor mode")
    args = parser.parse_args()

    ai = SimpleAI(base_url=args.base_url, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, privacy_mode=args.privacy_mode, supervisor=args.supervisor)

    if not args.no_tui:
        def tui(stdscr):
            curses.curs_set(0)
            stdscr.nodelay(1)
            stdscr.timeout(100)

            user_input = ""
            assistant_response = ""

            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "User: ")
                stdscr.addstr(0, 6, user_input)
                stdscr.addstr(2, 0, "Xero: " + assistant_response)
                stdscr.refresh()

                key = stdscr.getch()
                if key == curses.KEY_ENTER or key == 10:
                    if user_input.strip():
                        if user_input.startswith('/'):
                            response = ai.handle_command(user_input)
                            if response == "__EXIT__":
                                break
                            assistant_response = response
                        else:
                            assistant_response = ai.chat(user_input, stream=args.stream)
                        ai.log_trace(user_input, assistant_response)
                        user_input = ""
                    else:
                        assistant_response = ""
                elif key == curses.KEY_BACKSPACE or key == 127:
                    user_input = user_input[:-1]
                elif key >= 32 and key <= 126:
                    user_input += chr(key)

        curses.wrapper(tui)
    else:
        while True:
            user_input = input("User: ")
            if user_input.startswith('/'):
                response = ai.handle_command(user_input)
                if response == "__EXIT__":
                    break
                print("Xero:", response)
            else:
                assistant_response = ai.chat(user_input, stream=args.stream)
                ai.log_trace(user_input, assistant_response)

if __name__ == "__main__":
    main()
