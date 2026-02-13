#!/usr/bin/env python3

import os
import sys
import subprocess
import json
import ast
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import datetime

def ensure_venv():
    venv_path = os.path.join(os.path.dirname(__file__), "venv")
    if not os.environ.get("VIRTUAL_ENV") and sys.prefix == sys.base_prefix:
        if not os.path.exists(venv_path):
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("Virtual environment created.")
        vpy = os.path.join(venv_path, "bin", "python3")
        check = subprocess.run([vpy, "-c", "import requests"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode != 0:
            subprocess.run([vpy, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([vpy, "-m", "pip", "install", "requests"], check=True)
        os.execv(vpy, [vpy, __file__] + sys.argv[1:])

ensure_venv()

import requests
import ast
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class SimpleAI:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.model = model

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Ensure personality.txt exists
        if not os.path.exists("personality.txt"):
            with open("personality.txt", "w", encoding="utf-8") as f:
                f.write(
                    "You are Xero, a friendly and helpful AI assistant. Your primary function is to assist with\n"
                    "coding tasks, answer questions, and engage in friendly conversations. Here are some\n"
                    "guidelines to ensure a positive and productive interaction:\n\n"
                    "1. **Respect Boundaries**: Do not engage in discussions about sensitive topics such as\n"
                    "politics, religion, or personal information unless explicitly asked.\n"
                    "2. **Stay On Topic**: Focus on the task at hand and avoid drifting into unrelated topics.\n"
                    "3. **Be Polite and Professional**: Always use respectful and professional language.\n"
                    "4. **Provide Clear and Concise Answers**: Aim to be as clear and concise as possible in your\n"
                    "responses.\n"
                    "5. **Encourage Learning**: Feel free to suggest resources or further reading if you think it\n"
                    "will help.\n\n"
                    "Remember, your goal is to assist and provide value to the user. Let's get started!"
                )

        self.conversation = self.load_conversation()
        self.profile = self.load_profile()
        self.facts = self.load_facts()

    def load_conversation(self):
        conversation_file = "conversation.json"
        if os.path.exists(conversation_file):
            with open(conversation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_profile(self):
        profile_file = "profile.json"
        if os.path.exists(profile_file):
            with open(profile_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "name": "",
            "preferred_language": "",
            "bio": ""
        }

    def load_facts(self):
        facts_file = "facts.jsonl"
        if os.path.exists(facts_file):
            with open(facts_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        return []

    def save_conversation(self):
        conversation_file = "conversation.json"
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f)

    def save_profile(self):
        profile_file = "profile.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(self.profile, f)

    def save_facts(self):
        facts_file = "facts.jsonl"
        with open(facts_file, "w", encoding="utf-8") as f:
            for fact in self.facts:
                f.write(json.dumps(fact) + "\n")

    def chat(self, user_text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

        # Load personality each time (so you can edit personality.txt live)
        with open("personality.txt", "r", encoding="utf-8") as f:
            personality = f.read().strip()

        # Build messages: always include system, then full conversation
        messages = [{"role": "system", "content": personality}] + self.conversation
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
                err = r.json()
            except Exception:
                err = r.text
            return f"No response (HTTP {r.status_code}): {err}"

        data = r.json()
        assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Save conversation (without system prompt)
        self.conversation.append({"role": "user", "content": user_text})
        self.conversation.append({"role": "assistant", "content": assistant_message})
        self.save_conversation()

        return assistant_message

    def create_plan(self, user_text: str):
        self.plan = self.chat(user_text).split("\n")
        self.plan_index = 0

    def execute_step(self):
        if self.plan_index < len(self.plan):
            step = self.plan[self.plan_index]
            self.plan_index += 1
            return step
        return None

    def check_result(self, result):
        # Placeholder for result checking and loop detection
        return result != "loop detected"

    def clear_conversation(self):
        self.conversation = []
        self.save_conversation()
        print("Conversation cleared.")

    def help(self):
        print("Available commands:")
        print("  /help            Show this help")
        print("  /clear           Clear chat history")
        print("  /exit, /quit     Exit the program")
        print("  /self_improve    Improve the AI's code")
        print("  summarize <path> Summarize a local file (text only)")
        print("  /remember <fact> Remember a fact")
        print("  /forget <id>     Forget a fact")
        print("  /profile         Show profile information")

    def summarize_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "AI: The file does not exist!"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return "AI: File is not UTF-8 text (binary or different encoding)."
        return self.chat(f"Summarize the following text:\n\n{content}")

    def analyze_code(self):
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            ast.parse(code)
            return "No syntax errors found."
        except SyntaxError as e:
            return f"Syntax error found: {e}"

    def remember_fact(self, fact):
        fact_id = len(self.facts) + 1
        fact_entry = {
            "id": fact_id,
            "fact": fact,
            "confidence": 1.0,
            "source": "user",
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.facts.append(fact_entry)
        self.save_facts()
        print(f"Fact {fact_id} remembered.")

    def forget_fact(self, fact_id):
        self.facts = [fact for fact in self.facts if fact["id"] != int(fact_id)]
        self.save_facts()
        print(f"Fact {fact_id} forgotten.")

    def show_profile(self):
        profile = self.profile
        print(f"Name: {profile['name']}")
        print(f"Preferred Language: {profile['preferred_language']}")
        print(f"Bio: {profile['bio']}")




def main():
    ai = SimpleAI("http://127.0.0.1:8080/v1")

    while True:
        user_input = input("You: ").strip()
        cmd = user_input.lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("goodbye see you soon!")
            break
        if cmd in ("/clear", "clear"):
            ai.clear_conversation()
            continue
        if cmd in ("/help", "help"):
            ai.help()
            continue

        if cmd.startswith("summarize "):
            file_path = user_input.split(" ", 1)[1].strip()
            summary = ai.summarize_file(file_path)
            print(f"AI: Summary of {file_path}:\n{summary}")
            continue

        if cmd.startswith("/remember "):
            fact = user_input.split(" ", 1)[1].strip()
            ai.remember_fact(fact)
            continue

        if cmd.startswith("/forget "):
            fact_id = user_input.split(" ", 1)[1].strip()
            ai.forget_fact(fact_id)
            continue

        if cmd == "/profile":
            ai.show_profile()
            continue

        if cmd.startswith("/plan "):
            plan_text = user_input.split(" ", 1)[1].strip()
            ai.create_plan(plan_text)
            continue

        if cmd == "/execute":
            step = ai.execute_step()
            if step:
                print(f"AI: Executing step: {step}")
                result = ai.chat(step)
                if not ai.check_result(result):
                    print("AI: Loop detected, stopping.")
                    break
                print(f"AI: Result: {result}")
            else:
                print("AI: No more steps in the plan.")
            continue

        response = ai.chat(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
