#!/usr/bin/env python3

import os
import sys
import subprocess

def ensure_venv():
    venv_path = os.path.join(os.path.dirname(__file__), "venv")
    if not os.environ.get("VIRTUAL_ENV") and sys.prefix == sys.base_prefix:
        if not os.path.exists(venv_path):
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("Virtual environment created.")
        vpy = os.path.join(venv_path, "bin", "python3")
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

    def load_conversation(self):
        conversation_file = "conversation.json"
        if os.path.exists(conversation_file):
            with open(conversation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_conversation(self):
        conversation_file = "conversation.json"
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f)

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


        response = ai.chat(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
