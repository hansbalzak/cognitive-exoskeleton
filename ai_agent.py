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
        subprocess.run([os.path.join(venv_path, "bin", "python3"), __file__], env=os.environ.copy())
        sys.exit(0)

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
                    "You are Xero, a friendly chatting coding bot but can also just have friendly conversations."
                )

        self.conversation = []

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

        return assistant_message

    def clear_conversation(self):
        self.conversation = []
        print("Conversation cleared.")

    def help(self):
        print("Available commands:")
        print("  /help            Show this help")
        print("  /clear           Clear chat history")
        print("  /exit, /quit     Exit the program")
        print("  /improve         Improve the AI's code")
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


    def improve_self(self):
        print("AI: Analyzing code for improvements...")
        analysis_result = self.analyze_code()
        print(f"AI: {analysis_result}")

        # Placeholder for local improvement checks
        improvements_found = False

        # Example: Check for unused imports
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
            tree = ast.parse(code)
            used_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        used_imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    used_imports.add(node.module.split('.')[0])

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_imports.discard(node.id)

            unused_imports = []
            for imp in ast.walk(tree):
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        if alias.name.split('.')[0] not in used_imports:
                            unused_imports.append(imp)
                elif isinstance(imp, ast.ImportFrom):
                    if imp.module and imp.module not in used_imports:
                        unused_imports.append(imp)
            if unused_imports:
                improvements_found = True
                print("AI: Unused imports found:")
                for imp in unused_imports:
                    print(f"AI: {ast.unparse(imp)}")

                # Remove unused imports from the code
                with open(__file__, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                with open(__file__, "w", encoding="utf-8") as f:
                    for line in lines:
                        if not any(ast.unparse(imp) in line for imp in unused_imports):
                            f.write(line)

                print("AI: Unused imports removed from the code.")

        if not improvements_found:
            print("AI: No specific improvement opportunities found.")
        else:
            # Install necessary packages
            print("AI: Installing necessary packages...")
            for imp in unused_imports:
                module_name = ast.unparse(imp).split()[1].strip("'\"")
                subprocess.run([os.path.join(os.path.dirname(__file__), "venv", "bin", "pip"), "install", module_name], check=True)
            print("AI: Packages installed.")


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

        if cmd in ("/self_improve", "self_improve"):
            ai.improve_self()
            continue

        response = ai.chat(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
