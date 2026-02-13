import os
import requests
import webbrowser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
            with open("personality.txt", "w") as file:
                file.write("You are Xero, a friendly chatting coding bot but can also just have friendly conversations.")

    def chat(self, user_text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}
        with open("personality.txt", "r") as file:
            personality = file.read().strip()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": personality},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
            "stream": False,
        }

        r = self.session.post(url, headers=headers, json=payload, timeout=120)
        print(f"Request URL: {url}")
        print(f"Response Status Code: {r.status_code}")
        if r.status_code != 200:
            try:
                print("Error JSON:", r.json())
            except Exception:
                print("Error text:", r.text)
            return "No response"

        response_data = r.json()
        assistant_message = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if "look up" in assistant_message.lower() or "search" in assistant_message.lower():
            query = assistant_message.split("for", 1)[1].strip()
            self.search_internet(query)
            return f"Searching the internet for: {query}"

        return assistant_message

    def search_internet(self, query: str):
        """
        Simple internet lookup by opening a browser search.
        This does NOT scrape content, it just launches the user's browser.
        """
        base_url = "https://www.google.com/search?q="
        search_url = base_url + requests.utils.quote(query)
        print(f"Opening browser for: {search_url}")
        webbrowser.open(search_url)

    def hello(self):
        print(self.chat("hello"))

    def how_are_you(self):
        print(self.chat("how are you?"))

    def goodbye(self):
        print(self.chat("goodbye"))

if __name__ == "__main__":
    ai = SimpleAI("http://127.0.0.1:8080/v1")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("goodbye see you soon!")
            break
        elif user_input.lower() == "quit":
            print("goodbye see you soon!")
            break
        else:
            response = ai.chat(user_input)
            print(f"AI: {response}")
