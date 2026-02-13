import requests

class SimpleAI:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, user_text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
            "stream": False,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"Request URL: {url}")
        print(f"Response Status Code: {r.status_code}")
        if r.status_code != 200:
            try:
                print("Error JSON:", r.json())
            except Exception:
                print("Error text:", r.text)
            return "No response"

        data = r.json()
        return data["choices"][0]["message"]["content"]

    def hello(self):
        print(self.chat("hello"))

    def how_are_you(self):
        print(self.chat("how are you?"))

    def goodbye(self):
        print(self.chat("goodbye"))

if __name__ == "__main__":
    ai = SimpleAI("http://127.0.0.1:8080/v1")
    ai.hello()
    ai.how_are_you()
    ai.goodbye()
