import requests

class SimpleAI:
    def __init__(self, url):
        self.url = url

    def send_command(self, command):
        response = requests.post(self.url, json={"command": command})
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")
        return response.json().get("response", "No response")

    def hello(self):
        print(self.send_command("hello"))

    def how_are_you(self):
        print(self.send_command("how are you?"))

    def goodbye(self):
        print(self.send_command("goodbye"))

if __name__ == "__main__":
    ai = SimpleAI("http://127.0.0.1:8080")
    ai.hello()
    ai.how_are_you()
    ai.goodbye()
