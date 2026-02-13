import requests

class LlamaAgent:
    def __init__(self, url):
        self.url = url

    def query(self, prompt):
        response = requests.post(self.url, json={"prompt": prompt})
        return response.json().get("result", "")
