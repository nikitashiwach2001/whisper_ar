import requests
import os

api_key = "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"
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())