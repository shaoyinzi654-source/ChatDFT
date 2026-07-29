import os
import openai
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(
    api_key=api_key if api_key else "placeholder-key",
    base_url=base_url
)

def test_api():
    print("Testing tokken.cc API with model gpt-5.6-luna...")
    try:
        completion = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one word."}
            ],
            temperature=0.0
        )
        print("API Response:")
        print(completion.choices[0].message.content)
    except Exception as e:
        print("API Call failed:")
        print(e)

if __name__ == "__main__":
    test_api()
