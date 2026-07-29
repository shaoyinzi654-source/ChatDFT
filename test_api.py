import openai
from openai import OpenAI

client = OpenAI(
    api_key="sk-omWl5smTaCBOrRalgbbBk09Migy7e1w1J9raZXRDYjkzDfoY",
    base_url="https://tokken.cc/v1"
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
