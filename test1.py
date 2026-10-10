import os
from openai import OpenAI

# client = OpenAI(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
# )

client = OpenAI(
     base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
 )


completion = client.chat.completions.create(
    model="gemma-4-31b-it",
    messages=[
        {"role": "user", "content": "你是誰？ 你能做什麼?"}
    ],
    stream=False
)

print("=== full response ===")
print(completion)

print("\n=== choices ===")
for i, choice in enumerate(completion.choices):
    print(f"[{i}] finish_reason: {choice.finish_reason}")
    print(f"[{i}] role: {choice.message.role}")
    print(f"[{i}] content: {choice.message.content}")