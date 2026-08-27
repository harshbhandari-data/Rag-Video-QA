from ollama import chat

response = chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "user",
            "content": "What is machine learning? Explain it simply."
        }
    ]
)

print(response["message"]["content"])