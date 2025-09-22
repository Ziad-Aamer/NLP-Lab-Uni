from litellm import completion
import os

# Set OpenAI-compatible vars
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "http://131.220.150.238:8080"

response = completion(
    model="mistral-tiny",   # model name stays as-is
    messages=[{"role": "user", "content": "hello from litellm"}],
)

print(response)
