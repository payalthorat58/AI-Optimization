import json
import ollama

# Load Shalmalee's output
with open("input.json", "r") as f:
    data = json.load(f)

original_prompt = data["original_prompt"]
requirements = data["requirements"]

# Create prompt for Granite
optimizer_prompt = f"""
You are an expert prompt engineer.

Improve the user's prompt while preserving its intent.

Requirements:
{json.dumps(requirements, indent=2)}

Original Prompt:
{original_prompt}

Return ONLY the improved prompt.
"""

# Call Granite
response = ollama.chat(
    model="granite4.1:8b",
    messages=[
        {
            "role": "user",
            "content": optimizer_prompt
        }
    ]
)

optimized_prompt = response["message"]["content"]

print("Optimized Prompt:\n")
print(optimized_prompt)