import json
import ollama

# Load Shalmalee's output
with open("input.json", "r") as f:
    data = json.load(f)

original_prompt = data["original_prompt"]
document_text = data["document_text"]
requirements = data["requirements"]
metadata = data["metadata"]

# Prompt for Granite
optimizer_prompt = f"""
You are an expert prompt engineer.

Improve the user's prompt while preserving its original intent.

Requirements:
{json.dumps(requirements, indent=2)}

Document:
{document_text}

Original Prompt:
{original_prompt}

Rules:
- Preserve the user's intent.
- Include relevant document context if needed.
- Make the prompt clear, precise and complete.
- Return ONLY the optimized prompt.
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

optimized_prompt = response["message"]["content"].strip()

# Save everything for downstream agents
output = {
    "optimized_prompt": optimized_prompt,
    "document_text": document_text,
    "requirements": requirements,
    "metadata": metadata
}

with open("optimized_prompt.json", "w") as f:
    json.dump(output, f, indent=4)

print("Optimized prompt saved to optimized_prompt.json")