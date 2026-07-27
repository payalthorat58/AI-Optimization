import json
import ollama

# Load optimized prompt + document + requirements
with open("optimized_prompt.json", "r") as f:
    data = json.load(f)

optimized_prompt = data["optimized_prompt"]
document_text = data["document_text"]
requirements = data["requirements"]

# Prompt for Granite
metrics_prompt = f"""
You are an AI Model Analysis Agent.

Analyze the complete request and return ONLY valid JSON.

Request Details:

Optimized Prompt:
{optimized_prompt}

Supporting Document:
{document_text}

Requirements:
{json.dumps(requirements, indent=2)}

Return EXACTLY this JSON structure:

{{
    "estimated_input_tokens": 0,
    "estimated_output_tokens": 0,

    "estimated_cost": {{
        "granite4.1:3b": 0,
        "gemini-2.5-flash": 0,
        "llama-3.3-70b": 0
    }},

    "estimated_latency": {{
        "granite4.1:3b": 0.0,
        "gemini-2.5-flash": 0.0,
        "llama-3.3-70b": 0.0
    }},

    "task_complexity": "low | medium | high",

    "task_style": "creative | balanced | deterministic",

    "temperature": 0.0
}}

Rules:
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT explain anything.
- Base your estimates on the optimized prompt, the supporting document, and the user requirements.
- estimated_cost is a relative score from 1 (lowest cost) to 5 (highest cost).
- estimated_latency is an estimated execution time in seconds.
- Choose temperature according to:
    creative -> 0.8
    balanced -> 0.5
    deterministic -> 0.2
"""

# Call Granite
response = ollama.chat(
    model="granite4.1:8b",
    messages=[
        {
            "role": "user",
            "content": metrics_prompt
        }
    ]
)

# Get response
content = response["message"]["content"].strip()

# Remove markdown if Granite returns it
if content.startswith("```json"):
    content = content.replace("```json", "", 1)

if content.endswith("```"):
    content = content[:-3]

# Convert to JSON
metrics = json.loads(content.strip())

# Save metrics
with open("dynamic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Dynamic metrics saved successfully!")