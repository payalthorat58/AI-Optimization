import json
import ollama

# Load optimized prompt data
with open("optimized_prompt.json", "r") as f:
    data = json.load(f)

optimized_prompt = data["optimized_prompt"]
document_text = data["document_text"]
requirements = data["requirements"]
metadata = data["metadata"]

# Preserve task from Shalmalee
task = requirements["task"]

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

Metadata:
{json.dumps(metadata, indent=2)}

Return EXACTLY this JSON structure:

{{
    "estimated_input_tokens": 0,
    "estimated_output_tokens": 0,

    "estimated_cost": {{
        "granite4.1:3b": 0,
        "granite3.1-moe:3b": 0,
        "presenton": 0,
        "black-forest-labs/FLUX.1-dev": 0
    }},

    "estimated_latency": {{
        "granite4.1:3b": 0.0,
        "granite3.1-moe:3b": 0.0,
        "presenton": 0.0,
        "black-forest-labs/FLUX.1-dev": 0.0
    }},

    "task_complexity": "low | medium | high",

    "task_style": "creative | balanced | deterministic",

    "temperature": 0.0
}}

Rules:
- The task has already been identified as "{task}".
- DO NOT determine or modify the task.
- Base your estimates on the optimized prompt, supporting document, requirements and metadata.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT explain anything.
- estimated_cost is a relative score from 1 (lowest cost) to 5 (highest cost).
- estimated_latency is an estimated execution time in seconds.
- Temperature mapping:
    creative -> 0.8
    balanced -> 0.5
    deterministic -> 0.2
"""

# Call Granite
response = ollama.chat(
    model="granite4.1:3b",
    messages=[
        {
            "role": "user",
            "content": metrics_prompt
        }
    ]
)

content = response["message"]["content"].strip()

# Remove markdown if Granite returns it
if content.startswith("```json"):
    content = content.replace("```json", "", 1)

if content.endswith("```"):
    content = content[:-3]

content = content.strip()

# Convert to JSON
metrics = json.loads(content)

# Preserve all context for downstream modules
dynamic_metrics = {
    "task": task,
    "optimized_prompt": optimized_prompt,
    "document_text": document_text,
    "requirements": requirements,
    "metadata": metadata,
    **metrics
}

# Save metrics
with open("dynamic_metrics.json", "w") as f:
    json.dump(dynamic_metrics, f, indent=4)

print("Dynamic metrics saved successfully!")