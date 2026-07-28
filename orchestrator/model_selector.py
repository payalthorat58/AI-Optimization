import json

# ---------------------------------
# Load Static Metrics (Registry)
# ---------------------------------
with open("registry.json", "r") as f:
    registry = json.load(f)

# ---------------------------------
# Load Dynamic Metrics
# ---------------------------------
with open("dynamic_metrics.json", "r") as f:
    dynamic = json.load(f)

task = dynamic["task"]

estimated_cost = dynamic["estimated_cost"]
estimated_latency = dynamic["estimated_latency"]
task_complexity = dynamic["task_complexity"]

# ---------------------------------
# Reasoning Score Mapping
# ---------------------------------
REASONING_SCORE = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4
}

# ---------------------------------
# Select Eligible Models
# ---------------------------------
eligible_models = []

for model in registry:

    model_type = model["type"]

    if task == "presentation" and model_type == "presentation":
        eligible_models.append(model)

    elif task == "image_generation" and model_type == "image":
        eligible_models.append(model)

    elif task not in ["presentation", "image_generation"] and model_type == "text":
        eligible_models.append(model)

# ---------------------------------
# If only one model matches
# ---------------------------------
if len(eligible_models) == 1:

    selected = eligible_models[0]

    output = {
        "task": task,
        "selected_model": selected,
        "model_score": 100
    }

    with open("selected_model.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"Selected Model: {selected['name']}")
    exit()

# ---------------------------------
# Score Text Models
# ---------------------------------
scores = {}

for model in eligible_models:

    name = model["name"]

    reasoning = REASONING_SCORE[model["reasoning_level"]]

    context = model["max_context"] / 128000

    privacy = 1 if model["supports_privacy"] else 0

    compliance = 1 if model["supports_compliance"] else 0

    cost_score = 6 - estimated_cost[name]

    latency_score = max(1, 6 - estimated_latency[name])

    # Increase reasoning weight for complex tasks
    if task_complexity == "high":

        reasoning_weight = 0.40
        cost_weight = 0.15
        latency_weight = 0.10

    elif task_complexity == "medium":

        reasoning_weight = 0.35
        cost_weight = 0.20
        latency_weight = 0.15

    else:

        reasoning_weight = 0.25
        cost_weight = 0.30
        latency_weight = 0.20

    final_score = (
        reasoning * reasoning_weight +
        cost_score * cost_weight +
        latency_score * latency_weight +
        context * 0.15 +
        privacy * 0.05 +
        compliance * 0.05
    )

    scores[name] = round(final_score, 3)

# ---------------------------------
# Best Model
# ---------------------------------
best_model_name = max(scores, key=scores.get)

selected_model = next(
    model for model in eligible_models
    if model["name"] == best_model_name
)

# ---------------------------------
# Save Selection
# ---------------------------------
output = {
    "task": task,
    "selected_model": selected_model,
    "model_score": scores[best_model_name]
}

with open("selected_model.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Selected Model: {best_model_name}")
print(f"Score: {scores[best_model_name]}")

