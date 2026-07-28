import json
import sys

# ---------------------------------
# Load Static Metrics
# ---------------------------------
with open("static_metrics.json", "r") as f:
    static_metrics = json.load(f)

# ---------------------------------
# Load Dynamic Metrics
# ---------------------------------
with open("dynamic_metrics.json", "r") as f:
    dynamic = json.load(f)

task = dynamic["task"].lower()

estimated_cost = dynamic["estimated_cost"]
estimated_latency = dynamic["estimated_latency"]
task_complexity = dynamic["task_complexity"].lower()

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

for model in static_metrics:

    model_type = model["type"].lower()

    if task == "presentation":
        if model_type == "presentation":
            eligible_models.append(model)

    elif task == "image_generation":
        if model_type == "image":
            eligible_models.append(model)

    else:
        # All text-based tasks (report, summary, rewrite, coding, etc.)
        if model_type == "text":
            eligible_models.append(model)

# ---------------------------------
# No Eligible Model
# ---------------------------------
if not eligible_models:
    raise ValueError(f"No eligible model found for task '{task}'.")

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
    sys.exit()

# ---------------------------------
# Score Text Models
# ---------------------------------
scores = {}

for model in eligible_models:

    model_name = model["name"]

    # -----------------------------
    # Static Metrics
    # -----------------------------
    reasoning = REASONING_SCORE.get(
        model["reasoning_level"].lower(),
        0
    )

    max_context = model.get("max_context")

    if max_context is None:
        context_score = 0
    else:
        context_score = min(max_context / 128000, 1)

    privacy_score = 1 if model.get("supports_privacy", False) else 0

    compliance_score = 1 if model.get("supports_compliance", False) else 0

    # -----------------------------
    # Dynamic Metrics
    # -----------------------------
    cost = estimated_cost.get(model_name, 5)
    latency = estimated_latency.get(model_name, 5)

    cost_score = max(1, 6 - cost)
    latency_score = max(1, 6 - latency)

    # -----------------------------
    # Dynamic Weighting
    # -----------------------------
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

    context_weight = 0.15
    privacy_weight = 0.05
    compliance_weight = 0.05

    # -----------------------------
    # Final Score
    # -----------------------------
    final_score = (
        reasoning * reasoning_weight +
        cost_score * cost_weight +
        latency_score * latency_weight +
        context_score * context_weight +
        privacy_score * privacy_weight +
        compliance_score * compliance_weight
    )

    scores[model_name] = round(final_score, 3)

# ---------------------------------
# Select Best Model
# ---------------------------------
best_model_name = max(scores, key=scores.get)

selected_model = next(
    model
    for model in eligible_models
    if model["name"] == best_model_name
)

# ---------------------------------
# Save Selection
# ---------------------------------
output = {
    "task": task,
    "selected_model": selected_model,
    "model_score": scores[best_model_name],
    "all_scores": scores
}

with open("selected_model.json", "w") as f:
    json.dump(output, f, indent=4)

# ---------------------------------
# Display Results
# ---------------------------------
print("\nModel Selection Completed Successfully!\n")

print("Task:", task)
print("Complexity:", task_complexity)

print("\nScores:")
for model, score in scores.items():
    print(f"  {model}: {score}")

print(f"\nSelected Model: {best_model_name}")
print(f"Final Score: {scores[best_model_name]}")
