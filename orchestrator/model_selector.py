import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
MODELS_FILE = BASE_DIR / "models" / "static_metrics.json"
DYNAMIC_METRICS_FILE = BASE_DIR / "dynamic_metrics.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "selected_model.json"


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def unique_names(items: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    ordered_names: List[str] = []
    for item in items:
        name = item.get("name")
        if name and name not in seen:
            seen.add(name)
            ordered_names.append(name)
    return ordered_names


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            try:
                return int(float(cleaned))
            except ValueError:
                return None
    return None


def get_model_by_name(models: List[Dict[str, Any]], name: str) -> Dict[str, Any] | None:
    for model in models:
        if isinstance(model, dict) and model.get("name") == name:
            return model
    return None


def infer_task(requirements: Dict[str, Any], dynamic_metrics: Dict[str, Any]) -> str:
    task = normalize_text(requirements.get("task") or dynamic_metrics.get("task"))
    if task:
        return task

    output_format = normalize_text(requirements.get("output_format"))
    if output_format in {"ppt", "pptx", "presentation"}:
        return "presentation"
    if output_format in {"png", "jpg", "jpeg", "image"}:
        return "image"
    return "text"


def select_primary_model(
    task: str,
    requirements: Dict[str, Any],
    dynamic_metrics: Dict[str, Any],
    models: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    if not models:
        raise ValueError("Model registry is empty.")

    task = task.lower()
    complexity = normalize_text(dynamic_metrics.get("task_complexity")) or "medium"
    style = normalize_text(dynamic_metrics.get("task_style")) or "balanced"
    output_format = normalize_text(requirements.get("output_format"))
    slides = coerce_int(requirements.get("slides"))

    if task == "presentation" or output_format in {"ppt", "pptx", "presentation"}:
        primary = get_model_by_name(models, "presenton") or models[0]
        fallback_models = [
            get_model_by_name(models, "granite4.1:3b"),
            get_model_by_name(models, "granite3.1-moe:3b"),
        ]
        fallback = [m for m in fallback_models if m is not None]
        reason = "Presentation-oriented task with PPTX output is best served by the dedicated presentation model."
        return primary, fallback + [m for m in models if m.get("name") != primary.get("name")], reason

    if task == "image" or output_format in {"png", "jpg", "jpeg", "image"}:
        primary = get_model_by_name(models, "black-forest-labs/FLUX.1-dev") or models[0]
        fallback_models = [
            get_model_by_name(models, "granite4.1:3b"),
            get_model_by_name(models, "granite3.1-moe:3b"),
        ]
        fallback = [m for m in fallback_models if m is not None]
        reason = "Image generation requires the image-capable model with the highest fit."
        return primary, fallback + [m for m in models if m.get("name") != primary.get("name")], reason

    scored_models: List[Tuple[float, Dict[str, Any]]] = []
    for model in models:
        if not isinstance(model, dict):
            continue

        score = 0.0
        name = model.get("name", "")
        model_type = normalize_text(model.get("type"))

        if model_type == "text":
            score += 40.0
        elif name == "presenton":
            score -= 100.0
        elif name == "black-forest-labs/FLUX.1-dev":
            score -= 100.0

        reasoning_level = normalize_text(model.get("reasoning_level"))
        if complexity == "high" and reasoning_level == "high":
            score += 25.0
        elif complexity == "medium" and reasoning_level in {"medium", "high"}:
            score += 15.0
        elif complexity == "low" and reasoning_level == "medium":
            score += 5.0

        if style == "creative" and reasoning_level == "high":
            score += 8.0
        elif style == "balanced" and reasoning_level in {"medium", "high"}:
            score += 5.0

        if requirements.get("tone") == "professional" and name in {"granite4.1:3b", "granite3.1-moe:3b"}:
            score += 3.0

        if slides is not None and slides > 6 and name == "presenton":
            score += 10.0

        estimated_cost = dynamic_metrics.get("estimated_cost", {})
        estimated_latency = dynamic_metrics.get("estimated_latency", {})
        if isinstance(estimated_cost, dict) and isinstance(estimated_latency, dict):
            cost_value = estimated_cost.get(name, 0)
            latency_value = estimated_latency.get(name, 0.0)
            score -= max(0, cost_value - 1) * 3.0
            score -= latency_value * 1.5

        scored_models.append((score, model))

    if not scored_models:
        raise ValueError("No valid model entries were available for scoring.")

    scored_models.sort(key=lambda item: item[0], reverse=True)
    primary = scored_models[0][1]
    ordered_fallbacks = [model for _, model in scored_models[1:] if model.get("name") != primary.get("name")]
    reason = (
        f"Text task selected the model with the best balance of reasoning and latency for {complexity} complexity "
        f"and {style} style."
    )
    return primary, ordered_fallbacks, reason


def build_selection_payload(
    task: str,
    requirements: Dict[str, Any],
    dynamic_metrics: Dict[str, Any],
    models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary_model, fallbacks, reason = select_primary_model(task, requirements, dynamic_metrics, models)
    primary_name = primary_model.get("name")
    selection_reason = reason

    if not selection_reason:
        if task == "presentation":
            selection_reason = "Presentation task matched the dedicated presentation model."
        elif task == "image":
            selection_reason = "Image task matched the image-capable model."
        else:
            selection_reason = (
                f"Text task selected {primary_name} based on task complexity, style, and runtime estimates."
            )

    return {
        "task": task,
        "selected_model": primary_name,
        "selection_reason": selection_reason,
        "model_details": primary_model,
        "fallback_models": unique_names(fallbacks),
        "requirements": requirements,
        "dynamic_metrics": dynamic_metrics,
        "generated_at": "sync"
    }


def main() -> None:
    try:
        models = load_json(MODELS_FILE)
        dynamic_metrics = load_json(DYNAMIC_METRICS_FILE)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not isinstance(models, list):
        raise SystemExit("Expected the model registry to be a list of model definitions.")
    if not isinstance(dynamic_metrics, dict):
        raise SystemExit("Expected the dynamic metrics payload to be an object.")

    if not all(isinstance(model, dict) for model in models):
        raise SystemExit("All model entries must be objects.")

    requirements = dynamic_metrics.get("requirements", {})
    if not isinstance(requirements, dict):
        raise SystemExit("Expected requirements inside the dynamic metrics payload to be an object.")

    task = infer_task(requirements, dynamic_metrics)
    selection_payload = build_selection_payload(task, requirements, dynamic_metrics, models)
    save_json(OUTPUT_FILE, selection_payload)

    print(f"Selected model: {selection_payload['selected_model']}")
    print(f"Saved selection to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
