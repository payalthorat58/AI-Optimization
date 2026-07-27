import json
import os
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------------------
# Resolve all paths relative to this file's directory, NOT the caller's cwd.
# This eliminates the need for os.chdir() and makes the module safely importable.
# ------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------
# Reasoning level ordering (low -> high capability)
# ------------------------------------------------------------------------------
REASONING_RANK: Dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
}

VALID_TASK_STYLES = {"deterministic", "balanced", "creative"}


class ModelSelectionError(RuntimeError):
    """Raised when model selection cannot produce a result."""


class ModelSelector:
    """
    Step 4b -- Intelligent Model Selection Engine.

    Pipeline position:
        dynamic_metrics.json       --+
                                     +--> ModelSelector --> selected_model.json
        models/static_metrics.json --+

    Usage:
        selector = ModelSelector()
        report   = selector.run(privacy_required=False, compliance_required=False)
    """

    # File paths resolved at class definition time (relative to this file)
    STATIC_METRICS_PATH: str = os.path.join(_HERE, "models", "static_metrics.json")
    DYNAMIC_METRICS_PATH: str = os.path.join(_HERE, "dynamic_metrics.json")
    OUTPUT_PATH: str = os.path.join(_HERE, "selected_model.json")

    def __init__(self) -> None:
        """Load static_metrics.json and dynamic_metrics.json into memory."""
        self.static_metrics: List[Dict] = []
        self.dynamic_metrics: Dict = {}
        # Tracks which models were excluded and why (populated by apply_constraints)
        self._exclusion_log: Dict[str, str] = {}
        self._load_metrics()

    # --------------------------------------------------------------------------
    # 1. Data Loading  (FIX #6 -- proper error handling)
    # --------------------------------------------------------------------------

    def _load_metrics(self) -> None:
        """
        Load both metric files with clear, user-friendly error messages.

        Raises:
            FileNotFoundError: if either JSON file is missing.
            ValueError:        if either JSON file is malformed / not valid JSON.
        """
        for label, path, attr in [
            ("static metrics", self.STATIC_METRICS_PATH, "static_metrics"),
            ("dynamic metrics", self.DYNAMIC_METRICS_PATH, "dynamic_metrics"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {label} file: {path}\n"
                    f"Ensure the previous pipeline steps have run successfully."
                )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {label} file: {path}\n"
                    f"Details: {exc}"
                ) from exc

    # --------------------------------------------------------------------------
    # 2. Constraint Validation
    # --------------------------------------------------------------------------

    def validate_context_window(self, model: Dict,
                                total_tokens: Optional[int] = None) -> bool:
        """
        Ensure the model's max_context can handle the request.

        FIX #9: total_tokens is now optional -- if omitted, the method computes
        it from self.dynamic_metrics so it can be called standalone.

        Args:
            model:        A single entry from static_metrics.json.
            total_tokens: Override for the token count; computed automatically if None.

        Returns:
            True if the model can handle the token count, False otherwise.
        """
        if total_tokens is None:
            total_tokens = (
                self.dynamic_metrics["estimated_input_tokens"]
                + self.dynamic_metrics["estimated_output_tokens"]
            )
        return model["max_context"] >= total_tokens

    def validate_constraints(self, model: Dict,
                             privacy_required: bool = False,
                             compliance_required: bool = False) -> Tuple[bool, str]:
        """
        Check context-window, privacy, and compliance hard constraints.

        Args:
            model:               Static metric entry for the candidate model.
            privacy_required:    Whether the task demands on-prem / private execution.
            compliance_required: Whether the task demands compliance support.

        Returns:
            (passed: bool, reason: str)  -- reason is "passed" when all checks pass.
        """
        total_tokens = (
            self.dynamic_metrics["estimated_input_tokens"]
            + self.dynamic_metrics["estimated_output_tokens"]
        )

        # Hard constraint 1 -- context window
        if not self.validate_context_window(model, total_tokens):
            return False, (
                f"Context window too small: needs {total_tokens} tokens "
                f"but {model['name']} supports only {model['max_context']}"
            )

        # Hard constraint 2 -- privacy
        if privacy_required and not model.get("supports_privacy", False):
            return False, f"{model['name']} does not support private/local execution"

        # Hard constraint 3 -- compliance
        if compliance_required and not model.get("supports_compliance", False):
            return False, f"{model['name']} does not meet compliance requirements"

        return True, "passed"

    def apply_constraints(self, privacy_required: bool = False,
                          compliance_required: bool = False) -> List[Dict]:
        """
        Filter the full model list to only those that pass all hard constraints.

        FIX #4: Records every excluded model and reason in self._exclusion_log
        so the constraints summary can report accurately.

        Returns:
            List of eligible static-metric dicts.
        """
        self._exclusion_log.clear()
        eligible: List[Dict] = []

        for model in self.static_metrics:
            passed, reason = self.validate_constraints(
                model, privacy_required, compliance_required
            )
            if passed:
                eligible.append(model)
            else:
                self._exclusion_log[model["name"]] = reason
                print(f"  [EXCLUDED] {model['name']}: {reason}")

        return eligible

    # --------------------------------------------------------------------------
    # 3. Capability Scoring
    # --------------------------------------------------------------------------

    def _capability_score(self, model: Dict) -> float:
        """
        Map (task_complexity, model reasoning_level) to a 0--1 capability score.

        Rules (from plan.md):
          - high complexity   -> prefer very_high reasoning
          - medium complexity -> prefer high reasoning
          - low complexity    -> medium reasoning is sufficient; speed matters more
        """
        complexity = self.dynamic_metrics.get("task_complexity", "low")
        model_rank = REASONING_RANK.get(model.get("reasoning_level", "low"), 1)

        target_rank = {
            "high":   REASONING_RANK["very_high"],
            "medium": REASONING_RANK["high"],
        }.get(complexity, REASONING_RANK["medium"])  # default -> low complexity

        if model_rank >= target_rank:
            # Small bonus for over-qualifying, capped at 1.0
            bonus = (model_rank - target_rank) * 0.05
            return min(1.0, 0.85 + bonus)
        # Proportional penalty for under-qualifying
        return model_rank / target_rank * 0.85

    # --------------------------------------------------------------------------
    # 4. Weighted Scoring
    # --------------------------------------------------------------------------

    def _determine_weights(self) -> Tuple[float, float, float]:
        """
        Return (latency_weight, cost_weight, capability_weight) based on task_style.

        From plan.md:
          deterministic -> latency=0.5, cost=0.3, capability=0.2
          balanced      -> latency=0.3, cost=0.4, capability=0.3
          creative      -> latency=0.2, cost=0.2, capability=0.6

        FIX #8: Warns if task_style is an unrecognised value instead of silently
        defaulting, making debugging much easier.
        """
        style = self.dynamic_metrics.get("task_style", "balanced")

        if style not in VALID_TASK_STYLES:
            warnings.warn(
                f"Unknown task_style '{style}'. "
                f"Expected one of {sorted(VALID_TASK_STYLES)}. "
                f"Defaulting to 'balanced' weights.",
                UserWarning,
                stacklevel=2,
            )
            style = "balanced"

        weight_map = {
            "deterministic": (0.5, 0.3, 0.2),
            "balanced":      (0.3, 0.4, 0.3),
            "creative":      (0.2, 0.2, 0.6),
        }
        return weight_map[style]

    def score_models(self, candidates: List[Dict]) -> List[Dict]:
        """
        Score each candidate using a normalised 3-factor weighted formula.

        Steps (from plan.md):
          1. Normalise cost    -> higher score = cheaper model
          2. Normalise latency -> higher score = faster model
          3. Compute capability score via _capability_score()
          4. Weighted sum: final = lat_w*lat + cost_w*cost + cap_w*cap

        FIX #1: Removed the redundant post-sort swap.
        The sort key already handles tie-breaking correctly by sorting on
        (-final_score, raw_cost), which puts the cheaper model first whenever
        two models have the same score -- no additional swap needed.

        Returns:
            List of scored dicts sorted best-first.
        """
        if not candidates:
            return []

        costs = {
            m["name"]: self.dynamic_metrics["estimated_cost"].get(m["name"], 99)
            for m in candidates
        }
        latencies = {
            m["name"]: self.dynamic_metrics["estimated_latency"].get(m["name"], 99)
            for m in candidates
        }

        cost_values    = list(costs.values())
        latency_values = list(latencies.values())

        min_cost, max_cost = min(cost_values), max(cost_values)
        min_lat,  max_lat  = min(latency_values), max(latency_values)

        lat_w, cost_w, cap_w = self._determine_weights()

        scored: List[Dict] = []
        for model in candidates:
            name = model["name"]

            # Normalise: lower raw value -> higher score
            cost_score = (
                1.0
                if max_cost == min_cost
                else 1.0 - (costs[name] - min_cost) / (max_cost - min_cost)
            )
            lat_score = (
                1.0
                if max_lat == min_lat
                else 1.0 - (latencies[name] - min_lat) / (max_lat - min_lat)
            )
            cap_score = self._capability_score(model)

            final = (lat_w * lat_score) + (cost_w * cost_score) + (cap_w * cap_score)

            scored.append({
                "model":            model,
                "latency_score":    round(lat_score,  4),
                "cost_score":       round(cost_score, 4),
                "capability_score": round(cap_score,  4),
                "latency_weight":   lat_w,
                "cost_weight":      cost_w,
                "capability_weight": cap_w,
                "final_score":      round(final, 4),
                "raw_cost":         costs[name],
                "raw_latency":      latencies[name],
            })

        # Sort best-first; tie-break by raw cost (cheaper wins) -- Rule 5 of plan.md
        scored.sort(key=lambda x: (-x["final_score"], x["raw_cost"]))
        return scored

    # --------------------------------------------------------------------------
    # 5. Best Model Selection
    # --------------------------------------------------------------------------

    def select_best_model(self, scored: List[Dict]) -> Dict:
        """
        Return the winner and ranked alternatives from the scored list.

        Returns:
            {"winner": <scored_entry>, "alternatives": [<scored_entry>, ...]}

        Raises:
            ModelSelectionError: if no eligible models remain.
        """
        if not scored:
            raise ModelSelectionError(
                "No eligible models remain after constraint filtering. "
                "Review privacy/compliance requirements or add more models."
            )
        return {"winner": scored[0], "alternatives": scored[1:]}

    # --------------------------------------------------------------------------
    # 6. Reason Generation
    # --------------------------------------------------------------------------

    def _build_winner_reason(self, winner: Dict) -> str:
        """Compose a human-readable selection reason for the winning model."""
        latency   = winner["raw_latency"]
        reasoning = winner["model"]["reasoning_level"]
        complexity = self.dynamic_metrics.get("task_complexity", "low")
        style      = self.dynamic_metrics.get("task_style", "deterministic")

        return (
            f"Optimal balance of latency ({latency}s) and cost efficiency "
            f"for {complexity}-complexity {style} task. "
            f"Sufficient reasoning capability ({reasoning}) for the requested task."
        )

    def _build_alternative_reason(self, entry: Dict) -> str:
        """
        Compose a concise reason for each runner-up model.

        FIX #5: Score is no longer embedded inside the reason string --
        it already exists as the sibling 'score' key in the JSON output.
        """
        model     = entry["model"]
        latency   = entry["raw_latency"]
        reasoning = model["reasoning_level"]
        privacy   = model.get("supports_privacy", False)

        parts: List[str] = []
        if privacy:
            parts.append("Privacy-compliant local execution")

        reasoning_descriptions = {
            "very_high": f"Highest reasoning capability but slowest ({latency}s). "
                         f"Overkill for a deterministic task",
            "medium":    f"Lightweight local model; slightly slower ({latency}s)",
        }
        parts.append(
            reasoning_descriptions.get(
                reasoning, f"Reasoning level: {reasoning}, latency: {latency}s"
            )
        )
        return ". ".join(parts)

    # --------------------------------------------------------------------------
    # 7. Constraint Summary  (FIX #4 -- accurately reflects filter results)
    # --------------------------------------------------------------------------

    def _constraints_summary(self, privacy_required: bool,
                              compliance_required: bool) -> Dict:
        """
        Build the constraints_applied block for the output JSON.

        FIX #4: Reports per-constraint outcome accurately -- distinguishes
        'passed (all models)' from 'passed (N excluded)' so the report is
        never misleadingly optimistic.
        """
        total_tokens = (
            self.dynamic_metrics["estimated_input_tokens"]
            + self.dynamic_metrics["estimated_output_tokens"]
        )

        # Were any models excluded due to context window?
        ctx_exclusions = [
            name for name, reason in self._exclusion_log.items()
            if "Context window" in reason
        ]
        if ctx_exclusions:
            ctx_status = f"1 or more models excluded (too small): {ctx_exclusions}"
        else:
            ctx_status = f"passed (all models fit; total tokens needed: {total_tokens})"

        # Privacy
        if not privacy_required:
            privacy_status = "not_required"
        else:
            pv_exclusions = [
                n for n, r in self._exclusion_log.items() if "privacy" in r.lower()
            ]
            privacy_status = (
                f"applied; {len(pv_exclusions)} model(s) excluded: {pv_exclusions}"
                if pv_exclusions
                else "applied; all models passed"
            )

        # Compliance
        if not compliance_required:
            compliance_status = "not_required"
        else:
            co_exclusions = [
                n for n, r in self._exclusion_log.items() if "compliance" in r.lower()
            ]
            compliance_status = (
                f"applied; {len(co_exclusions)} model(s) excluded: {co_exclusions}"
                if co_exclusions
                else "applied; all models passed"
            )

        return {
            "context_window_check": ctx_status,
            "privacy_filter":       privacy_status,
            "compliance_filter":    compliance_status,
        }

    # --------------------------------------------------------------------------
    # 8. Report Generation & File Save
    # --------------------------------------------------------------------------

    def generate_selection_report(self, winner: Dict, alternatives: List[Dict],
                                  privacy_required: bool = False,
                                  compliance_required: bool = False) -> Dict:
        """
        Build the full selected_model.json payload.

        Matches the exact schema defined in plan.md.
        """
        w_model = winner["model"]

        return {
            "selected_model": w_model["name"],
            "provider":       w_model["provider"],
            "selection_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "selection_reason": self._build_winner_reason(winner),
            "selection_score":  winner["final_score"],
            "scoring_breakdown": {
                "latency_score":     winner["latency_score"],
                "cost_score":        winner["cost_score"],
                "capability_score":  winner["capability_score"],
                "latency_weight":    winner["latency_weight"],
                "cost_weight":       winner["cost_weight"],
                "capability_weight": winner["capability_weight"],
            },
            "alternatives": [
                {
                    "rank":   idx + 2,
                    "model":  alt["model"]["name"],
                    "score":  alt["final_score"],
                    "reason": self._build_alternative_reason(alt),  # FIX #5
                }
                for idx, alt in enumerate(alternatives)
            ],
            "constraints_applied": self._constraints_summary(
                privacy_required, compliance_required
            ),
            "routing_decision": {
                "model_to_call":          w_model["name"],
                "temperature":            self.dynamic_metrics.get("temperature", 0.5),
                "expected_cost":          winner["raw_cost"],
                "expected_latency_seconds": winner["raw_latency"],
            },
        }

    def save_selection(self, report: Dict,
                       output_path: Optional[str] = None) -> None:
        """
        Persist the selection report to selected_model.json.

        Args:
            report:      Dict returned by generate_selection_report().
            output_path: Override the default OUTPUT_PATH if needed.
        """
        path = output_path or self.OUTPUT_PATH
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    # --------------------------------------------------------------------------
    # 9. Main Orchestration  (FIX #7 -- no os.chdir(); paths are absolute)
    # --------------------------------------------------------------------------

    def run(self, privacy_required: bool = False,
            compliance_required: bool = False) -> Dict:
        """
        Full pipeline: load -> filter -> score -> select -> report -> save.

        No longer calls os.chdir() -- all file paths are resolved relative to
        this module's own directory, making it safe to import and call from
        any other script in the pipeline.

        Args:
            privacy_required:    Set True to enforce the privacy hard constraint.
            compliance_required: Set True to enforce the compliance hard constraint.

        Returns:
            The final report dict (also written to selected_model.json).

        Raises:
            FileNotFoundError:   if a required input file is missing.
            ValueError:          if a required input file contains invalid JSON.
            ModelSelectionError: if all models are filtered out by constraints.
        """
        print("\nLoading metrics ...")
        print(f"  Static models   : {[m['name'] for m in self.static_metrics]}")
        print(f"  Task complexity : {self.dynamic_metrics.get('task_complexity')}")
        print(f"  Task style      : {self.dynamic_metrics.get('task_style')}")

        # Step 1 -- Apply hard constraints
        print("\nApplying hard constraints ...")
        candidates = self.apply_constraints(privacy_required, compliance_required)

        if not candidates:
            raise ModelSelectionError(
                "All models were filtered out by hard constraints. "
                "Review privacy/compliance requirements or add more models."
            )
        print(f"  Eligible models : {[m['name'] for m in candidates]}")

        # Step 2 -- Score eligible candidates
        print("\nScoring models ...")
        scored = self.score_models(candidates)
        for entry in scored:
            print(
                f"  {entry['model']['name']:25s}  "
                f"score={entry['final_score']:.4f}  "
                f"(lat={entry['latency_score']:.2f}  "
                f"cost={entry['cost_score']:.2f}  "
                f"cap={entry['capability_score']:.2f})"
            )

        # Step 3 -- Pick winner + rank alternatives
        result      = self.select_best_model(scored)
        winner      = result["winner"]
        alternatives = result["alternatives"]

        # Step 4 -- Build and persist the report
        report = self.generate_selection_report(
            winner, alternatives, privacy_required, compliance_required
        )
        self.save_selection(report)

        # Step 5 -- Human-readable console summary
        print(
            f"\n[Selected Model]  {report['selected_model']}\n"
            f"  Score          : {report['selection_score']:.4f} / 1.0\n"
            f"  Reason         : {report['selection_reason']}\n"
            f"  Cost           : {report['routing_decision']['expected_cost']} "
            f"| Latency: {report['routing_decision']['expected_latency_seconds']}s "
            f"| Reasoning: {winner['model']['reasoning_level']}\n"
            f"  Output saved   : {self.OUTPUT_PATH}"
        )

        return report


# ------------------------------------------------------------------------------
# Entry point  (FIX #7 -- os.chdir() removed; _HERE handles all path resolution)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    selector = ModelSelector()
    selector.run(
        privacy_required=False,    # Set True if the task involves sensitive data
        compliance_required=False, # Set True if compliance support is required
    )
