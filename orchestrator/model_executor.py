import json
import os
import sys

# ------------------------------------------------------------------------------
# Resolve all paths relative to this file's directory.
# ------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Load standard library dotenv or try fallback
# ── CONFIG FLAG ──────────────────────────────────────────────────────────────
# Set GEMINI_ENABLED=false in orchestrator/.env to skip Gemini and go directly
# to the Groq fallback. Useful when the Gemini key has quota issues.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Will try to read .env manually.")
    load_dotenv = None

# Optional SDKs
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import requests
except ImportError:
    requests = None

class ModelExecutor:
    """
    Step 5 -- Model Executor
    
    Reads selected_model.json and optimized_prompt.json, then executes
    the request against the chosen model API.
    """

    SELECTED_MODEL_PATH = os.path.join(_HERE, "selected_model.json")
    PROMPT_PATH = os.path.join(_HERE, "optimized_prompt.json")
    OUTPUT_PATH = os.path.join(_HERE, "final_output.json")
    ENV_PATH = os.path.join(_HERE, ".env")

    def __init__(self):
        self.routing = {}
        self.prompt_data = {}
        self._load_data()
        self._load_env()

    def _load_data(self):
        with open(self.SELECTED_MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.routing = data.get("routing_decision", {})
            self.provider = data.get("provider", "")
            
        with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
            self.prompt_data = json.load(f)

    def _load_env(self):
        if load_dotenv:
            load_dotenv(self.ENV_PATH, override=True)
        else:
            if os.path.exists(self.ENV_PATH):
                with open(self.ENV_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            key, val = line.split("=", 1)
                            os.environ[key.strip()] = val.strip()

    def call_gemini(self, model_name: str, prompt: str, temperature: float) -> str:
        """Call Gemini API using standard HTTP requests to avoid SDK dependencies.
        
        Tries models in order: gemini-2.0-flash -> gemini-1.5-flash
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        if not requests:
            raise ImportError("The 'requests' package is required. Run: pip install requests")

        # Model fallback chain: try newer first, then stable fallback
        models_to_try = []
        if model_name in ("gemini-2.5-flash", "gemini-2.0-flash"):
            models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
        else:
            models_to_try = [model_name, "gemini-1.5-flash"]

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature}
        }
        headers = {"Content-Type": "application/json"}

        last_error = None
        for candidate in models_to_try:
            actual_model = candidate if candidate.startswith("models/") else f"models/{candidate}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{actual_model}:generateContent?key={api_key}"
            
            print(f"  -> Trying Gemini model: {actual_model} ...")
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                try:
                    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    raise RuntimeError(f"Unexpected response format from Gemini: {response.json()}") from e
            else:
                last_error = f"Gemini API Error {response.status_code}: {response.text}"
                print(f"     [!] {candidate} failed ({response.status_code}) - trying next...")

        raise RuntimeError(last_error)

    def call_groq(self, model_name: str, prompt: str, temperature: float) -> str:
        """Call Groq API (Llama) using the groq SDK."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        if not Groq:
            raise ImportError("The 'groq' package is required. Run: pip install groq")
            
        # Determine exact model ID
        actual_model = "llama-3.3-70b-versatile" if model_name == "llama-3.3-70b" else model_name
        
        print(f"  -> Sending request to Groq API ({actual_model})...")
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=actual_model,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
        
    def call_ollama(self, model_name: str, prompt: str, temperature: float) -> str:
        """Call Ollama locally for Granite."""
        if not requests:
            raise ImportError("The 'requests' package is required. Run: pip install requests")
            
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        print(f"  -> Sending request to local Ollama ({model_name})...")
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama Error {response.status_code}: {response.text}")
            
        return response.json().get("response", "")

    def _try_groq_fallback(self, full_prompt: str, temperature: float) -> str:
        """Fallback to Groq/Llama when primary provider fails due to quota."""
        print("\n  [FALLBACK] Primary provider failed. Falling back to Groq (llama-3.3-70b-versatile)...")
        return self.call_groq("llama-3.3-70b-versatile", full_prompt, temperature)

    def run(self):
        model_name = self.routing.get("model_to_call")
        temperature = self.routing.get("temperature", 0.5)
        
        sys.stdout.write("\n=== Executing Model Pipeline ===\n")
        sys.stdout.write(f"  Selected Model : {model_name}\n")
        sys.stdout.write(f"  Provider       : {self.provider}\n")
        sys.stdout.write(f"  Temperature    : {temperature}\n\n")

        # Combine document and prompt
        doc_text = self.prompt_data.get("document_text", "")
        task_prompt = self.prompt_data.get("optimized_prompt", "")
        full_prompt = f"Document/Context:\n{doc_text}\n\nTask:\n{task_prompt}"

        actual_model = model_name
        actual_provider = self.provider

        # Respect the GEMINI_ENABLED flag from .env
        gemini_enabled = os.getenv("GEMINI_ENABLED", "true").strip().lower() != "false"

        try:
            if self.provider.lower() == "google":
                if not gemini_enabled:
                    print("  [INFO] GEMINI_ENABLED=false — skipping Gemini, using Groq directly.")
                    result = self._try_groq_fallback(full_prompt, temperature)
                    actual_model = "llama-3.3-70b-versatile"
                    actual_provider = "groq (direct)"
                else:
                    try:
                        result = self.call_gemini(model_name, full_prompt, temperature)
                    except RuntimeError as e:
                        # Auto-fallback on quota exhaustion (429) or unavailability (404)
                        if "429" in str(e) or "404" in str(e):
                            result = self._try_groq_fallback(full_prompt, temperature)
                            actual_model = "llama-3.3-70b-versatile"
                            actual_provider = "groq (fallback)"
                        else:
                            raise
            elif self.provider.lower() == "groq":
                result = self.call_groq(model_name, full_prompt, temperature)
            elif self.provider.lower() == "ollama":
                result = self.call_ollama(model_name, full_prompt, temperature)
            else:
                raise ValueError(f"Unknown provider '{self.provider}' for model '{model_name}'")

            print("\n=== Execution Successful! ===\n")
            print(f"  Model used: {actual_model}  (provider: {actual_provider})")

            output_payload = {
                "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_used": actual_model,
                "provider": actual_provider,
                "temperature": temperature,
                "input_prompt": full_prompt,
                "generated_response": result
            }

            with open(self.OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=4)

            print(f"  Response saved to: {self.OUTPUT_PATH}")
            print("\n--- Output Preview ---")
            print(result[:400] + "..." if len(result) > 400 else result)
            print("----------------------\n")

        except Exception as e:
            print(f"\n=== Execution Failed: {str(e)} ===")
            raise

if __name__ == "__main__":
    from datetime import datetime, timezone
    executor = ModelExecutor()
    executor.run()
