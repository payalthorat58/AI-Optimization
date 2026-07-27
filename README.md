# AI-Optimization
Prerequisite for Prompt Optimization

Open the project folder:

Open VS Code

Open Terminal:

Terminal → New Terminal

1. Check Python
python --version
2. Check Ollama
ollama --version

If this doesn't work, install Ollama from:

https://ollama.com/download

3. Download Granite model (only once)
ollama pull granite4.1:8b

Wait until it finishes downloading.

4. Install Python dependency (only once)
pip install ollama

or

python -m pip install ollama
5. Go to your project

If you're not already there:

cd AI-Optimization
6. Go to the orchestrator folder
cd orchestrator
7. Run your file
python prompt_optimizer.py

Done ✅