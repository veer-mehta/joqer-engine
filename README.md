Here is a **small, clean README** with **Installation** and **Quick Start**, tailored to your Balatro + LLM AutoPlayer project.

---

# AutoPlayer (a Balatro Mod)

AutoPlayer is a Balatro mod that captures the current game state using Lovely Injector and Steamodded, sends it to a local(Ollama) or cloud(Gemini) LLM, and automatically plays or discards cards based on the model’s decision.

The mod runs fully in-game using Lua, with a Python bridge for LLM inference.

---

## Features

* Extracts full game state (hand, poker hands, jokers, blind, economy)
* Supports **play** and **discard** decisions
* Works with **local models** (Ollama: DeepSeek, Mistral, Qwen, etc.)

---

## Installation

### Requirements

* Balatro
* Steamodded + Lovely Injector installed
* Python 3.9+
* Optional: Ollama for local LLMs

### Steps

1. Clone or copy this mod into:

   ```
   %APPDATA%\Balatro\Mods\AutoPlayer
   ```

2. Create a Python virtual environment:

   ```
   python -m venv apenv
   apenv\Scripts\activate
   pip install ollama google-genai dkjson
   ```

3. Place `dkjson.lua` inside the AutoPlayer mod folder.

4. Ensure `decide.py` is present in:

   ```
   Mods/AutoPlayer/decide.py
   ```

5. Launch Balatro normally (Steamodded will auto-load the mod).
