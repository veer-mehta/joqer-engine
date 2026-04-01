"""
decide_bc.py — Drop-in replacement for decide.py using the trained BC model.

Usage (same as decide.py — called by main.lua via os.execute):
    python Mods/AutoPlayer/decide_bc.py

Reads:  state.json   (written by main.lua)
Writes: decision.json (read by main.lua)

Output schema (identical to Gemini decide.py):
    {
      "action": "play" | "discard",
      "card_indexes": [i0, i1, i2, i3, i4],   # 5 elements, -1 = unused
      "reasoning": "..."                        # optional debug info
    }

To switch the mod from Gemini to the BC model, change ONE line in main.lua:
    - python Mods/AutoPlayer/decide.py
    + python Mods/AutoPlayer/decide_bc.py
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

# ── Resolve paths relative to this script (works from any cwd) ────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_WEIGHTS = SCRIPT_DIR / "model.pt"
MODEL_CONFIG  = SCRIPT_DIR / "model_config.json"

# State / decision files are written to LÖVE's working dir (game root)
# main.lua uses love.filesystem which maps to the game's save dir;
# os.execute runs from the game's working directory, so plain filenames work.
STATE_PATH    = Path("state.json")
DECISION_PATH = Path("decision.json")

# Ablation flag: set env var BALATRO_BC_NO_DECK=1 to zero out deck features
USE_DECK = os.environ.get("BALATRO_BC_NO_DECK", "0") != "1"

# ── Imports from the mod directory ─────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from encode import encode_state, INPUT_DIM
from model  import load_model


# ── Load model (once per process) ─────────────────────────────────────────────
def _load():
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"model.pt not found at {MODEL_WEIGHTS}\n"
            "Run train.py first to generate the model."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(str(MODEL_WEIGHTS), str(MODEL_CONFIG), device=device)
    return model, device

model, device = _load()


# ── Read state.json ────────────────────────────────────────────────────────────
if not STATE_PATH.exists():
    json.dump({"error": "state.json not found"}, open(DECISION_PATH, "w"))
    sys.exit(1)

with open(STATE_PATH, encoding="utf-8") as f:
    state = json.load(f)


# ── Encode & infer ─────────────────────────────────────────────────────────────
t0 = time.perf_counter()

x = torch.from_numpy(encode_state(state, use_deck=USE_DECK)).unsqueeze(0).to(device)
hand_size = len(state.get("hand", []))

action, card_indexes = model.predict(x, hand_size=hand_size)

elapsed_ms = (time.perf_counter() - t0) * 1000

# ── Build reasoning string (debug, optional) ──────────────────────────────────
with torch.no_grad():
    action_logits, card_logits = model(x)
    action_probs = torch.softmax(action_logits[0], dim=0).tolist()
    if hand_size > 0:
        card_probs = torch.sigmoid(card_logits[0, :hand_size]).tolist()
        top3 = sorted(enumerate(card_probs), key=lambda kv: kv[1], reverse=True)[:3]
        card_debug = ", ".join(f"card[{i}]={p:.2f}" for i, p in top3)
    else:
        card_debug = "no cards"

reasoning = (
    f"BC model | {elapsed_ms:.1f}ms | "
    f"play={action_probs[0]:.2f} discard={action_probs[1]:.2f} | "
    f"top cards: {card_debug}"
    + (" | [hand-only mode]" if not USE_DECK else "")
)

# ── Write decision.json ────────────────────────────────────────────────────────
decision = {
    "action":       action,
    "card_indexes": card_indexes,
    "reasoning":    reasoning,
}

with open(DECISION_PATH, "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)

print(decision)
