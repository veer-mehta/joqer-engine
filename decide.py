import json, os, ollama
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEM_API_KEY"))

with open("state.json", "r", encoding="utf-8") as f:
    state = json.load(f)

hand_size = len(state.get("hand", []))

prompt = f"""
You are an expert Balatro player, a POKER-inspired roguelike deck-builder.

Your task is to choose the BEST action for the current hand.

Return ONLY valid JSON.
DO NOT use markdown.
DO NOT use ``` fences.
DO NOT add any text outside JSON.

The JSON MUST match EXACTLY this format:

{{
  "action": "play" | "discard",
  "card_indexes": [i0, i1, i2, i3, i4],
  "reasoning": "short explanation"
}}

Rules:
- "card_indexes" MUST ALWAYS contain EXACTLY 5 integers
- Valid indexes are 0-based and refer to state["hand"]
- Valid range is 0 to {hand_size - 1} inclusive (hand has {hand_size} cards)
- If fewer than 5 cards are selected, pad remaining slots with -1
- NEVER invent indexes that do not exist
- NEVER repeat an index
- You MAY choose "discard" to exchange cards for better ones using state["unused_discards"]
- If state["unused_discards"] is 0, you MUST play — do not discard

Key state fields:
- hand:             your current cards (0-indexed)
- deck_remaining:   cards still in the draw pile
- hands_left:       hands remaining this round (play these or lose)
- unused_discards:  discards remaining (0 = cannot discard)
- chips_required:   chips needed to beat the blind
- chips_scored:     chips accumulated so far this round
- poker_hands:      chip/mult/level for each hand type

Examples:
Play 3 cards (Three of a Kind):
{{ "action": "play", "card_indexes": [0, 1, 2, -1, -1], "reasoning": "Three Aces form Three of a Kind" }}
Discard 3 cards to chase a Flush:
{{ "action": "discard", "card_indexes": [2, 5, 6, -1, -1], "reasoning": "Discard off-suit cards to draw toward Flush" }}

Poker Hands (lowest to highest):
- High Card, Pair, Two Pair, Three of a Kind
- Straight, Flush, Full House, Four of a Kind
- Straight Flush, Flush House, Five of a Kind, Flush Five

Key mechanics:
- You may discard ANY number of cards (1-5) to draw new ones
- Discarding preserves unselected cards in hand
- Playing a weak hand early wastes a hand — consider discarding for higher EV
- Higher-level hands scale better than raw chips
- If chips_scored is already close to chips_required, play safe

STATE:
{json.dumps(state, indent=2)}
"""

# response = ollama.chat(model='mistral', messages=[{"role":"user", "content":prompt}])
# text = response["message"]["content"].strip()

response = client.models.generate_content(model="models/gemini-2.5-flash", contents=prompt)
text = response.text.strip()

try:
    decision = json.loads(text)
except:
    decision = { "error": "invalid", "raw": text }

with open("decision.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)
    print(decision)
