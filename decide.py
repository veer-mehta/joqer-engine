import json, os, ollama
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEM_API_KEY"))

with open("state.json", "r", encoding="utf-8") as f:
    state = json.load(f)

prompt = f"""
You are an expert Balatro player a POKER inspired rougelike deck-builder.

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
- Valid indexes are 0-based and refer to state.hand
- Valid range is 0 to 7 inclusive
- If fewer than 5 cards are selected, pad remaining slots with -1
- NEVER invent indexes that do not exist
- NEVER repeat an index

Examples:
Play 3 cards:
{{ "action": "play", "card_indexes": [0, 1, 2, -1, -1], "reasoning": "Three Aces form Three of a Kind" }}
Discard 2 cards:
{{ "action": "discard", "card_indexes": [2, 5, 6, 7, -1], "reasoning": "Discard cards to chase full house" }}

Poker Hands:
- High Card: if none of the above hands play
- Pair: 2 cards of same rank
- Two Pair: 2 pairs of 2 cards of same rank
- Three of a Kind: 3 cards of same rank
- Straight: 5 cards of consecutive ranks
- Flush: 5 cards of same suit
- Full House: Three of a Kind of one rank AND Pair of another
- Four of a Kind: 4 cards of same rank
- Straight Flush: Straight AND same suit
- Flush House: Full House AND same suit
- Five of a Kind: 5 cards of same rank
- Flush Five: 5 cards of same rank AND same suit

Key mechanics:
- You may discard ANY number of cards (0–5)
- Discarding preserves unselected cards
- Drawing new cards can improve hand potential
- Playing a weak hand too early often loses EV
- Higher-level hands scale better than raw chips

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
