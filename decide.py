import json
import numpy as np
import torch
import itertools
from os import path
from decision_engine.agent.dqn import DQN
from decision_engine.logic.discard_strats import get_discard_indices
from decision_engine.logic.evaluator import score_combination, best_hand
from decision_engine.utils.encoding import encode_state

MOD_PATH = "./Mods/JoQerEngine/"
MODEL_PATH = path.join(MOD_PATH, "apdqn.pth")
ROUND_STATE_PATH = path.join(MOD_PATH, "round_state.json")
DECISION_PATH = path.join(MOD_PATH, "decision.json")

SUIT_MAP = {"Spades": 0, "Hearts": 1, "Diamonds": 2, "Clubs": 3}
FULL_HAND_TYPES = {"straight", "flush", "straight_flush", "full_house"}


def find_best_play(hand):
    best_score = -1
    best_comb = None
    best_type = "high_card"

    for comb in itertools.combinations(range(len(hand)), 5):
        cards = [hand[i] for i in comb]
        hand_type, score, _ = score_combination(cards)

        if score > best_score:
            best_score = score
            best_comb = comb
            best_type = hand_type

    if best_type in FULL_HAND_TYPES:
        return list(best_comb)

    cards = [hand[i] for i in best_comb]
    _, _, contributing = score_combination(cards)
    return [i for i, (r, _) in zip(best_comb, cards) if r in contributing]


model = DQN(77, 6)
if path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    except Exception as e:
        print(f"Warning: Failed to load {MODEL_PATH} ({e}). Retrain using train.py.")
model.eval()

with open(ROUND_STATE_PATH, "r") as f:
    game_state = json.load(f)

hand = []
for card in game_state["hand"]:
    rank = 0 if card["rank"] == 14 else card["rank"] - 1
    hand.append((rank, SUIT_MAP[card["suit"]]))

discards_left = game_state.get("unused_discards", 0)
hands_left = game_state.get("hands_left", 4)
chips = game_state.get("chips", 0)
blind_chips = game_state.get("blind_chips", 300)
needed = max(0, blind_chips - chips)
curr_score = best_hand(hand)

if discards_left <= 0 or curr_score >= 150 or curr_score >= needed:
    action = 0
else:
    state = torch.tensor(encode_state(hand, discards_left, hands_left, chips, blind_chips, curr_score), dtype=torch.float32)
    with torch.no_grad():
        q_values = model(state).numpy()
    action = int(np.argmax(q_values))

if action == 0:
    decision = {"action": "play", "card_indexes": find_best_play(hand)}
else:
    decision = {"action": "discard", "card_indexes": get_discard_indices(hand, action)}

with open(DECISION_PATH, "w") as f:
    json.dump(decision, f)