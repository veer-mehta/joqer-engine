import json
import numpy as np
import torch
import itertools
from os import path

from decision_engine.agent.dqn import DQN
from decision_engine.logic.discard_strats import apply_strategy
from decision_engine.logic.evaluator import evaluate_hand, HAND_SCORES, card_chips
from decision_engine.utils.encoding import encode_state


MOD_PATH = r"./Mods/JoQerEngine/"
MODEL_PATH = path.join(MOD_PATH, "apdqn.pth")
ROUND_STATE_PATH = path.join(MOD_PATH, "round_state.json")
DECISION_PATH = path.join(MOD_PATH, "decision.json")


SUIT_MAP = {"Spades": 0, "Hearts": 1, "Diamonds": 2, "Clubs": 3}


def convert_rank(rank):
	return 0 if rank == 14 else rank - 1


def best_hand_indices(hand):

	best_score = -1
	best_indices = []

	for comb in itertools.combinations(range(len(hand)), 5):

		cards = [hand[i] for i in comb]

		hand_type = evaluate_hand(cards)
		base_chips, mult = HAND_SCORES[hand_type]
		card_sum = sum(card_chips(r) for r, _ in cards)

		score = mult * (base_chips + card_sum)

		if score > best_score:

			best_score = score
			best_indices = list(comb)

	return best_indices



def get_discard_indices(old_hand, new_hand):

	removed = []
	used = [False] * len(new_hand)

	for i, card in enumerate(old_hand):

		found = False

		for j, nc in enumerate(new_hand):

			if not used[j] and nc == card:

				used[j] = True
				found = True
				break

		if not found:

			removed.append(i)

	return removed



# ----------------------------
# Load model 
# ----------------------------
model = DQN(70, 6)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()




# ----------------------------
# Load state
# ----------------------------
with open(ROUND_STATE_PATH, "r") as f:
	state_json = json.load(f)



# ----------------------------
# Convert state
# ----------------------------
hand = []

for card in state_json["hand"]:

	hand.append((convert_rank(card["rank"]), SUIT_MAP[card["suit"]]))

discards_remaining = state_json.get("unused_discards", 0)



# ----------------------------
# Force play if no discards
# ----------------------------
if discards_remaining == 0:

	decision = {
		"action": "play",
		"card_indexes": best_hand_indices(hand)
	}

	with open(DECISION_PATH, "w") as f:
		json.dump(decision, f)

	exit()



# ----------------------------
# Encode state
# ----------------------------
state_vec = encode_state(hand, discards_remaining)
state_tensor = torch.tensor(state_vec, dtype=torch.float32)



# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():

	q_values = model(state_tensor)
	q_values_np = q_values.numpy()

	action = int(np.argmax(q_values_np))

	print("Q-values:", q_values_np)
	print("Chosen action:", action)



# ----------------------------
# Safety: prevent invalid discard
# ----------------------------
if action != 0 and discards_remaining == 0:

	action = 0



# ----------------------------
# Decision
# ----------------------------
decision = {
	"action": "play" if action == 0 else "discard",
	"card_indexes": []
}



if action != 0:

	new_hand = apply_strategy(hand, action)

	removed = get_discard_indices(hand, new_hand)

	decision["card_indexes"] = removed



# ----------------------------
# Save
# ----------------------------
with open(DECISION_PATH, "w") as f:

	json.dump(decision, f)