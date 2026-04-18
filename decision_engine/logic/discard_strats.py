import random
from collections import Counter
from decision_engine.utils.cards import print_hand, random_card

def discard_low_cards(hand, num_discard=2):
	sorted_hand = sorted(hand, key=lambda x: x[0])  # sort by rank
	to_remove = sorted_hand[:num_discard]
	
	return remove_cards(hand, to_remove)


def discard_non_flush(hand):
	suits = [s for _, s in hand]
	count = Counter(suits)
	best_suit = max(count, key=count.get)
	to_keep = [card for card in hand if card[1] == best_suit]
	num_needed = len(hand) - len(to_keep)
	
	return draw_new_cards(to_keep, num_needed)


def discard_singletons(hand):

	ranks = [r for r, _ in hand]
	count = Counter(ranks)

	to_keep = [card for card in hand if count[card[0]] > 1]

	if len(to_keep) == 0:
		sorted_hand = sorted(hand, key=lambda x: x[0], reverse=True)
		to_keep = sorted_hand[:2]

	num_needed = len(hand) - len(to_keep)
	
	return draw_new_cards(to_keep, num_needed)


def discard_random(hand, num_discard=2):
	to_remove = random.sample(hand, num_discard)
	return remove_cards(hand, to_remove)


def discard_non_sequence(hand):
	ranks = sorted(set([r for r, _ in hand]))
	best_seq = []

	for i in range(len(ranks)):
		current_seq = [ranks[i]]
		
		for j in range(i + 1, len(ranks)):
			if ranks[j] == current_seq[-1] + 1:
				current_seq.append(ranks[j])
			else:
				break

		if len(current_seq) > len(best_seq):
			best_seq = current_seq

	# Handle Ace-high straight (10-J-Q-K-A)
	if set([9,10,11,12,0]).issubset(ranks):
		if len(best_seq) < 5:
			best_seq = [9,10,11,12,0]

	to_keep = [card for card in hand if card[0] in best_seq]

	if len(to_keep) < 2:
		sorted_hand = sorted(hand, key=lambda x: x[0], reverse=True)
		to_keep = sorted_hand[:2]

	num_needed = len(hand) - len(to_keep)

	return draw_new_cards(to_keep, num_needed)


def discard_low_unstructured(hand):

	rank_count = Counter([r for r, _ in hand])
	to_keep = []

	for card in hand:
		r, _ = card

		# keep duplicates OR high cards
		if rank_count[r] > 1 or r >= 9:
			to_keep.append(card)

	# fallback
	if len(to_keep) < 2:
		to_keep = sorted(hand, key=lambda x: x[0], reverse=True)[:2]

	num_needed = len(hand) - len(to_keep)
	return draw_new_cards(to_keep, num_needed)


def discard_weak_flush(hand):

	suits = [s for _, s in hand]
	count = Counter(suits)
	best_suit = max(count, key=count.get)

	to_keep = []

	for card in hand:
		r, s = card

		if s == best_suit or r >= 10:
			to_keep.append(card)

	if len(to_keep) < 2:
		to_keep = sorted(hand, key=lambda x: x[0], reverse=True)[:2]

	num_needed = len(hand) - len(to_keep)
	return draw_new_cards(to_keep, num_needed)


def discard_non_sequence_plus_pairs(hand):

	ranks = sorted(set([r for r, _ in hand]))
	rank_count = Counter([r for r, _ in hand])

	best_seq = []

	for i in range(len(ranks)):
		current = [ranks[i]]

		for j in range(i + 1, len(ranks)):
			if ranks[j] == current[-1] + 1:
				current.append(ranks[j])
			else:
				break

		if len(current) > len(best_seq):
			best_seq = current

	to_keep = []

	for card in hand:
		r, _ = card

		if r in best_seq or rank_count[r] > 1:
			to_keep.append(card)

	if len(to_keep) < 2:
		to_keep = sorted(hand, key=lambda x: x[0], reverse=True)[:2]

	num_needed = len(hand) - len(to_keep)
	return draw_new_cards(to_keep, num_needed)


def discard_worst_cards(hand, num_keep=4):

	# score cards by usefulness
	rank_count = Counter([r for r, _ in hand])
	suit_count = Counter([s for _, s in hand])

	def score(card):
		r, s = card
		return (
			r * 2 +              # high card importance
			rank_count[r] * 5 +  # pairs strong
			suit_count[s] * 2    # flush potential
		)

	sorted_hand = sorted(hand, key=score, reverse=True)

	to_keep = sorted_hand[:num_keep]
	num_needed = len(hand) - len(to_keep)

	return draw_new_cards(to_keep, num_needed)


def remove_cards(hand, to_remove):

	new_hand = hand.copy()

	for card in to_remove:
		new_hand.remove(card)

	for _ in range(len(to_remove)):
		new_hand.append(random_card())

	return new_hand


def draw_new_cards(current_hand, num_new):

	new_hand = current_hand.copy()
	for _ in range(num_new):
		new_hand.append(random_card())

	return new_hand



def apply_strategy(hand, action):

	if action == 0:
		return hand

	elif action == 1:
		return discard_non_flush(hand)

	elif action == 2:
		return discard_singletons(hand)

	elif action == 3:
		return discard_low_unstructured(hand)

	elif action == 4:
		return discard_non_sequence_plus_pairs(hand)

	elif action == 5:
		return discard_worst_cards(hand)

	else:
		open("else.txt",'w').write("fflkjhdsafjkasdjfhkfhkasdkjhfsdjfs\n")
		return hand