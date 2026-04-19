import random

# 0–12 → A,2,3,...,K
ranks = list(range(13))
rank_str = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

# 0–3 → ♠ ♥ ♦ ♣
suits = list(range(4))
suit_str = ["♠","♥","♦","♣"]


def random_card():
    return (random.choice(ranks), random.choice(suits))


def random_hand(n=8):
    return [random_card() for _ in range(n)]


def get_ranks(hand):
	return [card[0] for card in hand]


def get_suits(hand):
	return [card[1] for card in hand]


def print_hand(hand):
	return " ".join([rank_str[r] + suit_str[s] for r, s in hand])