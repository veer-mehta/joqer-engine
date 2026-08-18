import itertools
from collections import Counter

HAND_SCORES = {
    "straight_flush": (100, 8),
    "four_kind": (60, 7),
    "full_house": (40, 4),
    "flush": (35, 4),
    "straight": (30, 4),
    "three_kind": (20, 3),
    "two_pair": (15, 2),
    "pair": (10, 2),
    "high_card": (5, 1),
}


def card_chips(rank):
    if rank == 0:
        return 11
    if rank >= 10:
        return 10
    return rank + 1


def score_combination(cards):
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks))
    is_straight = False
    if len(unique_ranks) == 5:
        if unique_ranks[4] - unique_ranks[0] == 4 or unique_ranks == [0, 9, 10, 11, 12]:
            is_straight = True

    if is_straight and is_flush:
        hand_type = "straight_flush"
        contributing = set(ranks)
    elif counts[0] == 4:
        hand_type = "four_kind"
        contributing = {r for r, c in rank_counts.items() if c == 4}
    elif counts[0] == 3 and counts[1] == 2:
        hand_type = "full_house"
        contributing = set(ranks)
    elif is_flush:
        hand_type = "flush"
        contributing = set(ranks)
    elif is_straight:
        hand_type = "straight"
        contributing = set(ranks)
    elif counts[0] == 3:
        hand_type = "three_kind"
        contributing = {r for r, c in rank_counts.items() if c == 3}
    elif counts[0] == 2 and counts[1] == 2:
        hand_type = "two_pair"
        contributing = {r for r, c in rank_counts.items() if c == 2}
    elif counts[0] == 2:
        hand_type = "pair"
        contributing = {r for r, c in rank_counts.items() if c == 2}
    else:
        hand_type = "high_card"
        contributing = {max(ranks, key=card_chips)}

    base_chips, mult = HAND_SCORES[hand_type]
    card_chips_sum = sum(card_chips(r) for r, _ in cards if r in contributing)
    score = mult * (base_chips + card_chips_sum)

    return hand_type, score, contributing


def evaluate_hand(cards):
    hand_type, _, _ = score_combination(cards)
    return hand_type


def get_contributing_cards(cards, hand_type=None):
    _, _, contributing = score_combination(cards)
    return contributing


def best_hand(cards):
    best = 0
    for comb in itertools.combinations(cards, 5):
        _, score, _ = score_combination(comb)
        if score > best:
            best = score
    return best
