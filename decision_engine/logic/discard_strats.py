from collections import Counter
from decision_engine.utils.cards import random_card


def is_high_card(card):
    return card[0] == 0 or card[0] >= 9


def get_cards_to_keep(hand, action):
    if action == 1:
        suits = [s for _, s in hand]
        best_suit = max(Counter(suits), key=Counter(suits).get)
        to_keep = [card for card in hand if card[1] == best_suit or is_high_card(card)]
        if len(to_keep) < 2:
            to_keep = [card for card in hand if card[1] == best_suit]
        return to_keep

    elif action == 2:
        rank_count = Counter([r for r, _ in hand])
        to_keep = [card for card in hand if rank_count[card[0]] > 1 or is_high_card(card)]
        if len(to_keep) < 2:
            sorted_hand = sorted(hand, key=lambda c: (11 if c[0] == 0 else c[0]), reverse=True)
            to_keep = sorted_hand[:3]
        return to_keep

    elif action == 3:
        rank_count = Counter([r for r, _ in hand])
        to_keep = [card for card in hand if rank_count[card[0]] > 1 or is_high_card(card)]
        if len(to_keep) < 2:
            sorted_hand = sorted(hand, key=lambda c: (11 if c[0] == 0 else c[0]), reverse=True)
            to_keep = sorted_hand[:3]
        return to_keep

    elif action == 4:
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
        to_keep = [card for card in hand if card[0] in best_seq or rank_count[card[0]] > 1 or is_high_card(card)]
        if len(to_keep) < 2:
            sorted_hand = sorted(hand, key=lambda c: (11 if c[0] == 0 else c[0]), reverse=True)
            to_keep = sorted_hand[:3]
        return to_keep

    elif action == 5:
        rank_count = Counter([r for r, _ in hand])
        suit_count = Counter([s for _, s in hand])

        def card_val(card):
            r, s = card
            chips = 11 if r == 0 else (10 if r >= 10 else r + 1)
            return chips * 3 + rank_count[r] * 10 + suit_count[s] * 2

        sorted_hand = sorted(hand, key=card_val, reverse=True)
        return sorted_hand[:5]

    return hand


def get_discard_indices(hand, action):
    to_keep = get_cards_to_keep(hand, action)
    kept_counts = Counter(to_keep)
    discard_indices = []
    for i, card in enumerate(hand):
        if kept_counts[card] > 0:
            kept_counts[card] -= 1
        else:
            discard_indices.append(i)
    return discard_indices


def apply_strategy(hand, action):
    to_keep = get_cards_to_keep(hand, action)
    num_needed = len(hand) - len(to_keep)
    new_hand = to_keep.copy()
    for _ in range(num_needed):
        new_hand.append(random_card())
    return new_hand