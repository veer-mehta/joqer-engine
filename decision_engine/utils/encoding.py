import numpy as np


def get_max_straight_len(ranks):
    unique_ranks = sorted(set(ranks))
    if not unique_ranks:
        return 0
    max_len = 1
    curr_len = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i - 1] + 1:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 1
    ace_high_set = {9, 10, 11, 12, 0}
    ace_high_count = len(ace_high_set.intersection(unique_ranks))
    return max(max_len, ace_high_count)


def encode_state(hand, discards_remaining, hands_remaining=4, round_score=0, target=300, current_hand_score=0.0):
    card_counts = np.zeros(52)
    for r, s in hand:
        card_counts[r * 4 + s] += 1

    rank_counts = np.zeros(13)
    for r, _ in hand:
        rank_counts[r] += 1

    suit_counts = np.zeros(4)
    for _, s in hand:
        suit_counts[s] += 1

    ranks = [r for r, _ in hand]
    max_rank_count = max(rank_counts) if len(rank_counts) > 0 else 0
    max_suit_count = max(suit_counts) if len(suit_counts) > 0 else 0
    max_straight_count = get_max_straight_len(ranks)

    needed = max(0, target - round_score)

    return np.concatenate([
        card_counts, rank_counts, suit_counts,
        np.array([
            discards_remaining / 4.0,
            hands_remaining / 4.0,
            min(1.0, round_score / float(target)),
            min(1.0, needed / 600.0),
            min(1.0, current_hand_score / 400.0),
            min(1.0, max_rank_count / 4.0),
            min(1.0, max_suit_count / 5.0),
            min(1.0, max_straight_count / 5.0)
        ])
    ])
