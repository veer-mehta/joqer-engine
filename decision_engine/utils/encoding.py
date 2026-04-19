import numpy as np


def encode_state(hand, discards_remaining):
    # 52 card counts
    card_counts = np.zeros(52)

    for r, s in hand:
        index = r * 4 + s
        card_counts[index] += 1

    # Rank counts (13)
    rank_counts = np.zeros(13)

    for r, _ in hand:
        rank_counts[r] += 1

    # Suit counts (4)
    suit_counts = np.zeros(4)

    for _, s in hand:
        suit_counts[s] += 1

	# add remaining discards at the end
    state = np.concatenate(
        [card_counts, rank_counts, suit_counts, np.array([discards_remaining])]
    )

    return state
