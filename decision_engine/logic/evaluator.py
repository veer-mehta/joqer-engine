import itertools
from collections import Counter, defaultdict

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
    # 0 = Ace, 1 = 2, ..., 12 = King
    if rank == 0:
        return 11  # Ace
    if rank >= 10:
        return 10  # face cards
    return rank + 1


def evaluate_hand(cards):

    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    rank_count = Counter(ranks)
    suit_count = Counter(suits)
    counts = sorted(rank_count.values(), reverse=True)

    is_flush = check_flush(suit_count)
    is_straight = check_straight(ranks)
    is_straight_flush = check_straight_flush(cards)

    if is_straight_flush:
        return "straight_flush"

    if counts[0] == 4:
        return "four_kind"

    if counts[0] == 3 and counts[1] >= 2:
        return "full_house"

    if is_flush:
        return "flush"

    if is_straight:
        return "straight"

    if counts[0] == 3:
        return "three_kind"

    if counts[0] == 2 and counts[1] == 2:
        return "two_pair"

    if counts[0] == 2:
        return "pair"

    return "high_card"


def check_flush(suit_count):
    return max(suit_count.values()) >= 5


def check_straight(ranks):
    unique = sorted(set(ranks))

    for i in range(len(unique) - 4):
        if unique[i + 4] - unique[i] == 4:
            return True

    # Ace-high (10-J-Q-K-A)
    if set([9, 10, 11, 12, 0]).issubset(unique):
        return True

    return False


def check_straight_flush(cards):

    suit_groups = defaultdict(list)

    for r, s in cards:
        suit_groups[s].append(r)

    for suit_cards in suit_groups.values():
        if len(suit_cards) >= 5:
            if check_straight(suit_cards):
                return True

    return False


def get_contributing_cards(comb, hand_type):

    ranks = [r for r, _ in comb]
    from collections import Counter
    count = Counter(ranks)

    contributing = set()

    if hand_type == "pair":
        for r in count:
            if count[r] == 2:
                contributing.add(r)

    elif hand_type == "two_pair":
        for r in count:
            if count[r] == 2:
                contributing.add(r)

    elif hand_type == "three":
        for r in count:
            if count[r] == 3:
                contributing.add(r)

    elif hand_type == "four":
        for r in count:
            if count[r] == 4:
                contributing.add(r)

    elif hand_type == "full_house":
        for r in count:
            if count[r] >= 2:
                contributing.add(r)

    else:
        return set(ranks)

    return contributing





def best_hand(cards):
    best = 0

    for comb in itertools.combinations(cards, 5):
        hand_type = evaluate_hand(comb)
        base_chips, mult = HAND_SCORES[hand_type]
        
        contributing = get_contributing_cards(comb, hand_type)
        
        card_sum = 0
        penalty = 0
        
        for r, _ in comb:
            if r in contributing:
                card_sum += card_chips(r)
            else:
                penalty += card_chips(r)
        
        score = mult * (base_chips + card_sum) - penalty
        
        if score > best:
            best = score

    return best
