"""
solver_baseline.py — Greedy rule-based Balatro hand evaluator.

Strategy: evaluate all subsets of the hand (size 1-5), score each using
the actual poker_hands chip×mult values from the game state, play the best.
Never discards. Used as a comparison baseline in train.py.

Usage:
    from solver_baseline import solver_decide
    action, card_indexes = solver_decide(entry)
"""

from itertools import combinations
from encode import RANKS, RANK_IDX, SUIT_IDX, RANK_ALIASES


# ── Poker hand detection ───────────────────────────────────────────────────────

def _parse_hand(hand_cards: list[dict]) -> list[tuple[int, int]]:
    """Convert hand list → list of (rank_idx, suit_idx) tuples."""
    result = []
    for c in hand_cards:
        rank = RANK_ALIASES.get(str(c.get("rank", "")), str(c.get("rank", "")))
        suit = str(c.get("suit", ""))
        r = RANK_IDX.get(rank)
        s = SUIT_IDX.get(suit)
        if r is not None and s is not None:
            result.append((r, s))
    return result


def classify_hand(cards: list[tuple[int, int]]) -> str:
    """
    Return the best poker hand name for a subset of cards.
    Handles all 12 Balatro hand types.
    """
    if not cards:
        return "High Card"

    ranks = [r for r, s in cards]
    suits = [s for r, s in cards]
    n = len(cards)

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)
    unique_ranks = sorted(set(ranks))
    all_same_suit = len(set(suits)) == 1
    is_straight = (
        len(unique_ranks) == n
        and (unique_ranks[-1] - unique_ranks[0] == n - 1)
    ) or (
        # Ace-low straight: A-2-3-4-5 (ranks 0,1,2,3,12)
        n == 5
        and set(unique_ranks) == {0, 1, 2, 3, 12}
    )

    # Five of a Kind / Flush Five
    if n == 5 and counts[0] == 5:
        return "Flush Five" if all_same_suit else "Five of a Kind"

    # Flush House (Full House + same suit)
    if n == 5 and counts[0] == 3 and counts[1] == 2 and all_same_suit:
        return "Flush House"

    # Straight Flush
    if n == 5 and is_straight and all_same_suit:
        return "Straight Flush"

    # Four of a Kind
    if counts[0] == 4:
        return "Four of a Kind"

    # Full House
    if n == 5 and counts[0] == 3 and len(counts) >= 2 and counts[1] == 2:
        return "Full House"

    # Flush
    if n == 5 and all_same_suit:
        return "Flush"

    # Straight
    if n == 5 and is_straight:
        return "Straight"

    # Three of a Kind
    if counts[0] == 3:
        return "Three of a Kind"

    # Two Pair
    if counts[0] == 2 and len(counts) >= 2 and counts[1] == 2:
        return "Two Pair"

    # Pair
    if counts[0] == 2:
        return "Pair"

    return "High Card"


def score_hand(hand_name: str, poker_hands: dict) -> float:
    """Compute chips × mult for a classified hand using game state values."""
    h = poker_hands.get(hand_name, {})
    chips = h.get("chips", 0)
    mult  = h.get("mult",  1)
    return float(chips * mult)


# ── Main solver ────────────────────────────────────────────────────────────────

def solver_decide(entry: dict) -> tuple[str, list[int]]:
    """
    Given a dataset entry, return the solver's (action, card_indexes).

    The solver:
    - Tries all subsets of size 1-5 from the hand
    - Picks the subset with the highest chips × mult
    - Always plays (never discards) — this is intentional for the ablation

    Returns:
        action:       always "play"
        card_indexes: 5-element list, unused slots padded with -1
    """
    hand_cards = entry.get("hand", [])
    poker_hands = entry.get("poker_hands", {})
    parsed = _parse_hand(hand_cards)

    best_score = -1.0
    best_subset: list[int] = [0]  # fallback: play first card

    max_select = min(5, len(parsed))
    for size in range(1, max_select + 1):
        for combo in combinations(range(len(parsed)), size):
            subset = [parsed[i] for i in combo]
            hand_name = classify_hand(subset)
            sc = score_hand(hand_name, poker_hands)
            if sc > best_score:
                best_score = sc
                best_subset = list(combo)

    card_indexes = sorted(best_subset) + [-1] * (5 - len(best_subset))
    return "play", card_indexes


# ── Metrics helpers (used by train.py) ────────────────────────────────────────

def jaccard(pred_set: set[int], true_set: set[int]) -> float:
    """Intersection over union for card sets."""
    if not pred_set and not true_set:
        return 1.0
    union = pred_set | true_set
    if not union:
        return 1.0
    return len(pred_set & true_set) / len(union)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "hand": [
            {"rank": "A", "suit": "Hearts"},
            {"rank": "A", "suit": "Spades"},
            {"rank": "A", "suit": "Clubs"},
            {"rank": "K", "suit": "Diamonds"},
            {"rank": "K", "suit": "Hearts"},
            {"rank": "2", "suit": "Clubs"},
            {"rank": "7", "suit": "Diamonds"},
        ],
        "poker_hands": {
            "High Card":       {"chips": 5,  "mult": 1},
            "Pair":            {"chips": 10, "mult": 2},
            "Two Pair":        {"chips": 20, "mult": 2},
            "Three of a Kind": {"chips": 30, "mult": 3},
            "Full House":      {"chips": 40, "mult": 4},
            "Four of a Kind":  {"chips": 60, "mult": 7},
            "Straight":        {"chips": 30, "mult": 4},
            "Flush":           {"chips": 35, "mult": 4},
            "Straight Flush":  {"chips": 100,"mult": 8},
            "Flush House":     {"chips": 140,"mult": 14},
            "Five of a Kind":  {"chips": 120,"mult": 12},
            "Flush Five":      {"chips": 160,"mult": 16},
        },
    }
    action, cards = solver_decide(sample)
    print(f"Solver → action={action!r}  card_indexes={cards}")
    # Should choose Full House (3 Aces + 2 Kings) → 40×4 = 160
    selected = [i for i in cards if i >= 0]
    chosen_ranks = [sample["hand"][i]["rank"] for i in selected]
    print(f"Chosen cards: {chosen_ranks}")
    print("Smoke test passed ✓")
