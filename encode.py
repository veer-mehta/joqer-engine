"""
encode.py — Feature encoder for the Balatro BC pipeline.

Converts one dataset.jsonl entry into a flat float32 numpy array of shape (144,).

Feature layout:
  [  0.. 51]  Hand bag encoding       — 52 floats  (count of each card / hand_size)
  [ 52..103]  Deck remaining presence — 52 floats  (binary)
  [104]       unused_discards         — 1  float    (/ 5.0)
  [105]       hands_left              — 1  float    (/ 4.0)
  [106]       chips_required          — 1  float    (log1p-normalised)
  [107]       chips_scored            — 1  float    (log1p-normalised)
  [108..143]  Poker hands (12 × 3)    — 36 floats  (chips/300, mult/50, level/10)

Total: 144 floats
"""

import math
import numpy as np

# ── Card vocabulary ────────────────────────────────────────────────────────────

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]

RANK_IDX = {r: i for i, r in enumerate(RANKS)}
SUIT_IDX = {s: i for i, s in enumerate(SUITS)}

# Balatro uses "T" or "10" and full suit names; normalise here
RANK_ALIASES = {"T": "10", "10": "10"}


def _card_idx(rank: str, suit: str) -> int | None:
    """Return 0-51 index for a (rank, suit) pair, or None if unrecognised."""
    rank = RANK_ALIASES.get(rank, rank)
    r = RANK_IDX.get(rank)
    s = SUIT_IDX.get(suit)
    if r is None or s is None:
        return None
    return r * 4 + s


# ── Poker hand vocabulary (fixed order, must match training & inference) ───────

HAND_TYPES = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Flush House",
    "Five of a Kind",
    "Flush Five",
]

INPUT_DIM = 144  # exported constant for model.py / decide_bc.py

# ── Normalisation constants ────────────────────────────────────────────────────

_CHIPS_NORM = math.log1p(10_000)   # log1p(10 000) ≈ 9.21


def _log_norm(x: float) -> float:
    return math.log1p(max(0.0, x)) / _CHIPS_NORM


# ── Public API ─────────────────────────────────────────────────────────────────

def encode_state(entry: dict, use_deck: bool = True) -> np.ndarray:
    """
    Convert one dataset.jsonl entry into a (144,) float32 feature vector.

    Args:
        entry:    A dict matching the dataset.jsonl schema.
        use_deck: If False, the deck-remaining slice is zeroed out.
                  Set to False to train the hand-only ablation variant.

    Returns:
        np.ndarray of shape (144,), dtype float32.
    """
    vec = np.zeros(144, dtype=np.float32)

    # ── Hand bag (indices 0-51) ────────────────────────────────────────────────
    hand_cards = entry.get("hand", [])
    hand_size = max(len(hand_cards), 1)  # avoid div-by-zero
    for card in hand_cards:
        idx = _card_idx(str(card.get("rank", "")), str(card.get("suit", "")))
        if idx is not None:
            vec[idx] += 1.0
    vec[0:52] /= hand_size  # normalise to [0, 1]

    # ── Deck remaining presence (indices 52-103) ───────────────────────────────
    if use_deck:
        for card in entry.get("deck_remaining", []):
            idx = _card_idx(str(card.get("rank", "")), str(card.get("suit", "")))
            if idx is not None:
                vec[52 + idx] = 1.0

    # ── Scalars (indices 104-107) ──────────────────────────────────────────────
    vec[104] = min(entry.get("unused_discards", 0) / 5.0, 1.0)
    vec[105] = min(entry.get("hands_left", 0) / 4.0, 1.0)
    vec[106] = _log_norm(entry.get("chips_required", 0))
    vec[107] = _log_norm(entry.get("chips_scored", 0))

    # ── Poker hands (indices 108-143) ─────────────────────────────────────────
    poker = entry.get("poker_hands", {})
    for i, hand_name in enumerate(HAND_TYPES):
        h = poker.get(hand_name, {})
        base = 108 + i * 3
        vec[base + 0] = min(h.get("chips", 0) / 300.0, 1.0)
        vec[base + 1] = min(h.get("mult",  0) /  50.0, 1.0)
        vec[base + 2] = min(h.get("level", 1) /  10.0, 1.0)

    return vec


def encode_labels(entry: dict) -> tuple[int, set[int]]:
    """
    Extract training labels from one dataset entry.

    Returns:
        action_label: 0 = play, 1 = discard
        card_set:     set of 0-based card indexes that were selected
                      (negative padding values are excluded)
    """
    decision = entry.get("decision", {})
    action_label = 0 if decision.get("action", "play") == "play" else 1
    card_set = {
        idx for idx in decision.get("card_indexes", [])
        if isinstance(idx, int) and idx >= 0
    }
    return action_label, card_set


def encode_card_targets(card_set: set[int], hand_size: int) -> np.ndarray:
    """
    Convert card_set into a binary float32 vector of length hand_size.
    Used during training for the card BCE loss.
    """
    targets = np.zeros(hand_size, dtype=np.float32)
    for idx in card_set:
        if 0 <= idx < hand_size:
            targets[idx] = 1.0
    return targets


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    def _sample_entry():
        return {
            "hand": [
                {"rank": "A", "suit": "Hearts"},
                {"rank": "A", "suit": "Spades"},
                {"rank": "K", "suit": "Clubs"},
                {"rank": "Q", "suit": "Diamonds"},
                {"rank": "J", "suit": "Hearts"},
                {"rank": "10", "suit": "Spades"},
                {"rank": "2", "suit": "Clubs"},
                {"rank": "3", "suit": "Hearts"},
            ],
            "deck_remaining": [
                {"rank": "4", "suit": "Diamonds"},
                {"rank": "5", "suit": "Spades"},
            ],
            "unused_discards": 3,
            "hands_left": 2,
            "chips_required": 300,
            "chips_scored": 120,
            "poker_hands": {
                "Pair":           {"chips": 10, "mult": 2,  "level": 1},
                "Two Pair":       {"chips": 20, "mult": 2,  "level": 1},
                "Three of a Kind":{"chips": 30, "mult": 3,  "level": 1},
                "High Card":      {"chips": 5,  "mult": 1,  "level": 1},
                "Straight":       {"chips": 30, "mult": 4,  "level": 1},
                "Flush":          {"chips": 35, "mult": 4,  "level": 1},
                "Full House":     {"chips": 40, "mult": 4,  "level": 1},
                "Four of a Kind": {"chips": 60, "mult": 7,  "level": 1},
                "Straight Flush": {"chips": 100,"mult": 8,  "level": 1},
                "Flush House":    {"chips": 140,"mult": 14, "level": 1},
                "Five of a Kind": {"chips": 120,"mult": 12, "level": 1},
                "Flush Five":     {"chips": 160,"mult": 16, "level": 1},
            },
            "decision": {
                "action": "discard",
                "card_indexes": [2, 3, -1, -1, -1]
            }
        }

    entry = _sample_entry()
    vec = encode_state(entry)
    print(f"encode_state → shape {vec.shape}, dtype {vec.dtype}")
    print(f"  hand slice    (0-51):   min={vec[0:52].min():.3f}  max={vec[0:52].max():.3f}")
    print(f"  deck slice   (52-103):  sum={vec[52:104].sum():.0f}  (expected 2)")
    print(f"  scalars (104-107):      {vec[104:108].tolist()}")
    print(f"  poker slice (108-143):  {vec[108:144].tolist()}")

    action, cards = encode_labels(entry)
    print(f"encode_labels → action={action} ({'discard' if action else 'play'}), cards={cards}")
    targets = encode_card_targets(cards, hand_size=8)
    print(f"encode_card_targets → {targets.tolist()}")
    print("Smoke test passed ✓")
