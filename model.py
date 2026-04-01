"""
model.py — BalatraBC neural network definition.

Architecture:
    Input (144)
      → Linear(144 → 512) → ReLU → Dropout(0.2)
      → Linear(512 → 256) → ReLU → Dropout(0.2)
      → action_head: Linear(256 → 2)         # logits: [play, discard]
      → card_head:   Linear(256 → MAX_HAND)  # logits per card slot

MAX_HAND = 12 is a generous ceiling; unused slots are masked to -inf at inference.
"""

import json
import torch
import torch.nn as nn
from encode import INPUT_DIM

MAX_HAND = 12  # ceiling for Juggler-expanded hands


class BalatraBC(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, max_hand: int = MAX_HAND):
        super().__init__()
        self.input_dim = input_dim
        self.max_hand = max_hand

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.action_head = nn.Linear(256, 2)       # play=0 / discard=1
        self.card_head   = nn.Linear(256, max_hand) # per-card selection logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Float tensor of shape (batch, input_dim)

        Returns:
            action_logits: (batch, 2)
            card_logits:   (batch, max_hand)  — mask unused positions yourself
        """
        shared = self.shared(x)
        return self.action_head(shared), self.card_head(shared)

    # ── Convenience: inference with dynamic masking ────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        hand_size: int,
        max_cards: int = 5,
        action_override: str | None = None,
    ) -> tuple[str, list[int]]:
        """
        Run inference and return a (action, card_indexes) decision.

        Args:
            x:               (1, input_dim) feature tensor
            hand_size:       actual number of cards in hand
            max_cards:       max cards to select (default 5)
            action_override: force "play" or "discard" (for ablation)

        Returns:
            action:       "play" | "discard"
            card_indexes: list of up to max_cards 0-based indexes, padded with
                          -1 to exactly 5 elements (main.lua compatibility)
        """
        self.eval()
        action_logits, card_logits = self(x)

        # ── Action ──────────────────────────────────────────────────────────
        if action_override:
            action = action_override
        else:
            action = "play" if action_logits[0].argmax().item() == 0 else "discard"

        # ── Cards: mask slots beyond actual hand size ────────────────────────
        masked = card_logits[0].clone()
        if hand_size < self.max_hand:
            masked[hand_size:] = float("-inf")

        # Select cards where sigmoid(logit) > 0.5
        probs = torch.sigmoid(masked[:hand_size])
        selected = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()

        # Clamp to max_cards by taking top-k if too many selected
        if len(selected) > max_cards:
            top_k = masked[:hand_size].topk(max_cards).indices.tolist()
            selected = top_k
        # If nothing selected, take top-1
        if len(selected) == 0:
            selected = [int(masked[:hand_size].argmax().item())]

        # Pad to exactly 5 entries with -1 (main.lua expects 5-element list)
        card_indexes = sorted(selected) + [-1] * (5 - len(selected))
        return action, card_indexes


# ── Save / load helpers ────────────────────────────────────────────────────────

def save_model(model: BalatraBC, weights_path: str, config_path: str) -> None:
    torch.save(model.state_dict(), weights_path)
    config = {"input_dim": model.input_dim, "max_hand": model.max_hand}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved weights → {weights_path}")
    print(f"Saved config  → {config_path}")


def load_model(weights_path: str, config_path: str, device: str = "cpu") -> BalatraBC:
    with open(config_path) as f:
        cfg = json.load(f)
    model = BalatraBC(input_dim=cfg["input_dim"], max_hand=cfg["max_hand"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = BalatraBC()
    total = sum(p.numel() for p in model.parameters())
    print(f"BalatraBC: {total:,} parameters")

    x = torch.randn(4, INPUT_DIM)
    al, cl = model(x)
    print(f"action_logits: {al.shape}   card_logits: {cl.shape}")

    action, cards = model.predict(x[:1], hand_size=8)
    print(f"predict → action={action!r}  card_indexes={cards}")
    print("Smoke test passed ✓")
