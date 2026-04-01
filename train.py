"""
train.py — Behavioral Cloning training for BalatraBC.

Usage:
    python train.py --data dataset.jsonl --epochs 50
    python train.py --data dataset.jsonl --epochs 50 --no-deck   # hand-only ablation

Outputs:
    model.pt           — trained weights
    model_config.json  — {input_dim, max_hand}
    training_log.json  — per-epoch metrics
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from encode import (
    INPUT_DIM,
    encode_card_targets,
    encode_labels,
    encode_state,
)
from model import MAX_HAND, BalatraBC, save_model
from solver_baseline import jaccard, solver_decide


# ── Dataset ────────────────────────────────────────────────────────────────────

class BalatraDataset(Dataset):
    def __init__(self, entries: list[dict], use_deck: bool = True):
        self.entries = entries
        self.use_deck = use_deck

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        x = torch.from_numpy(encode_state(entry, use_deck=self.use_deck))
        action_label, card_set = encode_labels(entry)
        hand_size = len(entry.get("hand", []))
        card_targets_full = np.zeros(MAX_HAND, dtype=np.float32)
        card_targets_np = encode_card_targets(card_set, hand_size)
        card_targets_full[:hand_size] = card_targets_np
        return (
            x,
            torch.tensor(action_label, dtype=torch.long),
            torch.from_numpy(card_targets_full),
            hand_size,
            card_set,
        )


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    model: BalatraBC,
    loader: DataLoader,
    device: str,
    entries: list[dict],
    entry_indices: list[int],
) -> dict:
    """Compute action accuracy, card Jaccard, exact match for model and solver."""
    model.eval()

    action_correct = 0
    jaccard_sum = 0.0
    exact_match = 0
    solver_action_correct = 0
    solver_jaccard_sum = 0.0
    solver_exact_match = 0
    total = 0

    with torch.no_grad():
        sample_ptr = 0
        for batch in loader:
            x, action_labels, card_targets, hand_sizes, card_sets = batch
            x = x.to(device)
            action_labels = action_labels.to(device)

            action_logits, card_logits = model(x)
            pred_actions = action_logits.argmax(dim=1)

            for i in range(len(x)):
                action_pred = int(pred_actions[i].item())
                action_true = int(action_labels[i].item())
                hand_size   = int(hand_sizes[i])
                true_cards  = card_sets[i]
                if isinstance(true_cards, torch.Tensor):
                    true_cards = set(true_cards.tolist())

                # ── Model card prediction ──────────────────────────────────
                masked = card_logits[i].clone()
                if hand_size < MAX_HAND:
                    masked[hand_size:] = float("-inf")
                probs = torch.sigmoid(masked[:hand_size])
                pred_cards = set((probs > 0.5).nonzero(as_tuple=True)[0].tolist())
                if not pred_cards and hand_size > 0:
                    pred_cards = {int(masked[:hand_size].argmax().item())}

                action_correct += int(action_pred == action_true)
                jac = jaccard(pred_cards, true_cards)
                jaccard_sum += jac
                exact_match += int(action_pred == action_true and pred_cards == true_cards)

                # ── Solver baseline ────────────────────────────────────────
                global_idx = entry_indices[sample_ptr + i]
                entry = entries[global_idx]
                s_action_str, s_card_idxs = solver_decide(entry)
                s_action = 0 if s_action_str == "play" else 1
                s_cards  = set(idx for idx in s_card_idxs if idx >= 0)

                solver_action_correct += int(s_action == action_true)
                solver_jaccard_sum    += jaccard(s_cards, true_cards)
                solver_exact_match    += int(s_action == action_true and s_cards == true_cards)

                total += 1
            sample_ptr += len(x)

    n = max(total, 1)
    return {
        "model": {
            "action_acc":  action_correct / n,
            "card_jaccard": jaccard_sum / n,
            "exact_match": exact_match / n,
        },
        "solver": {
            "action_acc":  solver_action_correct / n,
            "card_jaccard": solver_jaccard_sum / n,
            "exact_match": solver_exact_match / n,
        },
        "n_samples": total,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    entries = []
    with open(data_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {lineno}: {e}")

    print(f"Loaded {len(entries)} entries from {data_path}")
    if len(entries) < 10:
        print("WARNING: very few samples — metrics may be unreliable.")

    # Shuffle and split
    indices = list(range(len(entries)))
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx   = indices[split:]
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    use_deck = not args.no_deck
    train_entries = [entries[i] for i in train_idx]
    val_entries   = [entries[i] for i in val_idx]

    train_ds = BalatraDataset(train_entries, use_deck=use_deck)
    val_ds   = BalatraDataset(val_entries,   use_deck=use_deck)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = BalatraBC(input_dim=INPUT_DIM, max_hand=MAX_HAND).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimiser + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    action_loss_fn = nn.CrossEntropyLoss()

    log = []
    best_val_jaccard = -1.0
    best_epoch = 0

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'ActAcc':>7}  {'Jaccard':>8}  {'ExactM':>7}  {'LR':>8}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, action_labels, card_targets, hand_sizes, _ = batch
            x             = x.to(device)
            action_labels = action_labels.to(device)
            card_targets  = card_targets.to(device)

            optimizer.zero_grad()
            action_logits, card_logits = model(x)

            # Action loss
            l_action = action_loss_fn(action_logits, action_labels)

            # Card loss: multi-label BCE, masked to actual hand sizes
            l_card = torch.tensor(0.0, device=device)
            for i, hs in enumerate(hand_sizes):
                hs = int(hs)
                if hs == 0:
                    continue
                l_card += nn.functional.binary_cross_entropy_with_logits(
                    card_logits[i, :hs],
                    card_targets[i, :hs],
                )
            l_card /= max(len(hand_sizes), 1)

            loss = l_action + l_card
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        # Validation metrics every epoch (solver is also evaluated here)
        val_metrics = compute_metrics(model, val_loader, device, entries, val_idx)
        m = val_metrics["model"]
        s = val_metrics["solver"]

        print(
            f"{epoch:>5}  {avg_loss:>8.4f}  {m['action_acc']:>7.3f}"
            f"  {m['card_jaccard']:>8.3f}  {m['exact_match']:>7.3f}  {lr_now:>8.2e}"
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"        [solver] act={s['action_acc']:.3f}  "
                f"jacc={s['card_jaccard']:.3f}  exact={s['exact_match']:.3f}"
            )

        entry_log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": lr_now,
            "val": m,
            "solver": s,
        }
        log.append(entry_log)

        # Save best checkpoint
        if m["card_jaccard"] > best_val_jaccard:
            best_val_jaccard = m["card_jaccard"]
            best_epoch = epoch
            save_model(model, "model.pt", "model_config.json")

    print(f"\nBest val Jaccard: {best_val_jaccard:.4f} at epoch {best_epoch}")

    # Dump full training log
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Training log → training_log.json")

    # Final 3-way comparison table
    final = log[-1]
    print("\n── 3-Way Comparison (val set, final epoch) ──")
    print(f"{'Method':<25} {'ActionAcc':>10} {'CardJaccard':>12} {'ExactMatch':>11}")
    print("-" * 62)
    print(f"{'Solver Baseline':<25} {final['solver']['action_acc']:>10.3f} {final['solver']['card_jaccard']:>12.3f} {final['solver']['exact_match']:>11.3f}")
    print(f"{'BC Model':<25} {final['val']['action_acc']:>10.3f} {final['val']['card_jaccard']:>12.3f} {final['val']['exact_match']:>11.3f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BalatraBC model")
    parser.add_argument("--data",       default="dataset.jsonl", help="Path to dataset.jsonl")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--no-deck",    action="store_true",
                        help="Zero out deck_remaining features (hand-only ablation)")
    args = parser.parse_args()
    train(args)
