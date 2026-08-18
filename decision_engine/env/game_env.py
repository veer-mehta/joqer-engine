import random
from decision_engine.logic.discard_strats import get_discard_indices
from decision_engine.logic.evaluator import best_hand
from decision_engine.utils.cards import random_hand
from decision_engine.utils.encoding import encode_state


class GameEnv:
    BLIND_TARGETS = [300, 450, 600]

    def __init__(self, hand_size=8, max_hands=4, max_discards=4):
        self.hand_size = hand_size
        self.max_hands = max_hands
        self.max_discards = max_discards
        self.reset()

    def reset(self):
        self.target = random.choice(self.BLIND_TARGETS)
        self.round_score = 0
        self.hands_remaining = self.max_hands
        self.discards_remaining = self.max_discards
        self.hand = random_hand(self.hand_size)
        self.game_won = False
        return self._encode()

    def step(self, action):
        if action == 0 or self.discards_remaining <= 0:
            return self._play()
        else:
            return self._discard(action)

    def _play(self):
        hand_score = best_hand(self.hand)
        self.round_score += hand_score
        self.hands_remaining -= 1

        reward = hand_score / float(self.target)
        done = False

        if self.round_score >= self.target:
            efficiency = (self.hands_remaining + self.discards_remaining) / (self.max_hands + self.max_discards)
            reward += 1.5 + (0.5 * efficiency)
            done = True
            self.game_won = True
        elif self.hands_remaining == 0:
            reward = -1.0
            done = True
        else:
            self.hand = random_hand(self.hand_size)

        return self._encode(), reward, done, hand_score

    def _discard(self, strategy):
        old_score = best_hand(self.hand)

        discard_indices = get_discard_indices(self.hand, strategy)
        self.hand = [card for i, card in enumerate(self.hand) if i not in discard_indices]
        num_needed = self.hand_size - len(self.hand)
        from decision_engine.utils.cards import random_card
        for _ in range(num_needed):
            self.hand.append(random_card())

        self.discards_remaining -= 1
        new_score = best_hand(self.hand)

        if old_score >= 150:
            reward = -0.5
        elif new_score > old_score:
            reward = (new_score - old_score) / float(self.target)
        else:
            reward = -0.05

        return self._encode(), reward, False, 0

    def _encode(self):
        current_score = best_hand(self.hand)
        return encode_state(
            self.hand,
            self.discards_remaining,
            self.hands_remaining,
            self.round_score,
            self.target,
            current_score
        )

    def get_state_dict(self):
        return {
            "hand": self.hand,
            "discards": self.discards_remaining,
            "hands_remaining": self.hands_remaining,
            "round_score": self.round_score,
            "target": self.target,
        }
