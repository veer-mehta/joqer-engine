from decision_engine.logic.discard_strats import apply_strategy
from decision_engine.logic.evaluator import best_hand
from decision_engine.utils.cards import random_hand
from decision_engine.utils.encoding import encode_state


class GameEnv:
    def __init__(self, hand_size=8, max_discards=4):
        self.hand_size = hand_size
        self.max_discards = max_discards
        self.reset()

    def reset(self):
        self.hand = random_hand(self.hand_size)
        self.discards_remaining = self.max_discards
        return self.get_state()

    def step(self, action):

        done = False
        reward = -0.01
        score = 0

        # Action 0 = play
        if action == 0:
            done = True

            if self.discards_remaining > 0:
                reward = -0.03
            else:
                score = best_hand(self.hand)
                reward = score / 500.0

            return self.get_state(), reward, done, score

        # Discard action
        if self.discards_remaining > 0:
            old_score = best_hand(self.hand)

            self.hand = apply_strategy(self.hand, action)
            self.discards_remaining -= 1

            new_score = best_hand(self.hand)

            # Improvement reward (IMPORTANT)
            reward += (new_score - old_score) / 500.0

        # Force play
        if self.discards_remaining == 0:
            score = best_hand(self.hand)
            reward = score / 500.0
            done = True

        return self.get_state(), reward, done, score

    def get_state(self):
        return encode_state(self.hand, self.discards_remaining)

    def get_state_dict(self):
        return {"hand": self.hand, "discards": self.discards_remaining}
