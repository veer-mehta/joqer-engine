from utils.cards import random_hand, print_hand
from logic.discard_strats import apply_strategy
from logic.evaluator import best_hand
from env.game_env import GameEnv
from agent.dqn import DQN
import torch

print(torch.cuda.is_available())

##
print("\ntest cards.py")
hand = random_hand()
print(print_hand(hand))


##
print("\ntest evaluator.py")

hand = [
	(0,0), (0,1), 
	(1,0), (2,0),   
	(2,0),       
	(3,0),         
	(4,3), (5,3)
]
print(print_hand(hand))
print("Score:", best_hand(hand))


##
print("\ntest discard_strats.py")
hand = random_hand()
print("Original:", print_hand(hand))
new_hand = apply_strategy(hand, 5)
print("After discard:", print_hand(new_hand))


##
print("\ntest game_env.py")
env = GameEnv()
env.reset()
state = env.get_state_dict()
done = False
reward = 0

print("INIT HAND:", print_hand(state["hand"]))
print("INITIAL DISCARDS LEFT:", state["discards"])

while not done:
	action = 0  # try different actions manually
	state, reward, done = env.step(action)
	state = env.get_state_dict()
	print("HAND:", print_hand(state["hand"]))
	print("DISCARDS LEFT:", state["discards"])

print("FINAL REWARD:", reward)


##
print("\ntest encoding.py")
state = env.reset()
print("State shape:", state.shape)
print("State:", state)


##

print("\ntest dqn.py")

model = DQN(70, 6)
state = torch.randn(70)
output = model(state)

print("Output:", output)
print("Shape:", output.shape)