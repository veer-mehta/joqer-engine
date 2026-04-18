import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from decision_engine.agent.dqn import DQN
from decision_engine.agent.replay_buffer import ReplayBuffer
from decision_engine.env.game_env import GameEnv
from decision_engine.plots import *

env = GameEnv(8, 4)

input_dim = 70
num_actions = 6

model = DQN(input_dim, num_actions)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

target_model = DQN(input_dim, num_actions)
target_model.load_state_dict(model.state_dict())
target_model.eval()
target_update_freq = 100

buffer = ReplayBuffer(10000)
batch_size = 32

total_scores = []
total_rewards = []
action_counts = [0] * num_actions

gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.9998
epsilon_min = 0.05
episodes = 20000


for episode in range(episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    done = False
    total_reward = 0
    total_score = 0

    while not done:
        # ε-greedy action
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        action_counts[action] += 1

        next_state, reward, done, score = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        total_reward += reward
        total_score += score

        # Store transition
        buffer.push(state, action, reward, next_state, done)

        # Train from replay buffer
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)

            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.stack([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            # Current Q values
            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # ===== DOUBLE DQN FIX =====
            with torch.no_grad():
                # Action selection from main model
                next_actions = model(next_states).argmax(1)

                # Action evaluation from target model
                next_q_values = target_model(next_states)

                max_next_q = next_q_values.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze()

            targets = rewards + gamma * max_next_q * (1 - dones)

            # Loss
            loss = ((q_values - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Proper gradient clipping (correct placement)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    total_rewards.append(total_reward)
    total_scores.append(total_score)

    # Better target update (avoid episode 0 update)
    if episode % target_update_freq == 0 and episode != 0:
        target_model.load_state_dict(model.state_dict())

    if episode % 100 == 0:
        avg_reward = sum(total_rewards[-100:]) / len(total_rewards[-100:])
        avg_score = sum(total_scores[-100:]) / len(total_scores[-100:])
        print(
            f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}, Action distribution: {action_counts}, Avg Score: {avg_score:.3f}",
        )
        action_counts = [0] * num_actions


# Test trained model
state = env.reset()

with torch.no_grad():
    q_values = model(torch.tensor(state, dtype=torch.float32))
    action = torch.argmax(q_values).item()

print("Most Chosen action:", action)

plot_rewards(total_rewards)
plot_scores(total_scores)
plot_scores_over_time(total_scores)
plot_action_distribution(action_counts)
plot_reward_vs_score(total_rewards, total_scores)
plot_rolling_score(total_scores)


torch.save(model.state_dict(), "apdqn.pth")
