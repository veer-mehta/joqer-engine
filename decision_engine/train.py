import random
import torch
from decision_engine.agent.dqn import DQN
from decision_engine.agent.replay_buffer import ReplayBuffer
from decision_engine.env.game_env import GameEnv
from decision_engine.plots import *

STATE_DIM = 77
NUM_ACTIONS = 6
EPISODES = 50000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
GAMMA = 0.95
LR = 3e-4
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99988
TARGET_SYNC_EVERY = 100
MAX_STEPS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GameEnv()
buffer = ReplayBuffer(BUFFER_SIZE)

policy_net = DQN(STATE_DIM, NUM_ACTIONS).to(device)
target_net = DQN(STATE_DIM, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
epsilon = EPSILON_START


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    with torch.no_grad():
        return policy_net(state.to(device)).argmax().item()


def train_step():
    if len(buffer) < BATCH_SIZE:
        return

    batch = buffer.sample(BATCH_SIZE)
    states = torch.stack([b[0] for b in batch]).to(device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
    next_states = torch.stack([b[3] for b in batch]).to(device)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(1)
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()

    targets = rewards + GAMMA * next_q * (1 - dones)
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()


scores = []
rewards_log = []
win_history = []

for ep in range(1, EPISODES + 1):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    ep_reward = 0
    ep_score = 0

    for step in range(MAX_STEPS):
        action = choose_action(state, epsilon)
        next_state, reward, done, score = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        buffer.push(state, action, reward, next_state, done)
        train_step()

        ep_reward += reward
        ep_score += score
        state = next_state

        if done:
            break

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    win_history.append(1 if env.game_won else 0)
    scores.append(ep_score)
    rewards_log.append(ep_reward)

    if ep % TARGET_SYNC_EVERY == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if ep % 500 == 0:
        avg_r = sum(rewards_log[-500:]) / 500.0
        avg_s = sum(scores[-500:]) / 500.0
        recent_win_rate = (sum(win_history[-500:]) / 500.0) * 100.0
        print(f"Ep {ep}/{EPISODES} | Reward: {avg_r:.2f} | Score: {avg_s:.0f} | WinRate(500): {recent_win_rate:.1f}% | e: {epsilon:.3f}")

torch.save(policy_net.state_dict(), "apdqn.pth")

plot_rewards(rewards_log)
plot_scores(scores)
plot_scores_over_time(scores)
plot_rolling_score(scores)
