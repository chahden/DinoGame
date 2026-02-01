import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from env.Dino_env import DinoGame
from Model.dqn import DQN
from Model.replay_buffer import ReplayBuffer
from Model.epsilon_greedy import select_action
from Model.target_calculation import compute_target

# ---------------- HYPERPARAMETERS ----------------
STATE_SIZE = 6
ACTION_SIZE = 2
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 100
MAX_EPISODES = 2000

START_EPISODE = 700  
CHECKPOINT_EPISODE = 700  
# -----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize networks
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

env = DinoGame(render=True)  
step_count = 0

latest_model_path = f"dino_dqn_{CHECKPOINT_EPISODE}.pth"  
if os.path.exists(latest_model_path):
    print(f"Loading model from {latest_model_path}")
    policy_net.load_state_dict(torch.load(latest_model_path, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** START_EPISODE))
    print(f"Model loaded successfully!")
    print(f"Resuming from episode {START_EPISODE + 1} with epsilon {epsilon:.4f}")
else:
    print("No saved model found, training from scratch")
    epsilon = EPSILON_START
    START_EPISODE = 0

# ---------------- TRAINING LOOP ----------------

for episode in range(START_EPISODE + 1, MAX_EPISODES + 1):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, policy_net, epsilon)

        next_state, reward, done = env.step(action)
        total_reward += reward

        memory.push(state, action, reward, next_state, done)
        state = next_state

        if len(memory) >= BATCH_SIZE:  
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states_tensor = torch.FloatTensor(next_states).to(device)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states_tensor).gather(1, actions_tensor)

            target_q_values = []
            for i in range(BATCH_SIZE):
                target = compute_target(
                    rewards[i], next_states[i], dones[i], target_net, gamma=GAMMA
                )
                target_q_values.append([target])
            target_q_values = torch.FloatTensor(target_q_values).to(device)

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print(
        f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Score: {env.score}"
    )

    # Save model checkpoint every 100 episodes
    if episode % 100 == 0:
        torch.save(policy_net.state_dict(), f"dino_dqn_{episode}.pth")
        print(f"Model saved: dino_dqn_{episode}.pth")

print("Training complete!")


