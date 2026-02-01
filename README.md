# 🦖 Dino AI - Deep Q-Learning Agent

An AI agent that learns to play the Chrome Dino game using Deep Q-Network (DQN) reinforcement learning.

## 🎯 Overview

This project implements a Deep Q-Network (DQN) agent that learns to play a simplified version of the Chrome Dino game. The agent learns through trial and error, gradually improving its performance by learning when to jump to avoid obstacles.

## ✨ Features

- **Custom Pygame Environment**: Simplified Dino game with adjustable difficulty
- **Deep Q-Network**: Neural network that learns optimal jump timing
- **Experience Replay**: Stores and reuses past experiences for stable learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Checkpoint System**: Save and resume training progress
- **Reward Shaping**: 
  - +1 for surviving each frame
  - +10 for passing obstacles
  - -1 for unnecessary jumps (while in air)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
