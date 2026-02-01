import random 
from collections import deque 

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    Stores transitions (state, action, reward, next_state, done) and 
    samples random batches for training. Uses FIFO - when full, 
    oldest experiences are automatically removed.
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  
        return states, actions, rewards, next_states, dones  
    
    def __len__(self):
        
        return len(self.buffer)