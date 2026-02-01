import numpy as np
import torch  

def select_action(state, q_network, epsilon):  
    """
    Epsilon-greedy action selection:
    - With probability epsilon: explore (random action)
    - With probability (1-epsilon): exploit (best action from Q-network)
    
    This makes the agent explore randomly at first, then gradually
    use learned knowledge as epsilon decreases over time.
    """
    if np.random.rand() < epsilon:  
        return np.random.randint(0, 2)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return q_values.argmax().item()