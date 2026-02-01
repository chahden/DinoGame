import torch
import torch.optim as optim

def compute_target(reward, next_state, done, target_net, gamma=0.99):
    """
    Compute TD (Temporal Difference) target using Bellman Equation:
    
    If done (terminal state):
        target = reward
    Else:
        target = reward + gamma * max(Q(next_state))
    
    Args:
        reward: Immediate reward received
        next_state: Next state after taking action
        done: Whether episode ended
        target_net: Target Q-network (frozen weights)
        gamma: Discount factor (0.99 = values future rewards at 99%)
    """
    # Get device from target_net
    device = next(target_net.parameters()).device
    
    # Move tensor to same device as target_net
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Don't compute gradients for target network
        next_q_values = target_net(next_state_tensor)
        max_next_q = next_q_values.max().item()

    # Bellman equation: if done, only use reward; else add discounted future Q
    target = reward + (0 if done else gamma * max_next_q)
    return target