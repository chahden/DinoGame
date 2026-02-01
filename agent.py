import random 
from env.Dino_env import DinoGame  

EPISODES = 10

env = DinoGame(render=False)

for episodes in range(EPISODES):
    state = env.reset()
    done = False 
    total_reward = 0

    while not done:
        action = random.choice([0, 1])
        next_state, reward, done = env.step(action)  

        total_reward += reward
        state = next_state

    print(f"Episode: {episodes + 1}, Total Reward: {total_reward}, Score: {env.score}")
    
print("Random agent test finished")