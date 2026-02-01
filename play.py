import pygame
from env.Dino_env import DinoGame

game = DinoGame(render=True)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                state, reward, done = game.step(action=1)
                print(f"State: {state}, Reward: {reward}")
    
    state, reward, done = game.step(action=0)  
    print(f"State: {state}, Reward: {reward}")  
    
    if done:
        print("Game Over! Score:", game.score)
        running = False

pygame.quit()