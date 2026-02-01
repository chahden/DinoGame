import pygame 
import random 

WIDTH, HEIGHT = 800, 600
GROUND_y = 250 
FPS = 60

class Dino:
    """The Dino"""
    def __init__(self):
        self.x = 50
        self.y = GROUND_y
        self.vel_y = 0
        self.jump_force = -12
        self.gravity = 0.8 
        self.ground = True
        self.width = 40
        self.height = 40
    """ Jump System"""
    def jump(self):
        if self.ground:
            self.vel_y = self.jump_force
            self.ground = False
    """Update Dino Position"""
    def update(self):
        self.vel_y += self.gravity
        self.y += self.vel_y
        
        if self.y >= GROUND_y:
            self.y = GROUND_y
            self.vel_y = 0
            self.ground = True
    """Return Dino when he collide with Cactus"""
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Cactus:
    """Cactus move from right to left"""
    def __init__(self, speed):
        self.x = WIDTH 
        self.y = GROUND_y
        self.width = 20 
        self.height = random.randint(30, 50)
        self.speed = speed 
        self.passed = False  

    def update(self):
        self.x -= self.speed
    
    def rect(self):
        return pygame.Rect(self.x, self.y + (40 - self.height), self.width, self.height)

class DinoGame:
    """Main game class that handles game logic and rendering"""

    def __init__(self, render=True):
        pygame.init()
        self.render = render
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dino Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.reset()
    """Reset the game to initial state"""

    def reset(self):
        self.dino = Dino()
        self.cactus = []
        self.score = 0
        self.speed = 6
        self.spawn_timer = 0 
        self.done = False
        return self.get_state()

    
    def step(self, action=None):
        reward = 0
        if action == 1:
            self.dino.jump()
        
        self.dino.update()
        if action == 1 and self.dino.vel_y != 0:
            reward -= 10
        self.spawn_timer += 1
        if self.spawn_timer > 90:
            self.cactus.append(Cactus(self.speed))
            self.spawn_timer = 0
        
        for i in self.cactus:
            i.update()
        
        self.cactus = [o for o in self.cactus if o.x + o.width > 0]
        
        for i in self.cactus:
            if self.dino.rect().colliderect(i.rect()):
                self.done = True
        
        reward += 1 
        self.score += 1

        for i in self.cactus:
            if i.x + i.width < self.dino.x and not i.passed:
                i.passed = True
                reward += 10

        self.speed = 6 + self.score // 500
        
        if self.render:
            self.draw()
        
        return self.get_state(), reward, self.done


    def next_cactus(self):
        for i in self.cactus:
            if i.x > self.dino.x:
                return i
        return None

    def get_state(self):
        next_cactus = self.next_cactus()

        if next_cactus:
            distance = next_cactus.x - self.dino.x
            cactus_width = next_cactus.width
            cactus_height = next_cactus.height
        else:
            distance = WIDTH
            cactus_width = 0
            cactus_height = 0
        
        state = [
            distance / WIDTH,
            cactus_width / 50,
            cactus_height / 50,
            self.dino.y / GROUND_y,
            self.dino.vel_y / 10,
            self.speed / 20
        ]

        return state
    
    """Draw all game elements to the screen"""

    def draw(self):
        """Draw all game elements to the screen"""
        # IMPORTANT: Process pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.done = True
                return
        
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0), self.dino.rect())
        
        for i in self.cactus:
            pygame.draw.rect(self.screen, (0, 0, 0), i.rect())
        
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)