import pygame
import sys
import math
from Map import Map
from Map import map_array
from Ghost import Ghost
import numpy as np

class PacMan:
    def __init__(self, position):
        self.position = position
        self.map = Map()
        self.direction = (0, 0)
        self.visited_positions = set()
        self.visited_positions.add(position)
        self.score = 0
        self.power_mode = False
        self.power_mode_timer = 0

    def move(self, direction, ghost_positions):
        # Calculate potential new position
        new_position = (self.position[0] + direction[0], self.position[1] + direction[1])
        
        # Check if the new position is valid (not a wall)
        if not self.map.is_valid_position(new_position):
            return False
            
        # Update the position
        old_position = self.position
        self.position = new_position
        self.direction = direction
        self.visited_positions.add(new_position)
        
        # Update the map cell using the Map's update_cell method
        # This will handle cookie and power pellet consumption
        row, col = new_position
        old_row, old_col = old_position
        
        # Check if Pac-Man ate a cookie or power pellet before updating the cell
        map_array = self.map.get_map()
        current_cell = map_array[row][col]
        
        if current_cell == 'c':
            self.score += 10
        elif current_cell == 'B':
            self.power_mode = True
            self.power_mode_timer = 50  # 50 frames of power mode
            self.score += 50
        
        # First clear the old position
        self.map.update_cell(old_row, old_col, ' ')
        
        # Then update the new position with Pac-Man
        update_success = self.map.update_cell(row, col, 'P1')
        
        return update_success
    
    def update(self):
        if self.power_mode:
            self.power_mode_timer -= 1
            if self.power_mode_timer <= 0:
                self.power_mode = False

    def cost_func(self, position, ghost_positions):
        print(position)
        map_array = self.map.get_map()
        cost = np.array([[0 for _ in range(len(map_array[0]))] for _ in range(len(map_array))])
        ghost_weight = 5
        cookie_weight = 10
        for i in range(len(map_array)):
            for j in range(len(map_array[i])):
                if map_array[i][j] == 'c' or map_array[i][j] == 'B':
                    for k in range(i-1, i+1):
                        for l in range(j-1, j+1):
                            if map_array[k][l] == 'c' or map_array[k][l] == 'B':
                                cost[i][j] += cookie_weight * (1/math.dist(position, (k,l)))
                
                dist = float('inf')
                for ghost in ghost_positions:
                    if math.dist((i,j), ghost) < dist:
                        dist = math.dist((i,j), ghost)
                if dist != float('inf'):
                    cost[i][j] += ghost_weight * dist
        return cost

    
    def get_best_move(self, ghost_positions):
        """Use A* to find the best move for Pacman"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        best_move = (0, 0)
        best_cost = float('inf')
        dir = 1
        
        cost = self.cost_func(self.position, ghost_positions)
        best_cell = np.unravel_index(np.argmax(cost), cost.shape)
        best_angle = math.atan2(best_cell[0] - self.position[0], best_cell[1] - self.position[1])

        if best_angle < -3 * math.pi/4:
            start_idx = 2
            dir = 1
        elif best_angle < -math.pi/2:
            start_idx = 1
            dir = -1
        elif best_angle < -math.pi/4:
            start_idx = 1
            dir = 1
        elif best_angle < 0:
            start_idx = 0
            dir = -1
        elif best_angle < math.pi/4:
            start_idx = 0
            dir = 1
        elif best_angle < math.pi/2:
            start_idx = 3
            dir = -1
        elif best_angle < 3 * math.pi/4:
            start_idx = 3
            dir = 1
        else:
            start_idx = 0
            dir = -1

        for i in range(start_idx, start_idx + 4 * dir, dir):
            direction = directions[i % 4]
            new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
            if (0 <= new_pos[0] < len(map_array) and 0 <= new_pos[1] < len(map_array[0]) and map_array[new_pos[0]][new_pos[1]] != '#'): 
                return direction


def main():
    pygame.init()
    
    # Constants
    CELL_SIZE = 30
    WIDTH = 19 * CELL_SIZE
    HEIGHT = 21 * CELL_SIZE
    FPS = 10
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    PINK = (255, 192, 203)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)
    BLUE = (0, 0, 255)
    
    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()
    
    # Initialize game objects
    game_map = Map()  
    spawn_points = game_map.get_spawn_points()
    
    # Initialize Pac-Man
    pacman = PacMan(spawn_points['pacman'][0]) 
    
    # Initialize Ghosts
    ghost_colors = [RED, PINK, CYAN, ORANGE]
    ghost_behaviors = ['chase', 'chase', 'random', 'random']
    ghosts = []
    for i, spawn in enumerate(spawn_points['ghost']):
        color = ghost_colors[i % len(ghost_colors)]
        ghosts.append(Ghost(spawn, color, game_map, ghost_behaviors[i]))
    
    # Game loop
    running = True
    while running:
        #pacman.direction = pacman.get_best_move([ghost.position for ghost in ghosts])

        # # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    pacman.direction = (-1, 0)
                elif event.key == pygame.K_DOWN:
                    pacman.direction = (1, 0)
                elif event.key == pygame.K_LEFT:
                    pacman.direction = (0, -1)
                elif event.key == pygame.K_RIGHT:
                    pacman.direction = (0, 1)
        
        # Move Pacman based on current direction
        pacman.move(pacman.direction, [ghost.position for ghost in ghosts])
        
        # Update game state
        pacman.update()
        
        # Set ghost scared state based on Pacman's power mode
        for ghost in ghosts:
            ghost.set_scared(pacman.power_mode)
        
        # Move ghosts
        for i, ghost in enumerate(ghosts):
            other_ghost_positions = [g.position for g in ghosts if g != ghost]
            ghost.move(pacman.position, other_ghost_positions)
        
        # Check for collisions between Pacman and ghosts
        for i, ghost in enumerate(ghosts):
            if pacman.position == ghost.position:
                if pacman.power_mode:
                    # Reset ghost to its original spawn point
                    ghost.position = spawn_points['ghost'][i % len(spawn_points['ghost'])]
                    pacman.score += 200
                else:
                    # Game over
                    print(f"Game Over! Score: {pacman.score}")
                    running = False
                    break
        
        # Check win condition
        if not game_map.cookie_positions and not game_map.power_pellet_positions:
            print(f"You Win! Score: {pacman.score}")
            running = False
        
        # Draw everything
        screen.fill(BLACK)
        
        # Draw map
        map_array = game_map.get_map()
        for i, row in enumerate(map_array):
            for j, cell in enumerate(row):
                current_cell = game_map.map_array[i][j]
                if cell == '#':  
                    pygame.draw.rect(screen, BLUE, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif current_cell == 'c':
                    pygame.draw.circle(screen, WHITE, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 10)
                elif current_cell == 'B':
                    pygame.draw.circle(screen, WHITE, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 5)
                elif current_cell == ' ':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif current_cell == 'P1':
                    pygame.draw.circle(screen, YELLOW, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)
                elif current_cell == 'P2':
                    pygame.draw.circle(screen, YELLOW, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)
        
        # # Draw cookies
        # for cookie_pos in game_map.cookie_positions:
        #     pygame.draw.circle(screen, WHITE, 
        #                       (cookie_pos[1] * CELL_SIZE + CELL_SIZE // 2, 
        #                        cookie_pos[0] * CELL_SIZE + CELL_SIZE // 2), 
        #                       CELL_SIZE // 10)
        
        # # Draw power pellets
        # for pellet_pos in game_map.power_pellet_positions:
        #     pygame.draw.circle(screen, WHITE, 
        #                       (pellet_pos[1] * CELL_SIZE + CELL_SIZE // 2, 
        #                        pellet_pos[0] * CELL_SIZE + CELL_SIZE // 2), 
        #                       CELL_SIZE // 4)
        
        # # Draw Pac-Man
        # pygame.draw.circle(screen, YELLOW, 
        #                   (pacman.position[1] * CELL_SIZE + CELL_SIZE // 2, 
        #                    pacman.position[0] * CELL_SIZE + CELL_SIZE // 2), 
        #                   CELL_SIZE // 2)
        
        # Draw Ghosts
        for ghost in ghosts:
            color = BLUE if ghost.scared else ghost.color
            pygame.draw.circle(screen, color, 
                              (ghost.position[1] * CELL_SIZE + CELL_SIZE // 2, 
                               ghost.position[0] * CELL_SIZE + CELL_SIZE // 2), 
                              CELL_SIZE // 2)
        
        # Draw score
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
