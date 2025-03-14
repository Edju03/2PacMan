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
        # Initialize known map as empty 
        self.known_map = [[None for _ in range(len(self.map.get_map()[0]))] for _ in range(len(self.map.get_map()))]
        # Update initial position and surroundings
        self.update_known_map()
        # Track visible ghosts
        self.visible_ghosts = []

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
        
        # Update what Pac-Man knows about the map
        self.update_known_map()
        
        # Update visible ghosts based on current position
        self.update_visible_ghosts(ghost_positions)
        
        return update_success
    
    def update(self):
        if self.power_mode:
            self.power_mode_timer -= 1
            if self.power_mode_timer <= 0:
                self.power_mode = False
    
    def update_known_map(self):
        # Update what Pac-Man knows about the map within Manhattan distance of 5
        real_map = self.map.get_map()
        for i in range(max(0, self.position[0] - 5), min(len(real_map), self.position[0] + 6)):
            for j in range(max(0, self.position[1] - 5), min(len(real_map[0]), self.position[1] + 6)):
                # Check if within Manhattan distance of 5
                if abs(i - self.position[0]) + abs(j - self.position[1]) <= 5:
                    self.known_map[i][j] = real_map[i][j]
    
    def update_visible_ghosts(self, ghost_positions):
        # Clear previous visible ghosts
        self.visible_ghosts = []
        # Check which ghosts are within Manhattan distance of 5
        for ghost_pos in ghost_positions:
            if abs(ghost_pos[0] - self.position[0]) + abs(ghost_pos[1] - self.position[1]) <= 5:
                self.visible_ghosts.append(ghost_pos)

    # def cost_func(self, position, ghost_positions):
    #     print(position)
    #     map_array = self.map.get_map()
    #     cost = np.array([[0 for _ in range(len(map_array[0]))] for _ in range(len(map_array))])
    #     ghost_weight = 5
    #     cookie_weight = 10
    #     for i in range(len(map_array)):
    #         for j in range(len(map_array[i])):
    #             if map_array[i][j] == 'c' or map_array[i][j] == 'B':
    #                 for k in range(i-1, i+1):
    #                     for l in range(j-1, j+1):
    #                         if map_array[k][l] == 'c' or map_array[k][l] == 'B':
    #                             cost[i][j] += cookie_weight * (1/math.dist(position, (k,l)))
                
    #             dist = float('inf')
    #             for ghost in ghost_positions:
    #                 if math.dist((i,j), ghost) < dist:
    #                     dist = math.dist((i,j), ghost)
    #             if dist != float('inf'):
    #                 cost[i][j] += ghost_weight * dist
    #     return cost


    def cost_func(self, position, ghost_positions):
        # Use known map instead of full map - focus on immediate neighbors
        cost = np.zeros((5, 5))  # 5x5 grid centered on Pac-Man
        ghost_weight = -30  # Stronger negative weight to avoid ghosts
        cookie_weight = 15
        unexplored_weight = 5
        
        # First, set very negative cost for all walls and out-of-bounds cells
        for di in range(-2, 3):
            for dj in range(-2, 3):
                i, j = position[0] + di, position[1] + dj
                cost_i, cost_j = di + 2, dj + 2
                
                # Check if position is valid
                if (i < 0 or i >= len(self.known_map) or 
                    j < 0 or j >= len(self.known_map[0]) or 
                    (self.known_map[i][j] is not None and self.known_map[i][j] == '#')):
                    cost[cost_i, cost_j] = -1000
        
        # Process immediate neighbors (the cells Pac-Man can actually move to)
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            i, j = position[0] + di, position[1] + dj
            cost_i, cost_j = di + 2, dj + 2
            
            # Skip invalid positions
            if (i < 0 or i >= len(self.known_map) or 
                j < 0 or j >= len(self.known_map[0]) or 
                (self.known_map[i][j] is not None and self.known_map[i][j] == '#')):
                continue
            
            # Add value for cookies and power pellets
            if self.known_map[i][j] == 'c':
                cost[cost_i, cost_j] += cookie_weight
            elif self.known_map[i][j] == 'B':
                cost[cost_i, cost_j] += cookie_weight 
            
            # Look for nearby cookies to create paths with multiple rewards
            cookie_count = 0
            for k in range(max(0, i-3), min(len(self.known_map), i+4)):
                for l in range(max(0, j-3), min(len(self.known_map[0]), j+4)):
                    # Skip if cell is unknown or too far
                    if (k < 0 or k >= len(self.known_map) or 
                        l < 0 or l >= len(self.known_map[0]) or 
                        self.known_map[k][l] is None or
                        abs(k-i) + abs(l-j) > 3):  
                        continue
                    
                    if self.known_map[k][l] == 'c' or self.known_map[k][l] == 'B':
                        cookie_count += 1
            
            # Add bonus for being near multiple cookies
            cost[cost_i, cost_j] += cookie_weight * 0.5 * cookie_count
            
            # Penalize cells near ghosts (negative weight)
            for ghost in self.visible_ghosts:
                ghost_dist = math.dist((i,j), ghost)
                if ghost_dist < 4: 
                    cost[cost_i, cost_j] -= 1000
                        
            # Penalize backtracking unless necessary
            if (i, j) in self.visited_positions:
                # Check how many valid moves are available
                valid_moves = 0
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if (0 <= ni < len(self.known_map) and 
                        0 <= nj < len(self.known_map[0]) and 
                        (self.known_map[ni][nj] is None or self.known_map[ni][nj] != '#')):
                        valid_moves += 1
                
                # Only penalize backtracking if there are other options
                if valid_moves > 1:
                    cost[cost_i, cost_j] -= 10
        
        # Add some small random noise to break ties
        cost += np.random.random((5, 5)) * 0.1
        
        return cost
    
    def get_best_move(self, ghost_positions):
        """Use cost function to find the best move for Pacman"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        # Update visible ghosts before calculating cost
        self.update_visible_ghosts(ghost_positions)
        
        # Calculate cost for each direction
        best_direction = (0, 0)
        best_score = float('-inf')
        
        for direction in directions:
            new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
            
            # Skip invalid positions
            if (new_pos[0] < 0 or new_pos[0] >= len(self.known_map) or 
                new_pos[1] < 0 or new_pos[1] >= len(self.known_map[0]) or 
                (self.known_map[new_pos[0]][new_pos[1]] is not None and 
                 self.known_map[new_pos[0]][new_pos[1]] == '#')):
                continue
            
            # Calculate position in cost matrix (center is at (2,2))
            cost_i, cost_j = direction[0] + 2, direction[1] + 2
            cost_matrix = self.cost_func(self.position, ghost_positions)
            move_score = cost_matrix[cost_i, cost_j]
            
            # Choose direction with highest score
            if move_score > best_score:
                best_score = move_score
                best_direction = direction
        
        # If no valid move found, try to stay in place
        if best_direction == (0, 0) and best_score == float('-inf'):
            # Try any valid move as a fallback
            for direction in directions:
                new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
                if (0 <= new_pos[0] < len(self.known_map) and 
                    0 <= new_pos[1] < len(self.known_map[0]) and 
                    (self.known_map[new_pos[0]][new_pos[1]] is None or 
                     self.known_map[new_pos[0]][new_pos[1]] != '#')):
                    return direction
        
        return best_direction


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
    GRAY = (100, 100, 100) 
    
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
    ghosts = []
    for i, spawn in enumerate(spawn_points['ghost']):
        color = ghost_colors[i % len(ghost_colors)]
        ghosts.append(Ghost(spawn, color, game_map, speed = 0.5))
    
    # Game loop
    running = True
    while running:
        pacman.direction = pacman.get_best_move([ghost.position for ghost in ghosts])

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
        
        # Update visible ghosts after they've moved
        pacman.update_visible_ghosts([ghost.position for ghost in ghosts])
        
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
        
        # Draw map based on Pac-Man's knowledge
        for i in range(len(pacman.known_map)):
            for j in range(len(pacman.known_map[0])):
                cell = pacman.known_map[i][j]
                if cell is None:
                    pygame.draw.rect(screen, GRAY, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif cell == '#':  
                    pygame.draw.rect(screen, BLUE, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif cell == 'c':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.circle(screen, WHITE, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 10)
                elif cell == 'B':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.circle(screen, WHITE, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 5)
                elif cell == ' ':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif cell == 'P1':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.circle(screen, YELLOW, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)
                elif cell == 'P2':
                    pygame.draw.rect(screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.circle(screen, YELLOW, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)
        
        # Draw Pac-Man
        pygame.draw.circle(screen, YELLOW, 
                          (pacman.position[1] * CELL_SIZE + CELL_SIZE // 2, 
                           pacman.position[0] * CELL_SIZE + CELL_SIZE // 2), 
                          CELL_SIZE // 2)
        
        # Draw visible ghosts
        for ghost in ghosts:
            if ghost.position in pacman.visible_ghosts:
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
