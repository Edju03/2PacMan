import math
import random
from queue import PriorityQueue
'''
'#' - wall
' ' - free cell
'c' - cookie
'B' - power pellet
'Gs' - spawn point for ghosts
'Ps' - spawn point for Pac-Man
'P1' - Pac-Man 1
'P2' - Pac-Man 2
'''

map_array = [
    ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#'],
    ['#','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','#'],
    ['#','B','#','#','c','#','#','#','c','#','c','#','#','#','c','#','#','B','#'],
    ['#','c','c','c','c','#','c','c','c','#','c','c','c','#','c','c','c','c','#'],
    ['#','#','c','#','c','#','c','#','c','#','c','#','c','#','c','#','c','#','#'],
    ['#','c','c','#','c','c','c','#','c','c','c','#','c','c','c','#','c','c','#'],
    ['#','c','#','#','#','#','c','#','#','#','#','#','c','#','#','#','#','c','#'],
    ['#','c','c','c','c','c','c','c','c','r','c','c','c','c','c','c','c','c','#'],
    ['#','#','c','#','#','#','c','#','#','-','#','#','c','#','#','#','c','#','#'],
    ['c','c','c','c','c','#','c','#','s','p','o','#','c','#','c','c','c','c','c'],
    ['#','#','c','#','c','#','c','#','#','#','#','#','c','#','c','#','c','#','#'],
    ['#','c','c','#','c','c','c','c','c','c','c','c','c','c','c','#','c','c','#'],
    ['#','c','#','#','#','#','c','#','#','#','#','#','c','#','#','#','#','c','#'],
    ['#','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','#'],
    ['#','#','#','c','#','#','#','c','#','#','#','c','#','#','#','c','#','#','#'],
    ['#','c','c','c','#','c','c','c','c','P1','c','c','c','c','#','c','c','c','#'],
    ['#','B','#','c','#','c','#','c','#','#','#','c','#','c','#','c','#','B','#'],
    ['#','c','#','c','c','c','#','c','c','c','c','c','#','c','c','c','#','c','#'],
    ['#','c','#','#','#','c','#','#','#','c','#','#','#','c','#','#','#','c','#'],
    ['#','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','#'],
    ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
]

class Map:

    def __init__(self):

        self.map_array = map_array
        self.height = len(map_array)
        self.width = len(map_array[0])
        self.cookie_positions = []
        self.power_pellet_positions = []
        self.ghost_spawn_points = []
        self.pacman_spawn_points = []
        self.free_positions = []
        self.wall_positions = []
        self.initialize_positions()

    def get_map(self):
        return self.map_array
    
    def initialize_positions(self):
        for i, row in enumerate(self.map_array):
            for j, cell in enumerate(row):
                if cell == 'c':  # cookies
                    self.cookie_positions.append((i, j))
                elif cell == 'P':  # power pellet
                    self.power_pellet_positions.append((i, j))
                elif cell == 'B':  # power pellet (big cookie)
                    self.power_pellet_positions.append((i, j))
                elif cell in ['s', 'p', 'o', 'r']:
                    self.ghost_spawn_points.append((i, j))
                elif cell == 'P1':  # pacman spawn
                    self.pacman_spawn_points.append((i, j))
                elif cell == '#':
                    self.wall_positions.append((i, j))
                else:
                    self.free_positions.append((i, j))
    
    def get_spawn_points(self):
        return {
            'ghost': self.ghost_spawn_points,
            'pacman': self.pacman_spawn_points
        }
    
    def is_valid_position(self, position):
        """Check if a position is valid (not a wall)"""
        i, j = position
        if i < 0 or i >= len(self.map_array) or j < 0 or j >= len(self.map_array[0]) or (i, j) in self.wall_positions:
            return False
        return True

    def random_valid_position(self):
        while True:
            i, j = random.randrange(len(self.map_array)), random.randrange(len(self.map_array[0]))
            if self.is_valid_position((i, j)):
                return (i, j)
    
    def get_cookie_density(self, position, radius=3):
        """Calculate cookie density around a position"""
        i, j = position
        count = 0
        for x in range(max(0, i-radius), min(len(self.map_array), i+radius+1)):
            for y in range(max(0, j-radius), min(len(self.map_array[0]), j+radius+1)):
                if (x, y) in self.cookie_positions:
                    count += 1
        return count
    
    def calculate_pacman_cost(self, position, ghost_positions, visited_positions=None):
        """Calculate cost for Pacman movement based on cookie density and ghost proximity"""
        # First check if position is valid (not a wall)
        if not self.is_valid_position(position):
            return float('inf')  # Return infinite cost for walls
            
        cookie_density = self.get_cookie_density(position)
        
        # Ghost proximity penalty
        ghost_penalty = 0
        for ghost_pos in ghost_positions:
            distance = math.sqrt((position[0] - ghost_pos[0])**2 + (position[1] - ghost_pos[1])**2)
            if distance < 3:  # Close ghost
                ghost_penalty += 10 / (distance + 0.1)  
        
        # Revisit penalty
        revisit_penalty = 0
        if visited_positions and position in visited_positions:
            revisit_penalty = 1
            
        # Lower cost is more desirable
        return 10 - min(cookie_density, 8) + ghost_penalty + revisit_penalty

    def calculate_ghost_cost(self, position, other_ghost_positions, last_pacman_position=None):
        """Calculate cost for Ghost movement based on coverage and Pacman location"""
        # First check if position is valid (not a wall)
        if not self.is_valid_position(position):
            return float('inf')  # Return infinite cost for walls
            
        # Penalty for being close to other ghosts (for better coverage)
        ghost_proximity_penalty = 0
        for other_ghost_pos in other_ghost_positions:
            if position != other_ghost_pos: 
                distance = math.sqrt((position[0] - other_ghost_pos[0])**2 + (position[1] - other_ghost_pos[1])**2)
                if distance < 5:  # Close to another ghost
                    ghost_proximity_penalty += 5 / (distance + 0.1)
        
        # Bonus for being close to last known Pacman position
        pacman_bonus = 0
        if last_pacman_position:
            distance = math.sqrt((position[0] - last_pacman_position[0])**2 + 
                                (position[1] - last_pacman_position[1])**2)
            pacman_bonus = 10 / (distance + 1)
            
        # Lower cost means more desirable
        return 10 + ghost_proximity_penalty - pacman_bonus

    def astar(self, start, goal, cost_func = lambda a, b: max(a[0] - b[0], a[1]-b[1])):
        q = PriorityQueue()
        parents = {}
        dists = {}
        q.put((0, start))
        dists[start] = 0
        while q:
            cur = q.get()[1]
            if cur == goal:
                break
            dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            for d in dirs:
                neighbor = (cur[0] + d[0], cur[1] + d[1])
                dist = dists[cur] + 1
                if not self.is_valid_position(neighbor):
                    continue
                if neighbor not in dists or dist < dists[neighbor]:
                    dists[neighbor] = dist
                    parents[neighbor] = cur
                    q.put((dists[neighbor] + cost_func(neighbor, goal), neighbor))
        path = []
        current = goal
        while current in parents:
            path.append(current)
            current = parents[current]
        return path[::-1]



    def update_cell(self, row, col, value):
        """Update a cell in the map array"""
        if 0 <= row < len(self.map_array) and 0 <= col < len(self.map_array[0]):
            position = (row, col)
            cur_value = self.map_array[row][col]
            # Only Pac-Man should eat cookies, ghosts should not
            if value in ['P1', 'P2'] and cur_value == 'c':
                if position in self.cookie_positions:
                    self.cookie_positions.remove(position)
                    self.free_positions.append(position)
            elif value in ['P1', 'P2'] and cur_value == 'B':
                if position in self.power_pellet_positions:
                    self.power_pellet_positions.remove(position)
                    self.free_positions.append(position)
            
            # Ensure Pac-Man 1 and Pac-Man 2 don't intersect
            if (value == 'P1' and 'P2' in [self.map_array[row][col]]) or (value == 'P2' and 'P1' in [self.map_array[row][col]]):
                return False  
                
            self.map_array[row][col] = value
            return True