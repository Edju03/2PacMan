
import math
import random
from Map import Map
from Map import map_array

class Ghost:
    def __init__(self, position, color, map_obj, behavior = "chase", speed = 1):
        self.position = position
        self.color = color
        self.map = map_obj
        self.direction = (0, 0)
        self.scared = False
        self.move_queue = []
        self.behavior = behavior
        self.speed = speed
        self.move_counter = 0 # Used to slow down the ghost
    
    def move(self, pacman_position, other_ghost_positions):
        self.move_counter += 1
        if self.move_counter < 1/self.speed:
            return
        self.move_counter = 0
        
        if self.scared:
            # When scared, move randomly or away from Pacman
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            valid_directions = []
            
            for direction in directions:
                new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
                map_array = self.map.get_map()
                
                if (0 <= new_pos[0] < len(map_array) and 
                    0 <= new_pos[1] < len(map_array[0]) and 
                    map_array[new_pos[0]][new_pos[1]] != '#'):
                    
                    # Calculate distance to Pacman
                    dist_to_pacman = math.sqrt((new_pos[0] - pacman_position[0])**2 + 
                                              (new_pos[1] - pacman_position[1])**2)
                    valid_directions.append((direction, dist_to_pacman))
            
            if valid_directions:
                # Sort by distance (descending) to move away from Pacman
                valid_directions.sort(key=lambda x: -x[1])
                self.direction = valid_directions[0][0]
                self.position = (self.position[0] + self.direction[0], 
                                self.position[1] + self.direction[1])
        else:
            # Use A* to find the best move
            if self.behavior == "chase":
                best_move = self.chase_move(pacman_position)
                self.direction = best_move
                self.position = (self.position[0] + best_move[0], self.position[1] + best_move[1])
            if self.behavior == "random":
                best_move = self.random_move()
                self.direction = best_move
                self.position = (self.position[0] + best_move[0], self.position[1] + best_move[1])

    
    def chase_move(self, pacman_position):
        """Use A* to find the best move for Ghost"""
        # directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        # best_move = (0, 0)
        # best_cost = float('inf')
        #
        # for direction in directions:
        #     new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
        #     map_array = self.map.get_map()
        #
        #     # Check if the move is valid
        #     if (0 <= new_pos[0] < len(map_array) and
        #         0 <= new_pos[1] < len(map_array[0]) and
        #         map_array[new_pos[0]][new_pos[1]] != '#'):
        #
        #         # Calculate cost for this move
        #         cost = self.map.calculate_ghost_cost(new_pos, other_ghost_positions, pacman_position)
        #
        #         if cost < best_cost:
        #             best_cost = cost
        #             best_move = direction
        #
        # move_ret = (random.randint(-1, 1), random.randint(-1, 1))
        # if map_array[self.position[0] + move_ret[0]][self.position[1] + move_ret[1]] != '#':
        #     return move_ret
        path = self.map.astar(self.position, pacman_position)
        self.move_queue = path[1:]
        return path[0][0] - self.position[0], path[0][1] - self.position[1]

    def random_move(self):
        if self.move_queue:
            ret = self.move_queue.pop(0)
            return ret[0] - self.position[0], ret[1] - self.position[1]
        path = self.map.astar(self.position, self.map.random_valid_position())
        self.move_queue = path[1:]
        return path[0][0] - self.position[0], path[0][1] - self.position[1]
    
    def set_scared(self, scared):
        self.scared = scared
