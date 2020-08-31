import time
from flatland.envs.rail_env import RailEnv

class Node:
    """
    A node representing the position of the agent on the grid after a chosen path.
    """
    def __init__(self, parent=None, pos=None, fgh=(0,0,0)):
        """
        Node's attributes:
        parent: the previous tile in the path before the agent arrive at the current tile (None if it's the start or goal).
        pos: the coordinates of the tile on the grid.
        fn: the evaluated function f(n) - Estimated cost of the cheapest solution through n (current node).
        gn: cost from the start node to current node n.
        hn: estimated cost of the cheapest path from n to the goal node.

        fn, gn, hn are undetermined at initialization.
        """
        self.parent = parent
        self.pos = pos
        self.fn = fgh[0]
        self.gn = fgh[1]
        self.hn = fgh[2]
        self.trains = {}

    def __repr__(self):
        return ('({0},{1})'.format(self.pos, self.fn))

    def get_solution(self):
        """
        Return path taken from start to current node.
        """
        if self.parent is None:
            return None
        else:
            sol = []
            current = self

            while current is not None:
                sol.append(current.pos)
                current = current.parent
            sol.reverse()
            return sol


def print_grid(grid_,max_len):
    for line in grid_:
        for i in range(len(line)):
            if line[i] == -1:
                line[i] = (max_len - 1)*' ' + 'X'
            else:
                line[i] = (max_len - len(str(line[i])))*' '+str(line[i])
        print(line)

def print_current_grid(grid_,curr):
    time.sleep(1)
    print("Iter:")
    result = [row[:] for row in grid_]
    result[curr[0]][curr[1]] = '  C'
    print_grid(result,3)


def apply_solution(grid,sol):
    result = grid.copy()
    i = 1
    for pos in sol:
        result[pos[0]][pos[1]] = i
        i += 1
    return result


def grid_distance(start,goal):
    """
    Calculate grid distance (number of horizontal and vertical tiles from start to goal)
    between 2 nodes' positions.
    """
    return abs((goal.pos[0] - start.pos[0]) + (goal.pos[1] - start.pos[1]))


def planar_distance(start, goal):
    """
    Calculate the squared planar (or Euclidean) distance (using Pythagoras' theorem) between 2 nodes' positions.
    """
    return (goal.pos[0] - start.pos[0])**2 + (goal.pos[1] - start.pos[1])**2


def astar(grid, start_pos, goal_pos):
    """
    A* pathfinding on a grid where x and y axis are Python indexing - (0,0) on the top right corner.
    :param grid: 2-D array representing a the map. Non-passable tiles contains -1.
    :param start_x: 2-D array 1st dimension index representing horizontal position of start.
    :param start_y: 2-D array 1st dimension index representing vertical position of start.
    :param goal_x: 2-D array 1st dimension index representing horizontal position of goal.
    :param goal_y: 2-D array 1st dimension index representing vertical position of goal.
    """
    # Can only move horizontally or vertically
    moveset = [(1,0),(0,1),(0,-1),(-1,0)]

    # Max number of iterations (to avoid non-terminating problems)
    MAX_I = 6000

    # Grid's dimension
    GRID_LENGTH = len(grid[0])
    GRID_HEIGHT = len(grid)

    # COST PER MOVE:
    COST = 1

    start = Node(None, start_pos)
    goal = Node(None, goal_pos)

    closed_list = []
    open_list = []

    open_list.append(start)

    # Loop until goal is found or exceeds iteration max
    i = 0
    while not not open_list:
        if i >= MAX_I:
            print("No solutions found in time.")
            return None

        min_node_index = 0
        for i in range(len(open_list)):
            if open_list[i].fn < open_list[min_node_index].fn:
                min_node_index = i

        current = open_list.pop(min_node_index)
        closed_list.append(current)

        # Check if current is goal:
        if current.pos == goal_pos:
            sol = []
            return current.get_solution()

        # Get available tiles from current tile
        children = []

        for move in moveset:
            x = current.pos[0]
            y = current.pos[1]

            new_x = x + move[0]
            new_y = y + move[1]

            # Check if move is valid
            if (x > (GRID_HEIGHT - 1) or x < 0) or (y > (GRID_LENGTH - 1) or y < 0):
                continue
            if (grid[x][y] == -1):
                continue

            new_child = Node(current,(new_x,new_y))
            children.append(new_child)

        for child in children:
            child_in_closed = False
            for node in closed_list:
                if node.pos == child.pos:
                    child_in_closed = True
                    break

            if not child_in_closed:
                child.gn = current.gn + COST
                child.hn = grid_distance(child,goal)
                child.fn = child.gn + child.hn
            else:
                continue

            for node in open_list:
                if node.pos == child.pos and child.gn > node.gn:
                    break

            open_list.append(child)

def main():
    """
    grid = [[0,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0,-1, 0,-1, 0, 0],
            [0,-1, 0, 0,-1, 0],
            [0, 0, 0, 0,-1, 0]]
    """
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0,-1,-1,-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0]]
    start = (len(grid)-1,0)
    goal = (0,len(grid[0])-1)
    path = astar(grid,start,goal)

    max_len = len(str(len(path)))
    result = grid.copy()
    i = 1
    for pos in path:
        time.sleep(0.25)
        print("Time step {}:".format(i))
        result[pos[0]][pos[1]] = i
        i += 1

        print_grid(result,max_len)


if __name__ == "__main__":
    main()
