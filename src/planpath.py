import numpy as np
import copy
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from itertools import product

COST = 1
MAX_I = 500
MOVE_SET = [0,1,2,3,4]

class Node:
    def __init__(self, env, actions, parent=None):
        self.parent = parent
        self.env = env
        self.trains = env.agents
        self.actions_taken = actions
        self.goals = get_goals(self.trains)
        self.final = False
        self.f = 0
        self.g = 0
        self.h = 0

    def check_final(self):
        for i in range(len(self.trains)):
            agent = self.trains[i]
            #print(agent.position, self.goals[i])
            if (agent.status != RailAgentStatus.DONE_REMOVED) and (agent.status != RailAgentStatus.DONE):
                finished = False
                return False

        self.final = True
        return True

    def update_f(self):
        """
        Using planar distance to goal as h(n).
        """
        if self.check_final():
            return
        new_h = 0
        for i in range(len(self.trains)):
            agent = self.trains[i]
            if agent.position == None:
                continue
            else:
                new_h += (self.goals[i][0] - agent.position[0])**2 + (self.goals[i][1] - agent.position[1])**2
        self.h = new_h
        self.f = self.g + self.h

    def get_solution(self):
        if self.parent is None:
            return []
        else:
            sol = []
            #envs = []
            current = self

            while current is not None:
                sol.append(current.actions_taken)
                #envs.append(current.env)
                current = current.parent
            sol.reverse()
            #envs.reverse()
            #print("Supposed agents moves:")
            #for e in envs:
                #print(e.agents)

            return sol

    def __repr__(self):
        positions = {}
        for i in range(len(self.trains)):
            positions[i] = self.trains[i].position
        return ('({0},{1},{2})'.format(positions, self.actions_taken, self.f))

def old_search(src_env: RailEnv):
    #print(get_starts_and_goals(env))
    # Creates a schedule of 8 steps of random actions.
    schedule = []
    #print(env.agents)

    #agents_copy = env.agents[:]
    #agents_copy[0] = 0
    #print(env.agents)
    #print(agents_copy)
    env = copy.deepcopy(src_env)

    for _ in range(0, 8):
        _actions = {}
        for i in env.get_agent_handles():
            _actions[i] = np.random.randint(0, 5)
        schedule.append(_actions)
    print("Generated schedule:",schedule)
    return schedule

def search(src_env: RailEnv):
    schedule = []

    # Only work on copies of map
    env = copy.deepcopy(src_env)
    agents = copy.deepcopy(src_env.agents)
    n = len(agents) # Number of agents
    if n > 4:
        return

    # Cartesian product to produce all possible combinations of move per timestep for all agents
    # Will be of size 4^n -> Very inefficient !!!
    move_combinations = [p for p in product(MOVE_SET, repeat=n)]

    ### Initiate all agents (READY_TO_DEPART -> ACTIVE):
    # Checks if all agents are not initialised
    all_ready = True
    for agent in agents:
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            continue
        else:
            all_ready = False
            break
    # Initialise all agents
    if all_ready:
        _actions = {}
        for i in range(len(agents)):
            _actions[i] = 3
        env.step(_actions)
        schedule.append(_actions)

    # Initilise A* pathfinding
    closed_list = []
    open_list = []

    root = Node(env, _actions)
    root.update_f()

    open_list.append(root)

    iter = 0
    while len(open_list) > 0:
        if iter >= MAX_I:
            print("Timit limit reached, no solution found in time.")
            return None
        print("Iteration:",iter)
        iter += 1
        # Find node with smalles f
        min_node_index = 0
        for i in range(len(open_list)):
            if open_list[i].f < open_list[min_node_index].f:
                min_node_index = i

        current = open_list.pop(min_node_index)
        current_env = current.env
        closed_list.append(current)

        # Check if current is goal:
        if current.check_final():
            #print("Final node for solution:",current)
            schedule = current.get_solution()
            print("Generated schedule:",schedule)
            return schedule

        children = []

        # For each possible move
        for move in move_combinations:
            next_actions = {}
            for i in range(len(move)):  # Construct action dictionary
                next_actions[i] = move[i]

            # Create new hypothetical state
            new_env = copy.deepcopy(current_env)
            new_env.step(next_actions)
            #print(next_actions)
            ## TO DO: Implement checking for deadlocks for early exclusion of "useless" child
            ## TO DO: Implement checking for redundant children (actions that lead to same result)

            new_child = Node(new_env, next_actions, current)
            children.append(new_child)

        #print("Curr:",current)
        #print("Children:",children)
        for child in children:
            child_in_closed = False

            for node in closed_list:
                if node.env == child.env:
                    child_in_closed = True
                    #print("break1")
                    break

            if not child_in_closed:
                child.g = current.g + COST
                child.update_f()
            else:
                continue

            # Make sure same state doesn't exist with smaller g
            for node in open_list:
                if node.env == child.env and (child.g > node.g):
                    #print("break2")
                    break

            open_list.append(child)

    print("This shouldn't happen.")


def get_starts(agents):
    starts = {}
    for i in range(len(agents)):
        starts[i] = agents[i].initial_position
    return starts

def get_goals(agents):
    goals = {}
    for i in range(len(agents)):
        goals[i] = agents[i].target
    return goals
