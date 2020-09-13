import numpy as np
from queue import PriorityQueue
import copy
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from itertools import product

COST = 1
MAX_I = 5000
MOVE_SET = [0,1,2,3,4]
ACTIONS_SET = [RailEnvActions.DO_NOTHING, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.STOP_MOVING]
IDLE_PENALTY = 1 # If a train stopped moving, add to g(n) so as to minimize the stopping in solution.
IDLE_LIMIT = 15

class Node:
    def __init__(self, env, actions, parent=None):
        """
        Class representing state of all agents.
        """
        self.parent = parent    # Parent state
        self.env = env  # Current env object
        self.trains = env.agents    # Current agents in env
        self.actions_taken = actions    # Actions applied (to parent's env) that lead to the creation of this node
        self.goals = get_goals(self.trains)
        self.idling = 0
        self.final = False
        self.f = 0
        self.g = 0
        self.h = 0
        self.idles = {} # idles[agent_id] contains the number of turns the agent has been in moving = False
        if self.parent is not None:
            self.idles = self.parent.idles
            self.update_idles()
        else:
            for i in range(len(env.agents)):
                self.idles[i] = 0

    def check_final(self):
        """
        Checks to see if node represents a goal state.
        """
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
                new_h += 0
            else:
                new_h += (self.goals[i][0] - agent.position[0])**2 + (self.goals[i][1] - agent.position[1])**2
        self.h = new_h
        self.f = self.g + self.h

    def update_idles(self):
        """
        Update the idle counts for trains in state.
        Also adds idle penalty accordingly.
        """
        for i in range(len(self.trains)):
            agent = self.trains[i]
            if not agent.moving and agent.status == RailAgentStatus.ACTIVE:
                self.idles[i] = self.idles[i] + 1
                #self.idling += 1
                #self.g += 1
                #self.f = self.g + self.h
            else:
                self.idles[i] = 0

    def get_positions(self):
        pos = []
        for i in range(len(self.trains)):
            agent = self.trains[i]
            pos.append(agent.position)
        return pos

    def idle_too_long(self):
        """
        If a train has stopped moving for 15 turns, this node should not be expanded (deadlock).
        """
        for i in range(len(self.trains)):
            if self.idles[i] >= IDLE_LIMIT:
                return True
        return False

    def get_solution(self):
        """
        Goes through all the parent nodes and append action taken to retrieve the solution (path to this node).
        """
        if self.parent is None:
            return []
        else:
            sol = []
            current = self

            while current is not None:
                sol.append(current.actions_taken)
                current = current.parent
            sol.reverse()

            return sol

    def __repr__(self):
        """
        String representation of the node.
        """
        positions = {}
        for i in range(len(self.trains)):
            positions[i] = self.trains[i].position
        return ('({0},{1},{2})'.format(positions, self.actions_taken, self.f))

    def __eq__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        return (self.__repr__() == other.__repr__())

    def __ne__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        return (self.__repr__() != other.__repr__())

    def __lt__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        if (self.get_positions() == other.get_positions()) and (self.actions_taken == other.actions_taken):
            return self.f < other.f
        else:
            return False

    def __gt__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        if (self.get_positions() == other.get_positions()) and (self.actions_taken == other.actions_taken):
            return self.f > other.f
        else:
            return False

    def __le__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        if (self.get_positions() == other.get_positions()) and (self.actions_taken == other.actions_taken):
            return self.f <= other.f
        else:
            return False

    def __ge__(self,other):
        """
        Added comparation support to use with PriorityQueue.
        """
        if (self.get_positions() == other.get_positions()) and (self.actions_taken == other.actions_taken):
            return self.f >= other.f
        else:
            return False


def get_starts(agents):
    """
    Get the starting positions of all trains.
    """
    starts = {}
    for i in range(len(agents)):
        starts[i] = agents[i].initial_position
    return starts

def get_goals(agents):
    """
    Get the goal positions of all trains.
    """
    goals = {}
    for i in range(len(agents)):
        goals[i] = agents[i].target
    return goals


def check_good_action(env,move):
    """
    THIS IS NOT WORKING, IT USUALLY SKIPS PASS TOO MANY BRANCHES THUS MAKES A PROBLEM MUCH LONG TO SOLVE
    Checks if the action is redundant/invalid (LEFT/RIGHT on a straight track,etc.)
    If one agent received invalid action, return False
    """
    agents = env.agents
    for i in range(len(agents)):

        # If an agent is done, any actions on this agent is redudant
        if (agents[i].position is None) and (ACTIONS_SET[move[i]] != RailEnvActions.DO_NOTHING):
            return False
        elif (agents[i].position is None):
            continue
        new_direction, transition_valid = env.check_action(agents[i],ACTIONS_SET[move[i]])

        if transition_valid is False:
            return False
        else:
            continue
    return True


def search(src_env: RailEnv):
    """
    Implements simple A* pathfinding algorithm for multi-agent scheduling.
    Uses Node class from above and Python's PriorityQueue for the open list for instant retrieval.
    """
    schedule = []   # Contains the final answer

    # Only work on copies of map
    env = copy.deepcopy(src_env)
    agents = copy.deepcopy(src_env.agents)
    n = len(agents) # Number of agents

    #if n > 4:   # Usually doesn't terminate (in time) for more than this number of agents
        #return

    # Cartesian product to produce all possible combinations of move per timestep for all agents
    # Upper bound will be of size 5^n -> Very inefficient !!!
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
            _actions[i] = 2
        env.step(_actions)
        schedule.append(_actions)

    # Initilise A* pathfinding
    closed_list = []
    open_list = PriorityQueue()

    # Creates the first node (that contains the action that puts the agents on the map)
    root = Node(env, _actions)
    root.update_f()

    # Adds first node to open list, this is O(log(n)) operation (n is current size of open list -> grows with time)
    open_list.put((root.f,root))

    iter = 0    # Use for time/iterations limit
    last_current = None

    while not open_list.empty():
        #if iter >= MAX_I:
            #print("Timit limit reached, no solution found in time.")
            #return None

        iter += 1

        # Find node with smalles f
        """
        Because the open list was implemented with a Priority Queue with no further extensions,
        the tie-breaking rule is simply FIFO (or left most node first if we extend the tree from left to right).
        """
        current_f,current = open_list.get() # Get leftmost node with smallest f from open list (O(1) retrieval)

        # If duplicate states from priority queue insertion
        if last_current is not None:
            if current == last_current:
                #print("Same current as last current.")
                continue
        #print(current)

        # Removed due to false alarm on needed STOP actions for avoiding collisions
        # If current has idles for too long:
        #if current.idle_too_long():
            #continue

        current_env = current.env
        closed_list.append(current)

        # Check if current is goal:
        if current.check_final():
            #print("Final node for solution:",current)
            schedule = current.get_solution()
            print("Generated schedule:",schedule)
            return schedule

        # List of children from expansion of current node
        children = []

        # For each possible move in move combinations
        for move in move_combinations:
            next_actions = {}   # Dict of next actions

            bad_move = False    # Suppose the current combination does not contain a bad move
            for i in range(len(move)):  # Construct action dictionary
                next_actions[i] = move[i]

                if check_good_action(current_env,move) is False: # For each agent, checks if suggested move is a bad move
                    bad_move = True
                    #print("bad move",move)
                    break

            if not bad_move:    # If no move in the current action set was a bad move
                # Create new hypothetical state
                new_env = copy.deepcopy(current_env)
                new_env.step(next_actions)
                ## TO DO: Implement checking for deadlocks for early exclusion of "useless" child : DONE
                ## TO DO: Implement checking for redundant children (actions that lead to same result) : DONE

                ## For each agent, check their tiles to see valid moves.
                ## For each move, if move is not in agent's valid, break (do not create child)
                new_child = Node(new_env, next_actions, current)

                #if new_child.idle_too_long():
                    #print("IDLED TOO LONG",new_child.idles)
                    #continue

                children.append(new_child)
                new_child.update_f()

        appended_children = 0   # Count the number of children created in this step (for debug purposes)

        # For each child created
        for child in children:
            child_in_closed = False

            # Checks if child is in closed list
            for node in closed_list:
                if node == child:
                    child_in_closed = True
                    #print("break1")
                    break

            if child_in_closed:
                continue

            # Make sure same state doesn't exist with smaller g
            """
            REMOVED BECAUSE THIS IS LINEAR SEARCH,
            SOLVED BY SIMPLY PUTTING INTO PRIORITY QUEUE AND WHEN
            WE ARE GETTING CURRENT NODE, CHECK IF THE NEXT CURRENT IS SAME AS LAST CURRENT
            IF IT IS THE SAME, REMOVE (SO INSTEAD OF O(N) SEARCH + O(LOG(N)) ADD, WE ONLY
            HAVE O(LOG(N)) ADD)
            for node in open_list.queue:
                if node[1] > child:
                    #print("break2")
                    break
            """

            open_list.put((child.f,child))
            appended_children += 1
        last_current = current
        #print("Appended children and weight: ", appended_children, child.f)

    # If the program reaches here, this means there are no nodes left in open list and solution has not been found.
    print("This shouldn't happen.")
