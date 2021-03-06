import time
import numpy as np
import threading

import pyglet

from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow
from math import floor

import planpath

def evalfun(debug = False, refresh = 0.1): # refresh default = 0.1
    # A list of (mapsize, agent count) tuples, change or extend this to test different sizes.
    #problemsizes = [(5, 1), (7, 2), (10,3), (13,4), (40, 20)]
    #problemsizes = [(5, 1), (5,2), (6,3), (7,3), (14,4), (8,5)]
    problemsizes = [(6, 3)]

    _seed = np.random.randint(1, 9999999)

    #_seed = 2

    print("Seed:",_seed)
    print("%10s\t%8s\t%9s" % ("Dimensions", "Success", "Runtime"))
    for problemsize in problemsizes:

        dimension = problemsize[0]
        NUMBER_OF_AGENTS = problemsize[1];

        # Create new environment.
        env = RailEnv(
                    width=dimension,
                    height=dimension,
                    rail_generator=complex_rail_generator(
                                            nr_start_goal=int(1.5 * NUMBER_OF_AGENTS),
                                            nr_extra=int(1.2 * NUMBER_OF_AGENTS),
                                            min_dist=int(floor(dimension / 2)),
                                            max_dist=99999,
                                            seed=0),
                    schedule_generator=complex_schedule_generator(),
                    malfunction_generator_and_process_data=None,
                    number_of_agents=NUMBER_OF_AGENTS)

        env_renderer = RenderTool(env, screen_width=1920, screen_height=1080)

        # Initialize positions.
        env.reset(random_seed=_seed)
        env_renderer.render_env(show=True, frames=False, show_observations=False)
        # Time the search.
        start = time.time()
        schedule = planpath.search(env)
        #schedule = planpath.better_search(env)
        duration = time.time() - start;

        if debug:
            env_renderer.render_env(show=True, frames=False, show_observations=False)
            time.sleep(refresh)

        # Validate that environment state is unchanged.
        assert env.num_resets == 1 and env._elapsed_steps == 0

        # Run the schedule
        success = False
        for action in schedule:
            _, _, _done, _ = env.step(action)
            success = _done['__all__']
            #print(env.agents)
            if debug:
                #for agent in env.agents:
                    #if agent.position:
                        #agent_y, agent_x = agent.position
                        #print(env.get_valid_directions_on_grid(agent_y,agent_x))
                print(action)
                env_renderer.render_env(show=True, frames=False, show_observations=False)
                time.sleep(refresh)

        # Print the performance of the algorithm
        print("%10s\t%8s\t%9.6f" % (str(problemsize), str(success), duration))


if __name__ == "__main__":

    _debug = True
    _refresh = 0.3 # default = 0.3

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evalfun, args=(_debug,_refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
