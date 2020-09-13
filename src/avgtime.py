import time
import threading
import numpy as np
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


    _seeds = np.random.randint(1, 99, 15)
    #_seed = 2984379


    avg = {}
    for x in range(1,4):
        avg[x] = []
        problemsizes = [(5,x), (6,x), (8,x), (10,x), (15,x)]
        for problemsize in problemsizes:
            avg_time = 0
            successes = 0
            for seed in _seeds:
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

                env.reset(random_seed=int(seed))

                if len(env.agents) != NUMBER_OF_AGENTS:
                    continue

                start = time.time()
                schedule = planpath.search(env)
                duration = time.time() - start;

                assert env.num_resets == 1 and env._elapsed_steps == 0

                # Run the schedule
                success = False;
                if schedule is not None:
                    for action in schedule:
                        _, _, _done, _ = env.step(action)
                        success = _done['__all__']
                        #print(env.agents)

                    if success:
                        avg_time += duration
                        print("Success:",problemsize,seed,duration)
                        successes +=1

                    else:
                        print("Bad schedule - failed.")
                else:
                    print("Couldn't find solution for seed: ",duration,seed)
                    env_renderer.render_env(show=True, frames=False, show_observations=False)

            avg_time = avg_time / successes
            avg[x].append((problemsize[0],avg_time))
    #print("%10s\t%8s\t%9s" % ("Dimensions", "Success", "Runtime"))



        #

        # Initialize positions.


        # Time the search.


        #if debug:
            #env_renderer.render_env(show=True, frames=False, show_observations=False)
            #time.sleep(refresh)

        # Validate that environment state is unchanged.


        # Print the performance of the algorithm
        #print("%10s\t%8s\t%9.6f" % (str(problemsize), str(success), duration))
        print(avg)


if __name__ == "__main__":

    _debug = True
    _refresh = 0.01 # default = 0.3

    if (_debug):
        window = RailViewWindow()

    evalthread = threading.Thread(target=evalfun, args=(_debug,_refresh,))
    evalthread.start()

    if (_debug):
        pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
        pyglet.app.run()

    evalthread.join()
