'''
Created on 20 aug. 2020

@author: Frits de Nijs
'''

import time
import threading

import pyglet
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.utils.graphics_pgl import RailViewWindow

def demo(window: RailViewWindow):
    """Demo script to check installation"""
    env = RailEnv(width=15, height=15, rail_generator=complex_rail_generator(
        nr_start_goal=10,
        nr_extra=1,
        min_dist=8,
        max_dist=99999), schedule_generator=complex_schedule_generator(), number_of_agents=5)

    env._max_episode_steps = int(15 * (env.width + env.height))
    env_renderer = RenderTool(env)

    while window.alive:
        obs, info = env.reset()
        env_renderer.reset()
        _done = False
        # Run a single episode here
        step = 0
        while not _done and window.alive:
            # Compute Action
            _action = {}
            for _idx, _ in enumerate(env.agents):
                _action[_idx] = np.random.randint(0, 5)
            obs, all_rewards, done, _ = env.step(_action)
            _done = done['__all__']
            step += 1
            env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False
            )
            time.sleep(0.1)

if __name__ == "__main__":

    window = RailViewWindow()

    demothread = threading.Thread(target=demo, args=(window,))
    demothread.start()

    pyglet.clock.schedule_interval(window.update_texture, 1/120.0)
    pyglet.app.run()

    demothread.join()
