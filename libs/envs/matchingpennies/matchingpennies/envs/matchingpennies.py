import gym
import numpy as np

import re
import os
import random
import numpy as np

from tempfile import mkdtemp
import contextlib
from shutil import copyfile, rmtree
from pathlib import Path


@contextlib.contextmanager
def make_temp_directory(prefix=''):
    temp_dir = mkdtemp(prefix)
    try:
        yield temp_dir
    finally:
        rmtree(temp_dir)


class MatchingPennies(gym.Env):
    '''
    Matching pennies environment
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,))
        self.action_space = gym.spaces.Discrete(2)

    def step(self, actions):
        # returns obs, reward, done, info
        action0, action1 = actions
        if (action0 == 0) and (action1 == 0):
            reward = (7, -1)
            info0 = {"episode": {"r": reward[0]}, "winner": True}
            info1 = {"episode": {"r": reward[1]}}
        elif (action0 == 1) and (action1 == 1):
            reward = (1, -1)
            info0 = {"episode": {"r": reward[0]}, "winner": True}
            info1 = {"episode": {"r": reward[1]}}
        elif (action0==0) and (action1==1):
            reward = (-1, 1)
            info0 = {"episode": {"r": reward[0]}}
            info1 = {"episode": {"r": reward[1]}, "winner": True}
        elif (action0==1) and (action1==0):
            reward = (-1, 1)
            info0 = {"episode": {"r": reward[0]}}
            info1 = {"episode": {"r": reward[1]}, "winner": True}
        ob = (np.ones(10), np.ones(10))
        return ob, reward, True, [info0, info1]

    def reset(self, env_id=None, same=False):
        ob = (np.ones(10), np.ones(10))
        return ob
