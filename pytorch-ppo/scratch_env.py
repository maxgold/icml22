from gym_compete.new_envs import utils

import numpy as np
import gym
from gym import Env, spaces
from gym_compete.new_envs.multi_agent_scene import MultiAgentScene
from gym_compete.new_envs.agents import *
from gym_compete.new_envs.utils import create_multiagent_xml
from pytorch_sac_ae.video import VideoRecorder
import os
import six

import IPython as ipy; ipy.embed(colors='neutral')
env_id = "gym-run-to-goal-ants-v0"
env = gym.make(env_id, agent_names=["ant", "ant"], init_pos=[(-1,0,.75),(1,0,.75)])
#env = gym.make(env_id, agent_names=["ant"], init_pos=[(-1,0,.75)])
#env_id = "run-to-goal-ants-v0"
#env_id = "Ant-v3"
#env = gym.make(env_id)
video = VideoRecorder("tmp", fps=60)
video.init(enabled=True)
tmp = env.render(mode="rgb_array")
env.reset()
i = 0
done = False
rewards = []
while not done:
    i += 1
    video.record(env)
    _, r, done, _ = env.step(env.action_space.sample())
    if isinstance(done, tuple):
        done = any(done)
    rewards.append(r)
video.save("gymtest.mp4")
