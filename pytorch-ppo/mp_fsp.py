import copy
import glob
import os
import time
import pickle
import matchingpennies
from collections import deque

import gym
import gym_compete
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate, evaluate_multi, eval_movie
from stable_baselines3.common.running_mean_std import RunningMeanStd


class MpPolicy:
    def __init__(self):
        self.vals = np.array([1.1,0.1])
        self.counts = np.array([1,1])
        self.eps = .2
        #self.vals = np.zeros(2) + 1e-10

    def update(self, obs, action, reward):
        if isinstance(action, list):
            for a, r in zip(action, reward):
                self.vals[action] += reward
                self.counts[action] += 1
        else:
            self.vals[action] += reward
            self.counts[action] += 1

    @property
    def probs(self):
        probs = np.exp(self.vals/self.counts) / np.exp(self.vals/self.counts).sum()
        return probs

    def act(self, obs):
        probs = self.probs
        actions = []
        if isinstance(obs, list):
            for _ in obs:
                if np.random.rand() < self.eps:
                    actions.append(np.random.choice([0,1]))

                else:
                    actions.append(np.random.choice([0,1],p=probs))
        else:
            actions.append(np.random.choice([0,1],p=probs))
        return np.array(actions)

class Agent:
    def __init__(self, playerid):
        self.playerid = playerid
        self.policies = []

    def add_policy(self, policy):
        # TODO: need to add a wrapper that adds the obs_rms to each actor as we run it...
        self.policies.append(policy)
        self.weights = np.arange(1, len(self.policies)+1).astype(float)
        self.weights[-1] *= 2
        self.weights /= self.weights.sum()

    def sample_policy(self):
        ind = np.random.choice(range(len(self.policies)), p=self.weights)
        return self.policies[ind]

    def init(self, num_envs):
        self.env_policies = []
        for _ in range(num_envs):
            policy = self.sample_policy()
            self.env_policies.append(policy)

    def new_policy(self, ind):
        policy = self.sample_policy()
        self.env_policies[ind] = policy

    def act(self, obs, *args, **kwargs):
        values = []
        actions = []
        logprobs = []
        rhs = []
        with torch.no_grad():
            for i, o in enumerate(obs):
                policy = self.env_policies[i]
                action = policy.act(o)
                actions.append(action)
        actions = np.array(actions)
        #if len(actions) > 1:
        #    actions = np.concatenate(actions)

        return actions

class EnvWrapper:
    def __init__(self, env_name, num_envs):
        self.envs = []
        for _ in range(num_envs):
            self.envs.append(gym.make(env_name))

    def step(self, actions):
        steps = []
        for i, a in enumerate(actions):
            steps.append(self.envs[i].step(a))
        obs = [s[0] for s in steps]
        r = [s[1] for s in steps]
        dones = [s[2] for s in steps]
        infos = [s[3] for s in steps]
        return obs, np.array(r).T, dones, infos

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return obs
    

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = EnvWrapper(args.env_name, args.num_processes)
    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, device, False, dense=args.dense)
    envs.reset()
    
    agent0 = MpPolicy()
    agent1 = MpPolicy()


    player0_policies = []
    player1_policies = []
    obs_rms_cache = []


    episode_rewards0 = deque(maxlen=10)
    episode_rewards1 = deque(maxlen=10)

    train_stats = []
    eval_p0 = []
    eval_p1 = []


    cache_freq = args.cache_freq
    learn_agent0 = agent0
    old_agent0 = Agent(0)
    old_agent0.add_policy(copy.deepcopy(agent0))
    old_agent0.init(args.num_processes//2)
    learn_agent1 = agent1
    old_agent1 = Agent(1)
    old_agent1.add_policy(copy.deepcopy(agent1))
    old_agent1.init(args.num_processes//2)
    p0numep = 0
    p1numep = 0

    #num_updates = int(
    #    args.num_env_steps) // args.num_steps // args.num_processes
    num_updates = 50
    for j in range(num_updates):
        eps = -1
        p0draws, p1draws = 0, 0
        w0, w1 = 0, 0
        epoch_p0ep, epoch_p1ep = 0, 0
        while eps < args.num_episodes:
            start = time.time()

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    action0a = agent0.act([None for _ in range(args.num_processes//2)])
                    action0b = old_agent0.act([None for _ in range(args.num_processes//2)])
                    action1a = old_agent1.act([None for _ in range(args.num_processes//2)])
                    action1b = agent1.act([None for _ in range(args.num_processes//2)])
                    action0 = np.concatenate((action0a.squeeze(),action0b.squeeze()))
                    action1 = np.concatenate((action1a.squeeze(),action1b.squeeze()))
                #action0 = torch.tensor(action0)[:,None]
                #action1 = torch.tensor(action1)[:,None]

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(zip(action0, action1))
                envs.reset()
                agent0.update(None, action0, reward[0])
                agent1.update(None, action1, reward[1])

                for i, info in enumerate(infos[:args.num_processes//2]):
                    d = True
                    if 'episode' in info[0].keys():
                        old_agent1.new_policy(i)
                        epoch_p0ep += 1
                        if "winner" in info[0].keys():
                            d = False
                            w0 += 1
                        if "winner" in info[1].keys():
                            d = False
                        eps += 1
                        episode_rewards0.append(info[0]['episode']['r'])
                        if d:
                            p0draws += 1
                        p0numep += 1
                        if p0numep % cache_freq == 0:
                            print(f"ADDING POLICY !! Current length {len(old_agent0.policies)}")
                            old_agent0.add_policy(copy.deepcopy(agent0))
                for i, info in enumerate(infos[args.num_processes//2:]):
                    if 'episode' in info[1].keys():
                        old_agent0.new_policy(i)
                        epoch_p1ep += 1
                        if "winner" in info[0].keys():
                            d = False
                        if "winner" in info[1].keys():
                            d = False
                            w1 += 1
                        episode_rewards1.append(info[1]['episode']['r'])
                        if d:
                            p1draws += 1
                        p1numep += 1
                        if p1numep % cache_freq == 0:
                            old_agent1.add_policy(copy.deepcopy(agent1))


        if j % args.log_interval == 0 and len(episode_rewards0) > 1:
            total_num_steps = args.num_steps * args.num_processes
            end = time.time()
            print("=============================================================================================")
            print(f"Updates {j} player0 rewards {np.mean(episode_rewards0)} player1 rewards {np.mean(episode_rewards1)} FPS: {int(total_num_steps / (end-start))}")
            print(f"Updates {j}, Player0 wins {round(w0/epoch_p0ep,2)} Player1 wins {round(w1/epoch_p1ep,2)} P0Draws {round(p0draws/epoch_p0ep,2)} P1Draws {p1draws/epoch_p1ep} P0 Episodes {epoch_p0ep} P1 episodes {epoch_p1ep}")
            print("=============================================================================================")
            train_stats.append((w0,w1,p0draws,p1draws))
    import IPython as ipy; ipy.embed(colors='neutral')

if __name__ == "__main__":
    main()
