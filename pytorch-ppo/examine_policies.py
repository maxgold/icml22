import copy
import glob
import os
import time
import pickle
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
from evaluation import evaluate, evaluate_multi, eval_movie_post, eval_winrate_post
from stable_baselines3.common.running_mean_std import RunningMeanStd

def set_reward_weight(envs, new_weight):
    envs.venv.venv.set_attr('move_reward_weight', new_weight)
    #envs.venv.venv.envs[0].env.env.move_reward_weight = new_weight

def unnormalize_obs(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
    """
    Helper to unnormalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: unnormalized observation
    """
    epsilon = 1e-8
    return (obs * np.sqrt(obs_rms.var + epsilon)) + obs_rms.mean

def normalize_obs(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
    """
    Helper to normalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: normalized observation
    """
    epsilon = 1e-8
    clip_obs = 10
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)

def choose_policyk(agento, k, args, p=0):
    agent = Agent(p)
    agent.obs_rms = [agento.obs_rms[k]]
    agent.policies = [agento.policies[k]]
    agent.weights = np.arange(1, len(agent.policies)+1).astype(float)
    agent.weights[-1] *= 2
    agent.weights /= agent.weights.sum()
    agent.init(args.num_processes)
    return agent

def get_obs_rms(env):
    return getattr(utils.get_vec_normalize(env), 'obs_rms', None)

class Agent:
    def __init__(self, playerid):
        self.playerid = playerid
        self.policies = []
        self.obs_rms = []

    def add_policy(self, policy, obs_rms):
        # TODO: need to add a wrapper that adds the obs_rms to each actor as we run it...
        self.policies.append(policy)
        self.obs_rms.append(obs_rms)
        self.weights = np.arange(1, len(self.policies)+1).astype(float)
        self.weights[-1] *= 2
        self.weights /= self.weights.sum()

    def sample_policy(self):
        ind = np.random.choice(range(len(self.policies)), p=self.weights)
        return self.policies[ind], self.obs_rms[ind]

    def init(self, num_envs):
        self.env_policies = []
        self.env_obs_rms = []
        for _ in range(num_envs):
            policy, obs_rms = self.sample_policy()
            self.env_policies.append(policy)
            self.env_obs_rms.append(obs_rms)

    def new_policy(self, ind):
        policy, obs_rms = self.sample_policy()
        self.env_policies[ind] = policy
        self.env_obs_rms[ind] = obs_rms

    def act(self, obs, recurrent, dones, cur_obs_rms):
        values = []
        actions = []
        logprobs = []
        rhs = []
        with torch.no_grad():
            for i, (o, r, d) in enumerate(zip(obs, recurrent, dones)):
                # Need to call unnormalize obs with envs.obs_rms then
                # renormalize with this obs_rms so that distribution matches training
                policy = self.env_policies[i]
                obs_rms = self.env_obs_rms[i]
                o = unnormalize_obs(o.cpu().numpy(), cur_obs_rms[self.playerid])
                o = normalize_obs(o, obs_rms[self.playerid])
                o = torch.tensor(o).cuda().float()

                tmp = self.env_policies[i].act(o.unsqueeze(0),r,d,deterministic=True)
                values.append(tmp[0])
                actions.append(tmp[1])
                logprobs.append(tmp[2])
                rhs.append(tmp[3])

        return torch.concat(values), torch.concat(actions), torch.concat(logprobs), torch.concat(rhs).unsqueeze(1)

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    p0_path = f"{args.log_dir}/p0_policies.pk"
    p1_path = f"{args.log_dir}/p1_policies.pk"
    os.makedirs(f"{log_dir}/eval_movies", exist_ok=True)

    with open(f"{args.log_dir}/p0_policies.pk", "rb") as handle:
        old_player0s = pickle.load(handle)
    with open(f"{args.log_dir}/p1_policies.pk", "rb") as handle:
        old_player1s = pickle.load(handle)
#    with open(f"{args.log_dir}/exploit_p0_policies.pk", "rb") as handle:
#        player0s = pickle.load(handle)
#    with open(f"{args.log_dir}/exploit_p1_policies.pk", "rb") as handle:
#        player1s = pickle.load(handle)

    import IPython as ipy; ipy.embed(colors='neutral')
    res = {}
    for k0 in range(36, len(old_player0s.policies), 12):
        for k1 in range(0, len(old_player1s.policies),12):
            print(f"EVALUATING PLAYER0 {k0} against Player1 {k1}")
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                                 args.gamma, args.log_dir, device, False, dense=args.dense)
            actor_critic0 = choose_policyk(old_player0s, k0, args, p=0)
            actor_critic1 = choose_policyk(old_player1s, k1, args, p=1)
            #utils.get_vec_normalize(envs).obs_rms = copy.deepcopy(old_player1s.obs_rms[-1])

            for j in range(3):
                eval_movie_post(f"{args.log_dir}/eval_movies/m_P{k0}_P{k1}_v{j}.mp4", actor_critic0, actor_critic1, get_obs_rms(envs), args.env_name, args.seed,
                         args.num_processes, eval_log_dir, device, args)

            win_rate = eval_winrate_post(actor_critic0, actor_critic1, get_obs_rms(envs), args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, args, num_episodes=100)
            res[(k0,k1)] = win_rate
        with open(f"{args.log_dir}/examine_res.pk", "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
