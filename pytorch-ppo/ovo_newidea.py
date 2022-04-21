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
from a2c_ppo_acktr.model import Policy2
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate, evaluate_multi, eval_movie_new
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
        self.weights[-10:] *= 100
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
                # TODO here should take care of normalizing with obs_rms....
                # Need to call unnormalize obs with envs.obs_rms then
                # renormalize with this obs_rms
                policy = self.env_policies[i]
                obs_rms = self.env_obs_rms[i]
                o = unnormalize_obs(o.cpu().numpy(), cur_obs_rms[self.playerid])
                o = normalize_obs(o, obs_rms[self.playerid])
                o = torch.tensor(o).cuda().float()

                tmp = self.env_policies[i].act(o.unsqueeze(0),r,d)
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
    os.makedirs(f"{log_dir}/movies", exist_ok=True)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, dense=args.dense)

    actor_critic = Policy2(
        envs.observation_space[0].shape,
        envs.action_space[0],
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO2(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    player0_policies = []
    player1_policies = []
    obs_rms_cache = []


    rollouts0 = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic.recurrent_hidden_state_size)
    rollouts1 = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[1].shape, envs.action_space[1],
                              actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts0.obs[0].copy_(obs[0])
    rollouts0.to(device)
    rollouts1.obs[1].copy_(obs[1])
    rollouts1.to(device)

    episode_rewards0 = deque(maxlen=100)
    episode_rewards1 = deque(maxlen=100)

    train_stats = []
    eval_p0 = []
    eval_p1 = []


    p0numep = 0
    p1numep = 0

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        #if j < 250:
        #    set_reward_weight(envs, np.linspace(1,0,250)[j])
        #else:
        #    set_reward_weight(envs, 0)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent0.optimizer, j, num_updates,
                agent0.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent1.optimizer, j, num_updates,
                agent1.optimizer.lr if args.algo == "acktr" else args.lr)

        eps = -1
        p0draws, p1draws = 0, 0
        w0, w1 = 0, 0
        epoch_p0ep, epoch_p1ep = 0, 0
        # TODO: when should we reset obs_rms?
        while eps < args.num_episodes:
            start = time.time()

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value0, action0, action_log_prob0, recurrent_hiddenstate0 = agent.act(
                        rollouts0.obs[step], rollouts0.recurrent_hidden_states[step],
                        rollouts0.masks[step], 0)
                    value1, action1, action_log_prob1, recurrent_hiddenstate1 = agent.act(
                        rollouts1.obs[step], rollouts1.recurrent_hidden_states[step],
                        rollouts1.masks[step], 1)

                # Obser reward and next obs
                obs, reward, done, infos = envs.step((action0, action1))

                for i, info in enumerate(infos[:16]):
                    d = True
                    if 'episode' in info[0].keys():
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
                for i, info in enumerate(infos[16:]):
                    if 'episode' in info[1].keys():
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

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks0 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[0].keys() else [1.0]
                     for info in infos])
                bad_masks1 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[1].keys() else [1.0]
                     for info in infos])
                rollouts0.insert(obs[0], recurrent_hiddenstate0, action0,
                                 action_log_prob0, value0, reward[0], masks, bad_masks0)
                rollouts1.insert(obs[0], recurrent_hiddenstate1, action1,
                                 action_log_prob1, value1, reward[1], masks, bad_masks1)

            with torch.no_grad():
                obs = rollouts0.obs[-1]
                player0t = (torch.zeros(obs.shape[0], 1)).to(device).long()
                player1t = (torch.ones(obs.shape[0], 1)).to(device).long()
                next_value0 = actor_critic.get_value(
                    rollouts0.obs[-1], rollouts0.recurrent_hidden_states[-1],
                    rollouts0.masks[-1], player0t).detach()
                next_value1 = actor_critic.get_value(
                    rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1],
                    rollouts1.masks[-1], player1t).detach()


            rollouts0.compute_returns(next_value0, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)
            rollouts1.compute_returns(next_value1, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss0, action_loss0, dist_entropy0 = agent.update(rollouts0, 0)
            #value_loss1, action_loss1, dist_entropy1 = agent.update(rollouts1, 1)

            rollouts0.after_update()
            rollouts1.after_update()


            # save for every interval-th episode or for the last epoch
            if (j % args.save_interval == 0
                    or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + "0.pt"))

            if j % args.log_interval == 0 and len(episode_rewards0) > 1:
                total_num_steps = args.num_steps * args.num_processes
                end = time.time()
                print("=============================================================================================")
                print(f"Updates {j} player0 rewards {np.mean(episode_rewards0)} player1 rewards {np.mean(episode_rewards1)} FPS: {int(total_num_steps / (end-start))}")
                print(f"Updates {j}, Player0 wins {round(w0/epoch_p0ep,2)} Player1 wins {round(w1/epoch_p1ep,2)} P0Draws {round(p0draws/epoch_p0ep,2)} P1Draws {p1draws/epoch_p1ep} P0 Episodes {epoch_p0ep} P1 episodes {epoch_p1ep}")
                print("=============================================================================================")
                train_stats.append((w0,w1,p0draws,p1draws))
                with open(f"{log_dir}/logs.pk", "wb") as handle:
                    pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)


        if (args.eval_interval is not None and len(episode_rewards0) > 1
                and j % args.eval_interval == 0):
            eval_movie_new(f"{args.log_dir}/movies/m{j}.mp4", actor_critic, get_obs_rms(envs), args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, args)
            with open(f"{args.log_dir}/p0_policies.pk", "wb") as handle:
                pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{args.log_dir}/logs.pk", "wb") as handle:
        pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
