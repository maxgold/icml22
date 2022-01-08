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
from evaluation import evaluate, evaluate_multi, eval_movie
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
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic0 = Policy(
        envs.observation_space[0].shape,
        envs.action_space[0],
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic0.to(device)

    agent0 = algo.PPO(
        actor_critic0,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    actor_critic1 = Policy(
        envs.observation_space[1].shape,
        envs.action_space[1],
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic1.to(device)

    agent1 = algo.PPO(
        actor_critic1,
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


    rollouts0 = RolloutStorage(args.num_steps, args.num_processes//2,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic0.recurrent_hidden_state_size)
    opp_rollouts0 = RolloutStorage(args.num_steps, args.num_processes//2,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic0.recurrent_hidden_state_size)
    rollouts1 = RolloutStorage(args.num_steps, args.num_processes//2,
                              envs.observation_space[1].shape, envs.action_space[1],
                              actor_critic1.recurrent_hidden_state_size)
    opp_rollouts1 = RolloutStorage(args.num_steps, args.num_processes//2,
                              envs.observation_space[1].shape, envs.action_space[1],
                              actor_critic1.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts0.obs[0].copy_(obs[0][:args.num_processes//2])
    rollouts0.to(device)
    opp_rollouts0.obs[0].copy_(obs[0][args.num_processes//2:])
    opp_rollouts0.to(device)
    rollouts1.obs[1].copy_(obs[1][args.num_processes//2:])
    rollouts1.to(device)
    opp_rollouts1.obs[1].copy_(obs[1][:args.num_processes//2])
    opp_rollouts1.to(device)

    episode_rewards0 = deque(maxlen=10)
    episode_rewards1 = deque(maxlen=10)

    train_stats = []
    eval_p0 = []
    eval_p1 = []


    cache_freq = 1000

    learn_agent0 = agent0
    old_agent0 = Agent(0)
    old_agent0.add_policy(copy.deepcopy(actor_critic0), get_obs_rms(envs))
    old_agent0.init(args.num_processes//2)
    learn_agent1 = agent1
    old_agent1 = Agent(1)
    old_agent1.add_policy(copy.deepcopy(actor_critic1), get_obs_rms(envs))
    old_agent1.init(args.num_processes//2)
    p0numep = 0
    p1numep = 0

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if j < 250:
            set_reward_weight(envs, np.linspace(1,0,250)[j])
        else:
            set_reward_weight(envs, 0)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent0.optimizer, j, num_updates,
                agent0.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent1.optimizer, j, num_updates,
                agent1.optimizer.lr if args.algo == "acktr" else args.lr)

        eps = -1
        p0draws = 0
        p1draws = 0
        w0 = 0
        w1 = 0
        # TODO: when should we reset obs_rms?
        while eps < args.num_episodes:
            print(eps)
            start = time.time()

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value0a, action0a, action_log_prob0a, recurrent_hiddenstate0a = agent0.actor_critic.act(
                        rollouts0.obs[step], rollouts0.recurrent_hidden_states[step],
                        rollouts0.masks[step])
                    value0b, action0b, action_log_prob0b, recurrent_hiddenstate0b = old_agent0.act(
                        opp_rollouts0.obs[step], opp_rollouts0.recurrent_hidden_states[step],
                        opp_rollouts0.masks[step], get_obs_rms(envs))
                    value1a, action1a, action_log_prob1a, recurrent_hiddenstate1a = old_agent1.act(
                        opp_rollouts1.obs[step], opp_rollouts1.recurrent_hidden_states[step],
                        opp_rollouts1.masks[step], get_obs_rms(envs))
                    value1b, action1b, action_log_prob1b, recurrent_hiddenstate1b = agent1.actor_critic.act(
                        rollouts1.obs[step], rollouts1.recurrent_hidden_states[step],
                        rollouts1.masks[step])
                    action0 = torch.concat((action0a,action0b))
                    value0 = torch.concat((value0a,value0b))
                    action_log_prob0 = torch.concat((action_log_prob0a,action_log_prob0b))
                    recurrent_hiddenstate0 = torch.concat((recurrent_hiddenstate0a,recurrent_hiddenstate0b))
                    action1 = torch.concat((action1a,action1b))
                    value1 = torch.concat((value1a,value1b))
                    action_log_prob1 = torch.concat((action_log_prob1a,action_log_prob1b))
                    recurrent_hiddenstate1 = torch.concat((recurrent_hiddenstate1a,recurrent_hiddenstate1b))

                # Obser reward and next obs
                obs, reward, done, infos = envs.step((action0, action1))

                for i, info in enumerate(infos[:16]):
                    d = True
                    if 'episode' in info[0].keys():
                        old_agent1.new_policy(i)
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
                            old_agent0.add_policy(copy.deepcopy(actor_critic0), get_obs_rms(envs))
                for i, info in enumerate(infos[16:]):
                    if 'episode' in info[1].keys():
                        old_agent0.new_policy(i)
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
                            old_agent1.add_policy(copy.deepcopy(actor_critic1), get_obs_rms(envs))

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks0 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[0].keys() else [1.0]
                     for info in infos])
                bad_masks1 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[1].keys() else [1.0]
                     for info in infos])
                rollouts0.insert(obs[0][:args.num_processes//2], recurrent_hiddenstate0a, action0a,
                                 action_log_prob0a, value0a, reward[0][:args.num_processes//2], masks[:args.num_processes//2], bad_masks0[:args.num_processes//2])
                opp_rollouts1.insert(obs[1][:args.num_processes//2], recurrent_hiddenstate1a, action1a,
                                     action_log_prob1a, value1a, reward[1][:args.num_processes//2], masks[:args.num_processes//2], bad_masks1[:args.num_processes//2])
                opp_rollouts0.insert(obs[0][args.num_processes//2:], recurrent_hiddenstate0b, action0b,
                                     action_log_prob0b, value0b, reward[0][args.num_processes//2:], masks[args.num_processes//2:], bad_masks0[args.num_processes//2:])
                rollouts1.insert(obs[1][args.num_processes//2:], recurrent_hiddenstate1b, action1b,
                                 action_log_prob1b, value1b, reward[1][args.num_processes//2:], masks[args.num_processes//2:], bad_masks1[args.num_processes//2:])

            with torch.no_grad():
                next_value0 = actor_critic0.get_value(
                    rollouts0.obs[-1], rollouts0.recurrent_hidden_states[-1],
                    rollouts0.masks[-1]).detach()
                next_value1 = actor_critic1.get_value(
                    rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1],
                    rollouts1.masks[-1]).detach()


            rollouts0.compute_returns(next_value0, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)
            rollouts1.compute_returns(next_value1, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss0, action_loss0, dist_entropy0 = agent0.update(rollouts0)
            value_loss1, action_loss1, dist_entropy1 = agent1.update(rollouts1)

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
                    actor_critic0,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + "0.pt"))
                torch.save([
                    actor_critic1,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + "1.pt"))

            if j % args.log_interval == 0 and len(episode_rewards0) > 1:
                total_num_steps = args.num_steps * args.num_processes
                end = time.time()
                print("=============================================================================================")
                print(f"Updates {j} player0 rewards {np.mean(episode_rewards0)} player1 rewards {np.mean(episode_rewards1)} FPS: {int(total_num_steps / (end-start))}")
                print(f"Updates {j}, Player0 wins {w0} Player1 wins {w1} P0Draws {p0draws} P1Draws {p1draws}")
                print("=============================================================================================")
                train_stats.append((w0,w1,p0draws,p1draws))
                with open("logs.pk", "wb") as handle:
                    pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)


            if (j % args.cache_interval == 0) and (len(episode_rewards0) > 1):
                player0_policies.append(copy.deepcopy(actor_critic0))
                player1_policies.append(copy.deepcopy(actor_critic1))
                obs_rms_cache.append(copy.deepcopy(utils.get_vec_normalize(envs).obs_rms))

        if (args.eval_interval is not None and len(episode_rewards0) > 1
                and j % args.eval_interval == 0):
            obs_rms0 = [utils.get_vec_normalize(envs).obs_rms[0], obs_rms_cache[0][1]]
#            print("evaluating current player0 against first player1")
#            ep0_w0, ep0_w1, ep0_draw = evaluate_multi(actor_critic0, player1_policies[0], obs_rms0, args.env_name, args.seed,
#                     args.num_processes, eval_log_dir, device)
#
#            obs_rms1 = [obs_rms_cache[0][0], utils.get_vec_normalize(envs).obs_rms[1]]
#            print("evaluating current player1 against first player0")
#            ep1_w0, ep1_w1, ep1_draw = evaluate_multi(player0_policies[0], actor_critic1, obs_rms1, args.env_name, args.seed,
#                     args.num_processes, eval_log_dir, device)
#            eval_p0.append([ep0_w0, ep0_w1, ep0_draw])
#            eval_p1.append([ep1_w0, ep1_w1, ep1_draw])
#            with open("logs.pk", "wb") as handle:
#                pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)
            eval_movie(f"movies/m{j}.mp4", actor_critic0, actor_critic1, obs_rms0, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
            with open("p0_policies.pk", "wb") as handle:
                pickle.dump(old_agent0, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("p1_policies.pk", "wb") as handle:
                pickle.dump(old_agent1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("logs.pk", "wb") as handle:
        pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)
    import IPython as ipy; ipy.embed(colors='neutral')

if __name__ == "__main__":
    main()
