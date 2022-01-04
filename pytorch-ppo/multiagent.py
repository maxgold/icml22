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

def set_reward_weight(envs, new_weight):
    envs.venv.venv.envs[0].env.env.move_reward_weight = new_weight

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


    rollouts0 = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic0.recurrent_hidden_state_size)
    rollouts1 = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[1].shape, envs.action_space[1],
                              actor_critic1.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts0.obs[0].copy_(obs[0])
    rollouts0.to(device)
    rollouts1.obs[1].copy_(obs[1])
    rollouts1.to(device)

    episode_rewards0 = deque(maxlen=10)
    episode_rewards1 = deque(maxlen=10)

    train_stats = []
    eval_p0 = []
    eval_p1 = []

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
#        if j < 250:
#            set_reward_weight(envs, np.linspace(1,0,250)[j])
#        else:
#            set_reward_weight(envs, 0)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent0.optimizer, j, num_updates,
                agent0.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent1.optimizer, j, num_updates,
                agent1.optimizer.lr if args.algo == "acktr" else args.lr)

        eps = -1
        draws = 0
        w0 = 0
        w1 = 0
        # TODO: when should we reset obs_rms?
        while eps < args.num_episodes:
            start = time.time()

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value0, action0, action_log_prob0, recurrent_hidden_states0 = actor_critic0.act(
                        rollouts0.obs[step], rollouts0.recurrent_hidden_states[step],
                        rollouts0.masks[step])
                    value1, action1, action_log_prob1, recurrent_hidden_states1 = actor_critic1.act(
                        rollouts1.obs[step], rollouts1.recurrent_hidden_states[step],
                        rollouts1.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step((action0, action1))

                for info in infos:
                    d = True
                    if 'episode' in info[0].keys():
                        if "winner" in info[0].keys():
                            d = False
                            w0 += 1
                        eps += 1
                        episode_rewards0.append(info[0]['episode']['r'])
                    if 'episode' in info[1].keys():
                        if "winner" in info[1].keys():
                            d = False
                            w1 += 1
                        episode_rewards1.append(info[1]['episode']['r'])
                    if ("episode" in info[0].keys()) and ("episode" in info[1].keys()):
                        if d:
                            draws += 1

                # If done then clean the history of observations.
                if args.num_processes == 1:
                    done = [done[0][0]]
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks0 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[0].keys() else [1.0]
                     for info in infos])
                bad_masks1 = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info[1].keys() else [1.0]
                     for info in infos])
                rollouts0.insert(obs[0], recurrent_hidden_states0, action0,
                                action_log_prob0, value0, reward[0], masks, bad_masks0)
                rollouts1.insert(obs[1], recurrent_hidden_states1, action1,
                                action_log_prob1, value1, reward[1], masks, bad_masks1)

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
                print(f"Updates {j}, Player0 wins {w0} Player1 wins {w1} Draws {draws}")
                print("=============================================================================================")
                train_stats.append((w0,w1,draws))
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
    with open("logs.pk", "wb") as handle:
        pickle.dump([train_stats,eval_p0,eval_p1], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
