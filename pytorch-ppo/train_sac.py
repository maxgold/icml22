import numpy as np
import torch
import argparse
import os
import math
import gym
import gym_compete
import sys
import random
import time
import json
import copy

import pytorch_sac_ae.utils as utils
from pytorch_sac_ae.logger import Logger
from pytorch_sac_ae.video import VideoRecorder
from pytorch_sac_ae.sac_ae import StateSacAeAgent

from a2c_ppo_acktr.envs import make_vec_envs


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env_name', default='cheetah')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--multiagent', default=False, action='store_true')

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    for i in range(num_episodes):
        env.venv.venv.envs[0].needs_reset = True
        obs = env.reset()
        if args.multiagent:
            obs = obs[0]
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        i = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
                #action = env.action_space.sample()
                if args.multiagent:
                    action = (torch.tensor(action), torch.zeros(action.shape))
                else:
                    action = torch.tensor(action)
            obs, reward, done, _ = env.step(action)
            if args.multiagent:
                obs, reward, done = obs[0][0], reward[0][0][0], done.any()
            else:
                action = torch.tensor(action)
            video.record(env)
            episode_reward += reward
            i += 1

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


def make_agent(input_dim, action_shape, args, device):
    if args.agent == 'sac_ae':
        return StateSacAeAgent(
            input_dim=input_dim,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

#    env = dmc2gym.make(
#        domain_name=args.domain_name,
#        task_name=args.task_name,
#        seed=args.seed,
#        visualize_reward=False,
#        from_pixels=(args.encoder_type == 'pixel'),
#        height=args.image_size,
#        width=args.image_size,
#        frame_skip=args.action_repeat
#    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    env = make_vec_envs(args.env_name, args.seed, 1,
                         1, args.work_dir, device, False)
    env.seed(args.seed)


    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None, fps=60)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # the dmc2gym wrapper standardizes actions
    #assert env.action_space.low.min() >= -1
    #assert env.action_space.high.max() <= 1
    if args.multiagent:
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space[0].shape,
            action_shape=env.action_space[0].shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )

        agent = make_agent(
            input_dim=env.observation_space[0].shape[0],
            action_shape=env.action_space[0].shape,
            args=args,
            device=device
        )
    else:
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )

        agent = make_agent(
            input_dim=env.observation_space.shape[0],
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

                if episode % args.eval_freq == 0:
                    L.log('eval/episode', episode, step)
                    evaluate(env, agent, video, args.num_eval_episodes, L, step, args)
                    if args.save_model:
                        agent.save(model_dir, step)
                    if args.save_buffer:
                        replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            env.venv.venv.envs[0].needs_reset = True
            obs = env.reset()
            obs = obs[0]
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
            if args.multiagent:
                action = [torch.tensor(a) for a in action]
                action[1] = torch.zeros(action[1].shape)
                action = tuple(action)
            else:
                action = torch.tensor(action)
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
            if args.multiagent:
                action = (torch.tensor(action), torch.zeros(action.shape))
            else:
                action = torch.tensor(action)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)
        if args.multiagent:
            next_obs, reward, done = next_obs[0][0], reward[0][0][0], done.any()
        #reward = reward / 10

        # allow infinit bootstrap
        done_bool = float(done)
        #done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
        #    done
        #)
        episode_reward += reward

        replay_buffer.add(obs.cpu(), action[0], reward, next_obs.cpu(), done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
