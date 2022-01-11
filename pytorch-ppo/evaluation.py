import numpy as np
import torch
import imageio

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs

def get_obs_rms(env):
    return getattr(utils.get_vec_normalize(env), 'obs_rms', None)


def make_movie(movie_path, frames):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(movie_path, 0, 1, (width,height))
    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, 1,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def evaluate_multi(actor_critic0, actor_critic1, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, 1,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards0 = []
    eval_episode_rewards1 = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic0.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    w0 = 0
    w1 = 0
    draws = 0

    while len(eval_episode_rewards0) < 10:
        with torch.no_grad():
            _, action0, _, eval_recurrent_hidden_states = actor_critic0.act(
                obs[0],
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
            _, action1, _, eval_recurrent_hidden_states = actor_critic1.act(
                obs[1],
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step((action0, action1))

        eval_masks = torch.tensor(
            [[0.0] if any(done_) else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            d = True
            if 'episode' in info[0].keys():
                if "winner" in info[0].keys():
                    d = False
                    w0 += 1
                eval_episode_rewards0.append(info[0]['episode']['r'])
            if 'episode' in info[1].keys():
                if "winner" in info[1].keys():
                    d = False
                    w1 += 1
                eval_episode_rewards1.append(info[1]['episode']['r'])
            if ("episode" in info[0].keys()) and ("episode" in info[1].keys()):
                if d:
                    draws += 1

    eval_envs.close()

    print(f"P0 reward: {np.mean(eval_episode_rewards0)}, P1 reward: {np.mean(eval_episode_rewards1)}")
    print(f"P0 wins: {w0}, P1 wins: {w1}, draws: {draws}")
    return w0, w1, draws

def eval_movie(movie_path, actor_critic0, actor_critic1, obs_rms, env_name, seed, num_processes, eval_log_dir, device, args):
    eval_envs = make_vec_envs(env_name, seed + num_processes, 1,
                              None, eval_log_dir, device, True, dense=args.dense)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards0 = []
    eval_episode_rewards1 = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic0.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    frames = []
    rew0, rew1 = 0,0

    done = np.array((False,))
    while not done.any():

        with torch.no_grad():
            _, action0, _, eval_recurrent_hidden_states = actor_critic0.act(
                obs[0],
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
            _, action1, _, eval_recurrent_hidden_states = actor_critic1.act(
                obs[1],
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, rew, done, infos = eval_envs.step((action0, action1))
        rew0 += rew[0][0]
        rew1 += rew[0][1]
        frames.append(eval_envs.render(mode="rgb_array"))

        eval_masks = torch.tensor(
            [[0.0] if any(done_) else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

    eval_envs.close()
    imageio.mimsave(movie_path, frames, fps=50)
    print("DONE WITH MOVIE")
    print(f"PLAYER0 got {rew0} and PLAYER1 got {rew1}")

def eval_winrate(player, actor_critic0, actor_critic1, obs_rms, env_name, seed, num_processes, eval_log_dir, device, args, num_episodes=100):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, dense=args.dense)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards0 = []
    eval_episode_rewards1 = []

    obs = eval_envs.reset()
    if player == 0:
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic1.recurrent_hidden_state_size, device=device)
    else:
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic0.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    rew0, rew1 = 0,0
    win0 = 0
    win1 = 0
    total = 0

    done = np.array((False,))
    while total < num_episodes:

        with torch.no_grad():
            _, action0, _, eval_recurrent_hidden_states = actor_critic0.act(
                obs[0],
                eval_recurrent_hidden_states,
                eval_masks,
                get_obs_rms(eval_envs))
            _, action1, _, eval_recurrent_hidden_states = actor_critic1.act(
                obs[1],
                eval_recurrent_hidden_states,
                eval_masks,
                get_obs_rms(eval_envs))

        # Obser reward and next obs
        obs, rew, done, infos = eval_envs.step((action0, action1))
        for i, info in enumerate(infos):
            d = True
            if 'episode' in info[0].keys():
                if "winner" in info[0].keys():
                    if player == 0:
                        actor_critic0.new_policy(i)
                    d = False
                    win0 += 1
                total += 1
            if 'episode' in info[1].keys():
                if "winner" in info[1].keys():
                    if player == 1:
                        actor_critic1.new_policy(i)
                    d = False
                    win1 += 1

        eval_masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

    eval_envs.close()
    return win0 / total, win1/total, (total-win0-win1)/total
