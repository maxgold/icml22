from gym.envs.registration import register

register(
    id='matchingpennies-v0',
    entry_point='matchingpennies.envs:MatchingPennies',
    max_episode_steps=256,
    reward_threshold=-3.75,
)
