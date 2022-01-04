import numpy as np
from .agent import Agent
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box



class AntEnv(Agent, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        agentid,
        xml_file="gym_ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        n_agents=2
    ):
        Agent.__init__(self, agentid, xml_file, nagents=n_agents)
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.GOAL = 10
        self.frame_skip = 5

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost


    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.get_cfrc_ext())
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.get_qpos()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def set_goal(self, goal):
        self.GOAL = goal

    def before_step(self):
        self._xy_position_before = self.get_body_com("torso")[:2].copy()

    def after_step(self, action):
        #self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - self._xy_position_before) / self.env.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        if self.GOAL < 0:
            forward_reward = -x_velocity
        else:
            forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return reward, done, info

#    def _get_obs(self):
#        position = self.sim.data.qpos.flat.copy()
#        velocity = self.sim.data.qvel.flat.copy()
#        contact_force = self.contact_forces.flat.copy()
#
#        if self._exclude_current_positions_from_observation:
#            position = position[2:]
#
#        observations = np.concatenate((position, velocity, contact_force))
#
#        return observations

    def _get_obs(self):
        '''
        Return agent's observations
        '''
        # TODO: this is where to implement lag...
        my_pos = self.get_qpos()
        other_pos = self.get_other_qpos()
        my_vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)

#        obs = np.concatenate(
#            [my_pos.flat, my_vel.flat, cfrc_ext.flat,
#             other_pos.flat]
#        )
        obs = np.concatenate(
            [my_pos.flat, my_vel.flat]
        )
        return obs

    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        xpos = self.get_body_com('torso')[0]
        if self.GOAL > 0 and xpos > self.GOAL:
            return True
        elif self.GOAL < 0 and xpos < self.GOAL:
            return True
        return False

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
