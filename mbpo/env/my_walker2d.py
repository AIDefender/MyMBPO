import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os.path import expanduser
import matplotlib.pyplot as plt
from mbpo.env.configs.walker2d import params

class MyWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        self.model.opt.gravity[:] = np.array(params['gravity'])

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

if __name__ == '__main__':
    env = MyWalker2dEnv()
    # env = gym.make('Humanoid-v2')
    # env.env.model.opt.gravity[:] = np.array([150.,0,200])
    env.reset()
    while True:
        arr = env.render(mode='rgb_array')
        plt.imshow(arr)
        plt.draw()
        plt.pause(0.001)
        act = env.action_space.sample()
        _, _, done, _ = env.step(act)
        # if done:
        #     break