from mbpo.env.my_walker2d import MyWalker2dEnv
from mbpo.env.my_hopper import MyHopperEnv
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # env = MyWalker2dEnv()
    env = MyHopperEnv()
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