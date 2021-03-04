import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done