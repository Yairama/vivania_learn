import numpy as np
from matplotlib import pyplot as plt

import cv2

from vivania_env.VivaniaEnv import VivaniaEnv

if __name__ == '__main__':
    env1 = VivaniaEnv(hidden=False)
    env2 = VivaniaEnv(hidden=False)
    obs = env1.reset()
    obs = env2.reset()
    counter = 0
    while True:
        # Take a random action
        action = env1.action_space.sample()
        obs, reward, done, info = env1.step(action)
        obs, reward, done, info = env2.step(action)
        # img_bgr = env.render(mode="human")
        # img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        # # cv2.imshow('Vivania Core',img_bgr)
        if done:
            break

    env1.close()
    env2.close()

#print(np.load('results/TD3_Vivania_Env_0.npy').tolist())