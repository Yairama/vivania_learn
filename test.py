from matplotlib import pyplot as plt

import cv2

from vivania_env.VivaniaEnv import VivaniaEnv

if __name__ == '__main__':
    env = VivaniaEnv(hidden=False)
    obs = env.reset()
    counter = 0
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img_bgr = env.render(mode="human")
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Vivania Core',img_bgr)
        if done:
            break

    env.close()
