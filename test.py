import numpy as np
from matplotlib import pyplot as plt

import cv2
from stable_baselines3 import PPO

from vivania_env.VivaniaEnv import VivaniaEnv

env = VivaniaEnv(True)
model = PPO.load('baseline_models/VivaniaEnv 18052022.zip')

# del model # remove to demonstrate saving and loading


env.hidden = False
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

#print(np.load('results/TD3_Vivania_Env_0.npy').tolist())