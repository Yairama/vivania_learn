import os

import numpy as np
import torch
from TD3 import TD3

from vivania_env.VivaniaEnv import VivaniaEnv

# Selección del dispositivo (CPU o GPU)
device = torch.device("cpu")


env_name = "Vivania_Env"
seed = 0

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print("---------------------------------------")
print("Configuración: %s" % (file_name))
print("---------------------------------------")

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        print(f"Episodio {_} *** avg_rewrag: {avg_reward}")
        while not done:
            #print(obs)
            action = policy.select_action(np.array(obs, dtype=np.float16))
            action = action.astype(int)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("-------------------------------------------------")
    print("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
    print("-------------------------------------------------")
    return avg_reward


eval_episodes = 10
save_env_vid = True
env = VivaniaEnv(hidden=False)
max_episode_steps = env.max_episode_steps
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

env.reset()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = np.array(env.observation_space.sample().reshape(1,-1)).shape[1]
action_dim = env.action_space.shape[0]
max_action = float(8)
policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')
_ = evaluate_policy(policy, eval_episodes=eval_episodes)