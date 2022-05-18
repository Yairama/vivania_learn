import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
from stable_baselines3.common.vec_env import DummyVecEnv

from vivania_env.VivaniaEnv import VivaniaEnv

#env = DummyVecEnv([lambda: VivaniaEnv(hidden=True) for i in range(6)])
env = VivaniaEnv(True)
# model = A2C("MlpPolicy", env, verbose=1)
model = A2C.load("VivaniaEnv", tensorboard_log='vivania')
model.set_env(env)
model.num_timesteps = 1000000
model.learn(total_timesteps=1000000)
model.save("VivaniaEnv")

# del model # remove to demonstrate saving and loading


env.hidden = False
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)













# import gym
#
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
# # Parallel environments
# from vivania_env.VivaniaEnv import VivaniaEnv
#
# env = VivaniaEnv(True)
#
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=500000)
# model.save("VivaniaEnv")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO.load("VivaniaEnv")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
# # Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
#
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO.load("ppo_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

