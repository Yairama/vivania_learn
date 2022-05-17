import os
import gym
import numpy as np
import torch
from gym import wrappers
import time

from numpy import dtype

from ReplayBuffer import ReplayBuffer
from TD3 import TD3
from vivania_env.VivaniaEnv import VivaniaEnv

env_name = "Vivania_Env"  # Nombre del entorno (puedes indicar cualquier entorno continuo que quieras probar aquí)
seed = 0  # Valor de la semilla aleatoria
start_timesteps = 1e4  # Número de of iteraciones/timesteps durante las cuales el modelo elige una acción al azar, y después de las cuales comienza a usar la red de políticas
eval_freq = 5e3  # Con qué frecuencia se realiza el paso de evaluación (después de cuántos pasos timesteps)
max_timesteps = 9e5  # Número total de iteraciones/timesteps
save_models = True  # Check Boolean para saber si guardar o no el modelo pre-entrenado
expl_noise = 0.15  # Ruido de exploración: desviación estándar del ruido de exploración gaussiano
batch_size = 100  # Tamaño del bloque
discount = 0.99  # Factor de descuento gamma, utilizado en el cáclulo de la recompensa de descuento total
tau = 0.005  # Ratio de actualización de la red de objetivos
policy_noise = 0.35  # Desviación estándar del ruido gaussiano añadido a las acciones para fines de exploración
noise_clip = 0.5  # Valor máximo de ruido gaussiano añadido a las acciones (política)
policy_freq = 2  # Número de iteraciones a esperar antes de actualizar la red de políticas (actor modelo)

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print("---------------------------------------")
print("Configuración: %s" % file_name)
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env = VivaniaEnv(True)


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        #print(f"Episodio {_} *** avg_rewrag: {avg_reward}")
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


env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = np.array(env.observation_space.sample().reshape(1,-1)).shape[1]
action_dim = env.action_space.shape[0]
max_action = float(7)

policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
evaluations = [evaluate_policy(policy)]

# policy.load(file_name, directory="./pytorch_models")
# evaluations = np.load("./results/%s.npy" % (file_name)).tolist()
# episode_num=
# total_timesteps=
# episode_reward = 0
# episode_timesteps = 0

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env.max_episode_steps
save_env_vid = False
if save_env_vid:
    env = wrappers.Monitor(env, monitor_dir, force=True)
    env.reset()

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

# Iniciamos el bucle principal con un total de 500,000 timesteps
while total_timesteps < max_timesteps:

    # Si el episodio ha terminado
    if done:

        # Si no estamos en la primera de las iteraciones, arrancamos el proceso de entrenar el modelo
        if total_timesteps != 0:
            print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                         policy_freq)

        # Evaluamos el episodio y guardamos la política si han pasado las iteraciones necesarias
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
            evaluations.append(evaluate_policy(policy))
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)

        # Cuando el entrenamiento de un episodio finaliza, reseteamos el entorno
        obs = env.reset()

        # Configuramos el valor de done a False
        done = False

        # Configuramos la recompensa y el timestep del episodio a cero
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Antes de los 10000 timesteps, ejectuamos acciones aleatorias
    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:  # Después de los 10000 timesteps, cambiamos al modelo
        action = policy.select_action(np.array(obs, dtype=np.float16))
        # Si el valor de explore_noise no es 0, añadimos ruido a la acción y lo recortamos en el rango adecuado
        if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(
                0, 7)

    # El agente ejecuta una acción en el entorno y alcanza el siguiente estado y una recompensa
    action = action.astype(int)
    new_obs, reward, done, _ = env.step(action)

    # Comprobamos si el episodio ha terminado
    done_bool = 0 if episode_timesteps + 1 == env.max_episode_steps else float(done)

    # Incrementamos la recompensa total
    episode_reward += reward

    # Almacenamos la nueva transición en la memoria de repetición de experiencias (ReplayBuffer)
    replay_buffer.add((np.array(obs).reshape(-1), np.array(new_obs).reshape(-1), action, reward, done_bool))

    # Actualizamos el estado, el timestep del número de episodio, el total de timesteps y el número de pasos desde la última evaluación de la política
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Añadimos la última actualización de la política a la lista de evaluaciones previa y guardamos nuestro modelo
evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
