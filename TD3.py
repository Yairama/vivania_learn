import torch
import torch.nn.functional as F


# Construir el proceso de entrenamiento en una clase
from Actor import Actor
from Critic import Critic


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        #self.device = torch.device("dml" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clipping=0.5, policy_freq=2):
        for it in range(iterations):

            # Paso 4: Tomamos una muestra de transiciones (s, s’, a, r) de la memoria.
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # Paso 5: A partir del estado siguiente s', el Actor del Target ejecuta la siguiente acción a'.

            next_action = self.actor_target.forward(next_state)

            # Paso 6: Añadimos ruido gaussiano a la siguiente acción a' y lo cortamos para tenerlo en el rango de valores aceptado por el entorno.
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clipping, noise_clipping)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Paso 7: Los dos Críticos del Target toman un par (s’, a’) como entrada y devuelven dos Q-values Qt1(s’,a’) y Qt2(s’,a’) como salida.
            target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)

            # Paso 8: Nos quedamos con el mínimo de los dos Q-values: min(Qt1, Qt2). Representa el valor aproximado del estado siguiente.
            target_Q = torch.min(target_Q1, target_Q2)

            # Paso 9: Obtenemos el target final de los dos Crítico del Modelo, que es: Qt = r + γ * min(Qt1, Qt2), donde γ es el factor de descuento.
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Paso 10: Los dos Críticos del Modelo toman un par (s, a) como entrada y devuelven dos Q-values Q1(s,a) y Q2(s,a) como salida.
            current_Q1, current_Q2 = self.critic.forward(state, action)

            # Paso 11: Calculamos la pérdida procedente de los Crítico del Modelo: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Paso 12: Propagamos hacia atrás la pérdida del crítico y actualizamos los parámetros de los dos Crítico del Modelo con un SGD.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Paso 13: Cada dos iteraciones, actualizamos nuestro modelo de Actor ejecutando el gradiente ascendente en la salida del primer modelo crítico.
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()  ##OJO ME DEJÉ EL LOSS
                self.actor_optimizer.step()

                # Paso 14: Todavía cada dos iteraciones, actualizamos los pesos del Actor del Target usando el promedio Polyak.
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Paso 15: Todavía cada dos iteraciones, actualizamos los pesos del target del Crítico usando el promedio Polyak.
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Método para guardar el modelo entrenado
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    # Método para cargar el modelo entrenado
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))
