import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

from collections import deque
import random
import cv2

from torchvision import transforms

# --- Utilitaires Images (Gardés pour compatibilité et visualisation) ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()  # converts to [C,H,W] float32
])

N_ACTIONS = 16
ANGLES = np.linspace(-np.pi, np.pi, N_ACTIONS)

log_dir = f"logs/vector_dqn_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir)

def render_env_to_rgb(env, img_size=84):
    """
    Render a ShepherdEnv object to an RGB image (numpy array).
    UTILE pour la visualisation humaine, même si l'agent MLP utilise des vecteurs.
    """
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # white background

    def world_to_pixel(pos):
        x = int((pos[0] + 1) / 2 * (img_size - 1))
        y = int((1 - (pos[1] + 1) / 2) * (img_size - 1))  # flip y-axis for image
        return x, y

    # Draw goal
    gx, gy = world_to_pixel(env.goal)
    cv2.circle(canvas, (gx, gy), radius=5, color=(0, 255, 0), thickness=-1)  # green

    # Draw shepherd
    sx, sy = world_to_pixel(env.shepherd)
    cv2.circle(canvas, (sx, sy), radius=5, color=(255, 0, 0), thickness=-1)  # red

    # Draw obstacle
    ox, oy = world_to_pixel(env.obstacle)
    cv2.circle(canvas, (ox, oy), radius=5, color=(0, 0, 0), thickness=-1)  # black

    # Draw sheep
    for s in env.sheep:
        x, y = world_to_pixel(s)
        cv2.circle(canvas, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # blue

    return canvas

# --- 1. Architecture MLP (Vecteurs) au lieu de CNN ---
class MlpQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        
        # Réseau dense simple : Entrée -> 256 -> 256 -> Actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. Buffer de Replay (Adapté pour vecteurs) ---
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Conversion numpy -> torch float
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()

        # Note: On a supprimé la logique .permute() car ce ne sont plus des images
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    

# --- 3. Agent DQN Vectoriel ---
class VectorDQNAgent:
    def __init__(
        self,
        obs_dim,              
        n_actions,
        lr=1e-3,              
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=200_000,
        device="cpu"
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma

        # Utilisation de MlpQNetwork au lieu de ImageQNetwork
        self.q_net = MlpQNetwork(obs_dim, n_actions).to(device)
        self.target_net = MlpQNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

    def select_action(self, state):
        self.steps += 1

        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps / self.eps_decay)

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            # Ajout dimension batch [1, obs_dim]
            q_values = self.q_net(state.unsqueeze(0).to(self.device))

        return q_values.argmax(dim=1).item()
    
    def train_step(self, batch_size=64): # Batch size augmenté par défaut
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions)

        # Bellman target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        # Empêche les modifications trop violentes du réseau
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        if self.steps % 100 == 0:
            writer.add_scalar(
                "debug/q_mean",
                q_values.mean().item(),
                self.steps
            )

        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# --- 4. Fonction d'Evaluation (Adaptée Vecteurs) ---
@torch.no_grad()
def evaluate_agent(env, agent, episodes=5):
    total_rewards = []
    total_lengths = []

    for _ in range(episodes):
        state = env.reset() # Renvoie un vecteur
        state = torch.from_numpy(state).float()
        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            q_values = agent.q_net(state.unsqueeze(0).to(agent.device))
            action_idx = q_values.argmax(dim=1).item()
            angle = ANGLES[action_idx]

            # step prend [angle]
            next_state, reward, done, _ = env.step([angle])
            
            state = torch.from_numpy(next_state).float()
            ep_reward += reward
            ep_len += 1

        total_rewards.append(ep_reward)
        total_lengths.append(ep_len)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(total_lengths),
    }


# --- 5. Boucle d'Entraînement (Renommée train_vector_dqn) ---
def train_vector_dqn(
    env,
    eval_env,
    agent,
    episodes=1000,
    batch_size=256,
    target_update=500, # Mise à jour plus fréquente pour MLP
    eval_every=10,
    eval_episodes=5,
    train_freq=4,
    save_path="models/dqn_agent"
    ):

    step_count = 0
    global_step = 0
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset() # Vecteur direct du gym env
        state = torch.from_numpy(state).float()
        done = False
        ep_reward = 0
        ep_losses = []

        while not done:
            # 1. Action
            action_idx = agent.select_action(state)
            angle = ANGLES[action_idx]

            # 2. Step
            next_state, reward, done, _ = env.step([angle])
            next_state_t = torch.from_numpy(next_state).float()

            # 3. Stockage (Plus de rendu image ici !)
            agent.replay.push(state, action_idx, reward, next_state_t, done)

            # 4. Update state
            state = next_state_t
            

            # 5. Training step
            loss = agent.train_step(batch_size)

            if global_step % train_freq == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    ep_losses.append(loss)
                    writer.add_scalar("train/loss", loss, global_step)

            ep_reward += reward
            global_step += 1
            step_count += 1

            if step_count % target_update == 0:
                agent.update_target()

        # --- Training episode logs ---
        writer.add_scalar("train/episode_reward", ep_reward, ep)
        if ep_losses:
            writer.add_scalar(
                "train/episode_loss_mean",
                np.mean(ep_losses),
                ep
            )

        # --- Evaluation ---
        if ep % eval_every == 0:
            eval_stats = evaluate_agent(
                eval_env, agent, eval_episodes
            )

            writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], ep)
            writer.add_scalar("eval/std_reward", eval_stats["std_reward"], ep)
            writer.add_scalar("eval/mean_ep_length", eval_stats["mean_length"], ep)

            print(
                f"[Eval]\t Ep {ep:4d} | "
                f"Reward: {eval_stats['mean_reward']:.2f} ± "
                f"{eval_stats['std_reward']:.2f}"
                f" | Length: {eval_stats['mean_length']:.0f}"
            )

            if eval_stats['mean_reward'] > best_reward: # Sauvegarde basée sur la reward
                best_reward = eval_stats['mean_reward']
                torch.save(agent.q_net.state_dict(), f"{save_path}_best.pth")
                print(f"New best model saved to {save_path}_best.pth")

        print(f"[Train] Ep {ep:4d} | Reward: {ep_reward:.2f}")

        # Save the agent's Q-network
        torch.save(agent.q_net.state_dict(), f"{save_path}_last.pth")
    
    writer.close()
    return agent.q_net