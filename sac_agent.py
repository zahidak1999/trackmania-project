import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tmrl import get_environment

from torch import amp

import warnings

DEBUG = False

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    """
    The Actor network for PPO
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.xavier_uniform_(self.log_std.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std).clamp(min=1e-6)
        return mean, std


class Critic(nn.Module):
    """
    The Critic network for PPO, estimating state values.
    """

    def __init__(self, obs_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.value.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class LidarCNN(nn.Module):
    """
    A CNN to process LIDAR data as a feature extractor for PPO agent.
    """

    def __init__(self):
        super(LidarCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)  # (4, 64, 64) -> (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (32, 32, 32) -> (64, 16, 16)
        self.fc = nn.Linear(64 * 16 * 16, 64)  # Flatten and reduce to 64 features

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (32, 32, 32)
        x = F.relu(self.conv2(x))  # (64, 16, 16)
        x = x.view(x.size(0), -1)  # Flatten to (64*16*16) = 16384
        x = F.relu(self.fc(x))      # (64,)
        return x


class TrajectoryBuffer:
    """
    A buffer to store trajectories for PPO agent training.
    """

    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam

        # Storage
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

    def store(self, obs, action, log_prob, reward, done, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_advantages(self, last_value=0, last_done=0):
        """
        Compute GAE advantages.
        """
        advantages = []
        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * last_value * (1 - self.dones[step]) - self.values[step]
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            last_value = self.values[step]
            last_done = self.dones[step]

        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32).to(device)

    def get_batches(self, batch_size):
        """
        Yield mini-batches for training.
        """
        data_size = len(self.observations)
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        for start in range(0, data_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                torch.stack([self.observations[i] for i in batch_indices]).to(device),
                torch.stack([self.actions[i] for i in batch_indices]).to(device),
                torch.stack([self.log_probs[i] for i in batch_indices]).to(device),
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(
            self,
            obs_dim,
            act_dim,
            hidden_size=256,
            lr=1e-4,  # Reduced learning rate
            gamma=0.99,
            lam=0.95,
            clip_epsilon=0.2,
            K_epochs=4,
            batch_size=64,
            entropy_coef=0.01,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        # Networks
        self.actor = Actor(obs_dim, act_dim, hidden_size).to(device)
        self.critic = Critic(obs_dim, hidden_size).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # Replay buffer
        self.buffer = TrajectoryBuffer(obs_dim, act_dim, buffer_size=100000, gamma=gamma, lam=lam)

        # Mixed precision
        self.use_amp = device.type == 'cuda'
        if self.use_amp:
            self.scaler = amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        else:
            self.scaler = None

    def select_action(self, obs):
        """
        Select an action based on the current observation.
        """
        with torch.no_grad():
            mean, std = self.actor(obs)
            # Check for NaNs
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("Warning: Actor network output contains NaN values!")
                # Handle accordingly, e.g., skip this step or reset environment
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.tanh(action)
            log_prob = dist.log_prob(action) - torch.log(1 - action_clipped.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            value = self.critic(obs)

        return action_clipped, log_prob, value

    def update(self):
        """
        Update the actor and critic networks using PPO.
        """
        # Compute advantages and returns
        self.buffer.compute_advantages()
        advantages = self.buffer.advantages
        returns = self.buffer.returns

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Initialize accumulators for loss
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for obs, actions, old_log_probs, adv, ret in self.buffer.get_batches(self.batch_size):
                with amp.autocast(device_type='cuda', enabled=self.use_amp):
                    mean, std = self.actor(obs)
                    dist = torch.distributions.Normal(mean, std)
                    entropy = dist.entropy().sum(dim=1, keepdim=True)
                    new_log_probs = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6)
                    new_log_probs = new_log_probs.sum(1, keepdim=True)

                    # Ratios for PPO clipping
                    ratios = torch.exp(new_log_probs - old_log_probs)

                    # Surrogate objective
                    surr1 = ratios * adv
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss (MSE) - Squeeze the values tensor
                    values = self.critic(obs)
                    critic_loss = F.mse_loss(values.squeeze(-1), ret)

                    # Entropy loss
                    entropy_loss = -self.entropy_coef * entropy.mean()

                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Accumulate losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()

                # Backpropagation with mixed precision
                self.optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                    self.optimizer.step()

        # Compute average losses
        total_batches = self.K_epochs * (len(self.buffer.observations) // self.batch_size)
        if total_batches > 0:
            avg_actor_loss = total_actor_loss / total_batches
            avg_critic_loss = total_critic_loss / total_batches
            avg_entropy_loss = total_entropy_loss / total_batches
        else:
            avg_actor_loss = avg_critic_loss = avg_entropy_loss = 0.0

        print(
            f"Update completed. Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, Entropy Loss: {avg_entropy_loss:.4f}")

        # Clear buffer after update
        self.buffer.reset()


def preprocess_obs(obs, lidar_cnn):
    """
    Preprocess the observation by converting it to tensors and extracting features using LidarCNN.
    """
    speed = torch.tensor(obs[0], dtype=torch.float32).to(device)  # Shape: (1,)
    if DEBUG:
        print(f"Speed shape: {speed.shape}")

    lidar = torch.tensor(obs[1], dtype=torch.float32).to(device)  # Shape: (1,)
    if DEBUG:
        print(f"Lidar shape: {lidar.shape}")

    # Adjust padding based on the actual size of obs[2]
    num_existing_actions = len(obs[2].flatten())
    desired_prev_actions = 5
    padding_size = max(desired_prev_actions - num_existing_actions, 0)
    prev_actions_np = np.concatenate([obs[2].flatten(), np.zeros(padding_size)])
    if DEBUG:
        print(f"Previous actions (NumPy) shape: {prev_actions_np.shape}")

    prev_actions = torch.tensor(
        prev_actions_np, dtype=torch.float32
    ).to(device)
    if DEBUG:
        print(f"Previous actions (Tensor) shape: {prev_actions.shape}")

    lidar_image = torch.tensor(obs[3], dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 4, 64, 64)
    lidar_image = lidar_image / 255.0
    if DEBUG:
        print(f"Lidar image shape: {lidar_image.shape}")

    with torch.no_grad():
        lidar_features = lidar_cnn(lidar_image)  # Shape: (1, 64)
    if DEBUG:
        print(f"Lidar features shape: {lidar_features.shape}")

    processed_obs = torch.cat([speed, lidar, prev_actions, lidar_features.flatten()], dim=0)  # Shape: (71,)
    if DEBUG:
        print(f"Processed observation shape: {processed_obs.shape}")

    # Check for NaNs
    if torch.isnan(processed_obs).any() or torch.isinf(processed_obs).any():
        print("Warning: Processed observation contains NaN or Inf values!")

    return processed_obs


def save_agent(agent, filename="agents/ppo_agent.pth"):
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "buffer_observations": agent.buffer.observations,
            "buffer_actions": agent.buffer.actions,
            "buffer_log_probs": agent.buffer.log_probs,
            "buffer_rewards": agent.buffer.rewards,
            "buffer_dones": agent.buffer.dones,
            "buffer_values": agent.buffer.values,
        },
        filename,
    )
    print(f"Agent saved to {filename}")


def load_agent(agent, filename="agents/ppo_agent.pth"):
    checkpoint = torch.load(filename, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load buffer data
    agent.buffer.observations = checkpoint["buffer_observations"]
    agent.buffer.actions = checkpoint["buffer_actions"]
    agent.buffer.log_probs = checkpoint["buffer_log_probs"]
    agent.buffer.rewards = checkpoint["buffer_rewards"]
    agent.buffer.dones = checkpoint["buffer_dones"]
    agent.buffer.values = checkpoint["buffer_values"]

    # Advantages and returns are computed during update, no need to load them
    agent.buffer.advantages = []
    agent.buffer.returns = []

    print(f"Agent loaded from {filename}")


# Initialize the environment
env = get_environment()

# Define observation and action dimensions
# Adjust based on the actual observation structure
# Example: speed (1) + lidar (1) + prev_actions (5) + lidar_features (64) = 71
obs_dim = 71  # Ensure this matches the processed_obs
act_dim = 3  # [gas, brake, steer]

# Initialize LidarCNN
lidar_cnn = LidarCNN().to(device)
lidar_cnn.eval()  # Set to evaluation mode since it's a feature extractor

# Initialize PPO Agent
agent = PPOAgent(
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_size=256,
    lr=1e-4,  # Reduced learning rate
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    K_epochs=4,
    batch_size=64,
    entropy_coef=0.01,
)

# Define checkpoint file
checkpoint_file = "agents/ppo_agent.pth"

# Try loading the agent if a checkpoint exists
try:
    load_agent(agent, checkpoint_file)
except FileNotFoundError:
    print("No saved agent found, starting fresh.")

# Initialize environment and get initial observation
try:
    print("Resetting environment...")
    obs, info = env.reset()  # Initial environment reset
    print("Environment reset successfully.")
except Exception as e:
    print(f"Environment reset failed: {e}")
    exit()

# Training parameters
max_steps = 1000000  # Total number of training steps
update_interval = 2048  # Steps to collect before updating
log_interval = 1000  # Steps between logging

# Initialize variables
step = 0
episode_rewards = []
current_episode_reward = 0

start_time = time.time()

try:
    while step < max_steps:
        for _ in range(update_interval):
            # Preprocess observation
            processed_obs = preprocess_obs(obs, lidar_cnn)  # Shape: (71,)

            # Add batch dimension
            processed_obs = processed_obs.unsqueeze(0)  # Shape: (1, 71)

            # Select action
            action, log_prob, value = agent.select_action(processed_obs)

            # Convert action to numpy for environment
            action_np = action.cpu().detach().numpy().flatten()

            # Take a step in the environment
            try:
                obs_next, reward, terminated, truncated, info = env.step(action_np)
            except Exception as e:
                print(f"Environment step failed: {e}")
                break  # Exit the loop if environment step fails

            done = terminated or truncated

            # Debug: Print reward
            print(f"Step: {step}, Reward: {reward}, Done: {done}")

            # Preprocess next observation
            processed_obs_next = preprocess_obs(obs_next, lidar_cnn)  # Shape: (71,)

            # Store transition in buffer
            agent.buffer.store(
                processed_obs.squeeze(0),  # Shape: (71,)
                action.squeeze(0),          # Shape: (3,)
                log_prob.squeeze(0),        # Shape: (1,)
                reward,                     # float
                done,                       # bool
                value.squeeze(0),           # Shape: (1,)
            )

            current_episode_reward += reward
            step += 1

            # Reset environment if done
            if done:
                episode_rewards.append(current_episode_reward)
                print(f"Episode {len(episode_rewards)} finished with reward: {current_episode_reward}")
                current_episode_reward = 0
                try:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    print("Environment reset successfully.")
                except Exception as e:
                    print(f"Environment reset failed: {e}")
                    break  # Exit the loop if environment reset fails
            else:
                obs = obs_next

            # Check if maximum steps reached
            if step >= max_steps:
                break

        # Update agent
        agent.update()

        # Logging
        if step % log_interval == 0:
            elapsed_time = time.time() - start_time
            average_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            print(
                f"Step: {step}, Average Reward (last 100 episodes): {average_reward:.2f}, Elapsed Time: {elapsed_time:.2f}s")
            start_time = time.time()

        # Save agent periodically
        if step % (update_interval * 10) == 0:
            save_agent(agent, checkpoint_file)

except KeyboardInterrupt:
    print("Training interrupted by user.")
    save_agent(agent, checkpoint_file)
    exit()
except Exception as e:
    print(f"An error occurred during training: {e}")
    save_agent(agent, checkpoint_file)
    exit()
