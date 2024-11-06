import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tmrl import get_environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    The Actor network for the Soft Actor-Critic agent, producing actions based on observations.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

        # Initialize weights and biases
        self._initialize_weights()

    """Initialize weights using Xavier initialization and biases to zero."""

    def _initialize_weights(self):
        # Apply Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.log_std.weight)

        # Initialize biases to small values
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std).clamp(min=1e-6)
        return mu, std


class Critic(nn.Module):
    """
    The Critic network for the Soft Actor-Critic agent, estimating Q-values based on state and action pairs.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)


class LidarCNN(nn.Module):
    """
    A CNN to process LIDAR data as a feature extractor for SAC agent.

    Processes input shape (4, 64, 64) into a vector of size 64 (8 x 8).
    This is done to make the computations easier, as our computer wasn't powerful enough to handle the original input shape.
    """

    def __init__(self):
        super(LidarCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            4, 32, kernel_size=3, stride=2, padding=1
        )  # From (4, 64, 64) -> (32, 32, 32)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # From (32, 32, 32) -> (64, 16, 16)
        self.fc = nn.Linear(
            64 * 16 * 16, 64
        )  # Flatten the output and reduce to a vector of length 64

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU activation
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc(x))  # Fully connected layer to reduce to 64 features
        return x


class SACAgent:
    """
    The Soft Actor-Critic agent responsible for learning policies and value functions.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim, act_dim):
        # Define networks
        self.alpha = 0.10
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim).to(device)

        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    """Generate an action and its log probability given an observation."""

    def sample_action(self, obs):
        mu, std = self.actor(obs)

        if torch.isnan(mu).any() or torch.isnan(std).any():

            mu = torch.zeros_like(mu)
            std = torch.ones_like(std)

        std = std.clamp(min=1e-6)

        dist = torch.distributions.Normal(mu, std)

        x_t = dist.rsample()
        action = torch.tanh(x_t)

        log_prob = dist.log_prob(x_t) - torch.log(
            1 - action.pow(2) + 1e-6
        )  # Use log_prob for the tanh trick

        return action, log_prob.sum(1, keepdim=True)

    """Perform a SAC update on actor, critics, and entropy."""

    def update_parameters(self, replay_buffer, gamma=0.99, tau=0.005):
        obs, action, reward, next_obs, done = replay_buffer.sample()

        with torch.no_grad():
            next_action, log_prob = self.sample_action(next_obs)
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            q_target = reward + (1 - done) * gamma * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        new_action, log_prob = self.sample_action(obs)
        q1_new = self.critic1(obs, new_action)
        q2_new = self.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return reward


class ReplayBuffer:
    """
    A buffer to store and sample experience tuples for agent training.

    Args:
        buffer_size (int): Maximum size of the buffer.
        batch_size (int): Number of samples per batch.
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    """Add experience to the buffer."""

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    """Randomly sample a batch of experiences from the buffer."""

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.float32).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
        )

    """Returns the size of the replay buffer."""

    def size(self):
        return len(self.buffer)


"""Preprocess raw observation data for the agent's input."""


def preprocess_obs(obs, lidar_cnn):
    speed = obs[0].flatten()
    lidar = obs[1].flatten()
    prev_actions = np.concatenate([obs[2].flatten(), np.zeros(2)])

    lidar_image = torch.tensor(obs[3], dtype=torch.float32).unsqueeze(0).to(device)

    lidar_features = lidar_cnn(lidar_image).cpu().detach().numpy()

    processed_obs = np.concatenate(
        [speed, lidar, prev_actions, lidar_features.flatten()]
    )

    return processed_obs


"""Save agent and replay buffer states to a checkpoint file."""


def save_agent(agent, replay_buffer, filename="agents/sac_agent.pth"):
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic1_state_dict": agent.critic1.state_dict(),
            "critic2_state_dict": agent.critic2.state_dict(),
            "target_critic1_state_dict": agent.target_critic1.state_dict(),
            "target_critic2_state_dict": agent.target_critic2.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": agent.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": agent.critic2_optimizer.state_dict(),
            "log_alpha": agent.log_alpha,
            "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
            "replay_buffer": replay_buffer.buffer,
        },
        filename,
    )
    print("Agent saved to", filename)


"""Loads the SAC agent and its replay buffer if it exists."""


def load_agent(agent, replay_buffer, filename="agents/sac_agent.pth"):
    checkpoint = torch.load(filename)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic1.load_state_dict(checkpoint["critic1_state_dict"])
    agent.critic2.load_state_dict(checkpoint["critic2_state_dict"])
    agent.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
    agent.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])

    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
    agent.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
    agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

    agent.log_alpha = checkpoint["log_alpha"]
    replay_buffer.buffer = checkpoint["replay_buffer"]

    print("Agent loaded from", filename)


# Set up the environment
env = get_environment()

# Replay buffer and agent configuration
buffer_size = 1000000  # Size of the replay buffer
batch_size = 64  # Batch size for sampling

# Initialize the replay buffer
replay_buffer = ReplayBuffer(buffer_size, batch_size)

# Observation and action dimensions (adjusted to environment specifics)
obs_dim = (
    1 + 1 + 3 + (8 * 8)
)  # observation space using our CNN to transform our LIDAR images from (4, 64, 64) to (8 x 8)
act_dim = 3  # action space with [gas, brake, steer]

# Initialize the agent
agent = SACAgent(obs_dim, act_dim)

# Initialize the LIDAR CNN model for observation processing
lidar_cnn = LidarCNN().to(device)

# Define checkpoint file
checkpoint_file = "agents/sac_agent.pth"

# Try loading the agent if a checkpoint exists
try:
    load_agent(agent, replay_buffer, checkpoint_file)
except FileNotFoundError:
    print("No saved agent found, starting fresh.")

# Start training loop
obs, info = env.reset()  # Initial environment reset
for step in range(10000000):  # Total number of training steps
    time.sleep(0.01)  # Small delay to match environment timing if needed

    # Preprocess observation
    processed_obs = preprocess_obs(obs, lidar_cnn)

    # Convert processed observation to tensor for action sampling
    act, log_prob = agent.sample_action(
        torch.tensor(processed_obs, dtype=torch.float32).unsqueeze(0).to(device)
    )
    action = (
        act.cpu().detach().numpy().flatten()
    )  # Convert action to numpy for environment

    # Take a step in the environment
    obs_next, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # Check if episode is done

    # Store experience in replay buffer
    replay_buffer.store(
        processed_obs, action, reward, preprocess_obs(obs_next, lidar_cnn), done
    )

    # Start training if enough samples are available
    if replay_buffer.size() >= batch_size:
        agent.update_parameters(replay_buffer)

    # Save agent periodically
    if step % 10000 == 0:  # Save every 10,000 steps
        save_agent(agent, replay_buffer, checkpoint_file)

    # Reset environment if episode is done
    if done:
        obs, info = env.reset()
    else:
        obs = obs_next  # Update observation for next step
