import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from visualization import plot_rewards, create_results_dir


class ReplayBuffer:
    """Experience replay buffer to store and sample experiences"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for Battleship game"""
    
    def __init__(self, input_channels, board_size, action_dim):
        super(DQNNetwork, self).__init__()
        
        # CNN layers for processing the 2D grid state
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = board_size[0] * board_size[1] * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DQNAgent:
    """DQN agent that learns to play Battleship"""
    
    def __init__(self, state_shape, action_dim, board_size, device=None):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.board_size = board_size
        
        # Set device (CPU or GPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create networks
        self.policy_net = DQNNetwork(state_shape[0], board_size, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_shape[0], board_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.target_update = 10  # Update target network every N episodes
        
        # Tracking
        self.steps_done = 0
        self.episode_rewards = []
        
        # Add rewards tracking
        self.rewards_per_episode = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update_model(self):
        """Update the model by sampling from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_rewards_plot(self, title="DQN Rewards", filename="dqn_rewards.png"):
        """Save a plot of rewards over time"""
        results_dir = create_results_dir()
        plot_rewards(
            self.rewards_per_episode,
            title=title,
            filename=f"{results_dir}/{filename}"
        )


def train_dqn(env, agent, num_episodes=1000):
    """Train the DQN agent on the Battleship environment"""
    all_rewards = []
    total_training_time = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update model
            agent.update_model()
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Record rewards
        all_rewards.append(episode_reward)
        agent.rewards_per_episode.append(episode_reward)
        
        # Calculate episode time
        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time
        total_training_time += episode_time
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(all_rewards[-100:]) / 100
            avg_time_per_episode = total_training_time / (episode + 1)
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            print(f"Episode time: {episode_time:.2f}s, Avg time per episode: {avg_time_per_episode:.2f}s")
            print(f"Episode steps: {episode_steps}, Total steps: {total_steps}")
            print(f"Estimated remaining time: {avg_time_per_episode * (num_episodes - episode - 1):.2f}s")
            print("-" * 50)
    
    # Save rewards plot
    agent.save_rewards_plot()
    
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Total steps: {total_steps}")
    return agent


def test_dqn_agent(env, agent, num_episodes=100):
    """Test the trained DQN agent"""
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Select action without exploration
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    
    average_reward = total_reward / num_episodes
    return average_reward