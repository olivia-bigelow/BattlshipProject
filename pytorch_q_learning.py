import torch
import numpy as np
import random
import time
from visualization import plot_rewards, create_results_dir

def state_to_tuple(state):
    """Convert numpy array state to hashable tuple recursively."""
    if isinstance(state, np.ndarray):
        # Convert to bytes for complex arrays (more reliable)
        return state.tobytes()
    elif isinstance(state, (list, tuple)):
        return tuple(state_to_tuple(s) for s in state)
    elif isinstance(state, dict):
        return tuple(sorted((k, state_to_tuple(v)) for k, v in state.items()))
    else:
        return state

class PyTorchQlearning:
    """
    Q-learning agent implemented using PyTorch for the Battleship environment.
    
    This is a tabular Q-learning implementation (not deep) that uses PyTorch tensors
    for efficient computation.
    """
    
    def __init__(self, action_dim, alpha=0.1, gamma=0.99, epsilon=1.0, device=None):
        """
        Initialize a Q-learning agent.
        
        Args:
            action_dim: Dimension of the action space
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            device: PyTorch device (cpu or cuda)
        """
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Dictionary to store state-action values
        self.q_table = {}
        
        # Set device (CPU or GPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Add rewards_per_episode attribute to track rewards
        self.rewards_per_episode = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether we're in training mode (exploration allowed)
            
        Returns:
            action: Selected action
        """
        # Convert state to hashable tuple
        state_key = state_to_tuple(state)
        
        # If state not in Q-table, initialize it
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(self.action_dim, device=self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action
            return torch.argmax(self.q_table[state_key]).item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert states to hashable tuples
        state_key = state_to_tuple(state)
        next_state_key = state_to_tuple(next_state)
        
        # Initialize Q-values if states are not in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = torch.zeros(self.action_dim, device=self.device)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = torch.zeros(self.action_dim, device=self.device)
        
        # Q-learning update
        if not done:
            # Calculate target Q-value: reward + gamma * max(Q(s', a'))
            target = reward + self.gamma * torch.max(self.q_table[next_state_key])
        else:
            # If episode is done, target is just the reward
            target = reward
        
        # Calculate temporal difference error and update Q-value
        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_table_size(self):
        """Return the size of the Q-table (number of states)"""
        return len(self.q_table)
    
    def save_rewards_plot(self, title="Q-Learning Rewards", filename="q_learning_rewards.png"):
        """Save a plot of rewards over time"""
        results_dir = create_results_dir()
        plot_rewards(
            self.rewards_per_episode, 
            title=title,
            filename=f"{results_dir}/{filename}"
        )


def train_q_learning(env, agent, num_episodes=1000):
    """
    Train a Q-learning agent on the Battleship environment.
    
    Args:
        env: Gym environment
        agent: Q-learning agent
        num_episodes: Number of episodes to train
        
    Returns:
        agent: Trained agent
    """
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
            
            # Update Q-values
            agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        # Update epsilon
        agent.update_epsilon()
        
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
            print(f"Q-table size: {agent.get_table_size()} states")
            print("-" * 50)
    
    # Save rewards plot
    agent.save_rewards_plot()
    
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Total steps: {total_steps}")
    return agent


def test_q_learning_agent(env, agent, num_episodes=100):
    """
    Test a trained Q-learning agent.
    
    Args:
        env: Gym environment
        agent: Q-learning agent
        num_episodes: Number of episodes to test
        
    Returns:
        average_reward: Average reward over test episodes
    """
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