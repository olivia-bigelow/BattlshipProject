import time
import numpy as np
import gym
import gym_battleship
import torch
import random
from visualization import create_results_dir, plot_moves_per_game

def select_valid_action(agent, state, taken_actions):
    """
    Select an action that hasn't been taken before in the current game.
    
    Args:
        agent: Trained agent (Q-learning or DQN)
        state: Current game state
        taken_actions: Set of actions already taken in the current game
        
    Returns:
        action: A valid action that hasn't been taken before
    """
    # Get the action recommended by the agent
    action = agent.select_action(state, training=False)
    
    # If the action has already been taken, find a new valid action
    attempts = 0
    max_attempts = 1000  # Safety limit to prevent infinite loops
    
    while action in taken_actions and attempts < max_attempts:
        # Try to find a valid action by sampling randomly
        if hasattr(agent, 'action_dim'):
            # For Q-learning agents with known action dimension
            action = random.randint(0, agent.action_dim - 1)
        else:
            # For other agents, try to get the action space from the environment
            # This is a fallback that assumes the action space is bounded
            action = random.randint(0, 99)  # For a 10x10 Battleship board
        attempts += 1
    
    if attempts >= max_attempts:
        print("Warning: Could not find an untried action after many attempts.")
    
    return action

def play_games(agent, env, num_games=5, delay=0.5, render=True, max_steps=100, visualize_moves=True):
    """
    Play games with a trained agent and visualize the process.
    
    Args:
        agent: Trained agent (Q-learning or DQN)
        env: Battleship environment
        num_games: Number of games to play
        delay: Delay between moves in seconds (for visualization)
        render: Whether to render the game state
        max_steps: Maximum steps per game to prevent infinite loops
        visualize_moves: Whether to generate a plot of moves per game
    
    Returns:
        avg_reward: Average reward over games
        avg_steps: Average number of steps to complete a game
        moves_per_game: List containing the number of moves for each game
    """
    moves_per_game = []
    total_reward = 0
    total_steps = 0
    
    for game in range(num_games):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        # Track moves already taken in this game
        taken_actions = set()
        
        while not done and episode_steps < max_steps:
            if render:
                env.render()
                time.sleep(delay)
            
            # Get an action that hasn't been taken yet
            action = select_valid_action(agent, state, taken_actions)
            print(f"Action taken: {action}")
            
            # Record this action
            taken_actions.add(action)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
        moves_per_game.append(episode_steps)
        total_reward += episode_reward
        total_steps += episode_steps
    
    avg_reward = total_reward / num_games
    avg_steps = total_steps / num_games
    
    # Generate plot of moves per game after all games are completed
    if visualize_moves and moves_per_game:
        results_dir = create_results_dir()
        plot_moves_per_game(moves_per_game, title=f'Moves per Game', 
                          filename=f'{results_dir}/moves_per_game.png')
    
    print(f"Average reward: {avg_reward}, Average steps: {avg_steps}")
    
    return avg_reward, avg_steps, moves_per_game

def load_dqn_model(model_path, env, device):
    """
    Load a trained DQN model from a file.
    
    Args:
        model_path: Path to the saved model
        env: Battleship environment
        device: Device to load model on (cpu/cuda)
        
    Returns:
        agent: Loaded DQN agent
    """
    from dqn_agent import DQNAgent, DQNNetwork
    
    # Create agent with same parameters
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        board_size=env.board_size,
        device=device
    )
    
    # Load saved weights
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()  # Set to evaluation mode
    
    return agent