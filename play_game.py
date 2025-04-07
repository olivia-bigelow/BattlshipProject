import time
import numpy as np
import gym
import gym_battleship
import torch
import random

def play_games(agent, env, num_games=5, delay=0.5, render=True, max_steps=100):
    """
    Play games with a trained agent and visualize the process.
    
    Args:
        agent: Trained agent (Q-learning or DQN)
        env: Battleship environment
        num_games: Number of games to play
        delay: Delay between moves in seconds (for visualization)
        render: Whether to render the game state
        max_steps: Maximum steps per game to prevent infinite loops
    
    Returns:
        avg_reward: Average reward over games
        avg_steps: Average number of steps to complete a game
    """
    total_reward = 0
    total_steps = 0
    wins = 0
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_reward = 0
        steps = 0
        
        # Keep track of attempted actions to avoid loops
        attempted_actions = set()
        consecutive_repeats = 0
        
        print(f"\n===== Game {game+1}/{num_games} =====")
        if render:
            print("Actual ship positions:")
            env.render_board_generated()
            print("\nInitial board state:")
            env.render()
        
        while not done and steps < max_steps:
            # Get action from agent
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=False)
            else:
                # For traditional Q-learning
                state_key = agent.state_to_tuple(state) if hasattr(agent, 'state_to_tuple') else None
                if state_key and state_key in agent.q_table:
                    action = int(np.argmax(agent.q_table[state_key]))
                else:
                    action = env.action_space.sample()
            
            # If action was already attempted or we're seeing too many repeats, choose random action
            if action in attempted_actions:
                consecutive_repeats += 1
                if consecutive_repeats > 3:  # Allow a few repeats before intervening
                    # Pick random action that hasn't been tried
                    available_actions = [a for a in range(env.action_space.n) if a not in attempted_actions]
                    
                    # If we've tried all actions or nearly all, reset and try again with some randomness
                    if len(available_actions) < 5:
                        print("Too many repeated actions, introducing randomness...")
                        attempted_actions = set()  # Reset attempted actions
                        action = env.action_space.sample()
                    else:
                        action = random.choice(available_actions)
                    
                    consecutive_repeats = 0
                    print(f"Agent was stuck in a loop. Choosing random action instead.")
            else:
                consecutive_repeats = 0
                attempted_actions.add(action)
            
            # Convert action to coordinates for display
            x, y = action % env.board_size[0], action // env.board_size[0]
            print(f"Agent fires at: ({x}, {y}) [Action {action}]")
            
            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            game_reward += reward
            steps += 1
            
            # Print result
            if reward > 0 and reward < 100:  # Ship hit but not game won
                print(f"HIT! Reward: {reward}")
            elif reward == 0:  # Miss
                print(f"Miss! Reward: {reward}")
            elif reward < 0:  # Penalty (repeated action)
                print(f"Repeated action! Penalty: {reward}")
            elif reward >= 100:  # Game won
                print(f"Game Won! Reward: {reward}")
                wins += 1
            
            # Update state
            state = next_state
            
            # Render board if requested
            if render:
                env.render()
                time.sleep(delay)  # Add delay for better visualization
        
        # Check if we hit the max steps limit
        if steps >= max_steps:
            print(f"Game stopped after reaching maximum steps ({max_steps})")
        
        total_reward += game_reward
        total_steps += steps
        
        print(f"Game {game+1} complete! Reward: {game_reward}, Steps: {steps}")
    
    avg_reward = total_reward / num_games
    avg_steps = total_steps / num_games
    win_rate = (wins / num_games) * 100
    
    print(f"\n===== Results over {num_games} games =====")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Win rate: {win_rate:.1f}%")
    
    return avg_reward, avg_steps

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