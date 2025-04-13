import gym
import gym_battleship
import numpy as np
import torch
import argparse
import os
from dqn_agent import DQNAgent, train_dqn, test_dqn_agent
from pytorch_q_learning import PyTorchQlearning, train_q_learning, test_q_learning_agent
from hunt_and_target import HATAgent
from visualization import plot_rewards_comparison, create_results_dir
from play_game import play_games, load_dqn_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and test agents for Battleship environment')
    parser.add_argument('--agent', type=str, default='hunt_and_target', choices=['q_learning', 'dqn', 'both', "hunt_and_target"],
                        help='Agent type to use (q_learning, dqn, or both, or hunt and target)')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plot between both agents')
    parser.add_argument('--play', action='store_true',
                        help='Play games with the trained agent(s) after training')
    parser.add_argument('--play_games', type=int, default=5,
                        help='Number of games to play after training')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between moves in seconds (for visualization)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a saved model to load instead of training (DQN only)')
    return parser.parse_args()


def train_and_test_HAT(env, device, num_episodes, test_episodes):
    print(f"Training hunt and target agent for {num_episodes} episodes...")
    agent = HATAgent(action_dim=env.action_space.n)

    numActions = agent.play_game(env)

    print(numActions)

    print("END OF THIS TEST")
    return 0,0,0

def train_and_test_q_learning(env, device, num_episodes, test_episodes):
    print(f"Training Q-learning agent for {num_episodes} episodes...")
    agent = PyTorchQlearning(
        action_dim=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        device=device
    )
    
    # Train the agent
    trained_agent = train_q_learning(env, agent, num_episodes=num_episodes)
    
    # Test the agent
    total_reward = 0
    total_steps = 0
    
    for _ in range(test_episodes):
        state = env.reset()
        done = False
        episode_steps = 0
        
        while not done:
            action = trained_agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_steps += 1
            state = next_state
        
        total_steps += episode_steps
    
    avg_reward = total_reward / test_episodes
    avg_reward_per_step = total_reward / total_steps if total_steps > 0 else 0
    
    print(f"Q-Learning - Average reward: {avg_reward:.2f}")
    print(f"Q-Learning - Average reward per step: {avg_reward_per_step:.4f}")
    print(f"Q-Learning - Final Q-table size: {trained_agent.get_table_size()} states")
    
    return trained_agent, avg_reward, avg_reward_per_step

def train_and_test_dqn(env, device, num_episodes, test_episodes):
    print(f"Training DQN agent for {num_episodes} episodes...")
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        board_size=env.board_size,
        device=device
    )
    
    # Train the agent
    trained_agent = train_dqn(env, agent, num_episodes=num_episodes)
    
    # Test the agent
    total_reward = 0
    total_steps = 0
    
    for _ in range(test_episodes):
        state = env.reset()
        done = False
        episode_steps = 0
        
        while not done:
            action = trained_agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_steps += 1
            state = next_state
        
        total_steps += episode_steps
    
    avg_reward = total_reward / test_episodes
    avg_reward_per_step = total_reward / total_steps if total_steps > 0 else 0
    
    print(f"DQN - Average reward: {avg_reward:.2f}")
    print(f"DQN - Average reward per step: {avg_reward_per_step:.4f}")
    
    # Save the trained model
    torch.save(trained_agent.policy_net.state_dict(), "results/battleship_dqn_model.pth")
    print("DQN model saved to results/battleship_dqn_model.pth")
    
    return trained_agent, avg_reward, avg_reward_per_step

def generate_comparison_plots(q_agent, dqn_agent):
    """Generate comparison plots between Q-learning and DQN agents"""
    rewards_dict = {
        "Q-Learning": q_agent.rewards_per_episode,
        "DQN": dqn_agent.rewards_per_episode
    }
    
    results_dir = create_results_dir()
    plot_rewards_comparison(
        rewards_dict=rewards_dict,
        title="Q-Learning vs DQN Rewards",
        filename=f"{results_dir}/comparison_rewards.png"
    )
    print(f"Comparison plot saved to {results_dir}/comparison_rewards.png")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Automatically enable comparison when both agents are selected
    if args.agent == 'both':
        args.compare = True
        print("Automatically enabling comparison mode for 'both' agents")
    
    # Create the environment
    env = gym.make("Battleship-v0", episode_steps=100, reward_dictionary = {
    'win': 100,
    'missed': -1,
    'touched': 1,
    'repeat_missed': -1,
    'repeat_touched': -0.5
})
    env.reset()
    
    # Get environment dimensions
    state_shape = env.observation_space.shape
    board_size = env.board_size
    action_dim = env.action_space.n
    
    print(f"State shape: {state_shape}")
    print(f"Board size: {board_size}")
    print(f"Action dimension: {action_dim}")
    
    # Create results directory
    results_dir = create_results_dir()
    
    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    q_agent = None
    dqn_agent = None
    
    # Load model or train new agents
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        dqn_agent = load_dqn_model(args.load_model, env, device)
        print("Model loaded successfully")
        args.agent = 'dqn'  # Override agent type to dqn
    else:
        # Train and test the selected agent(s)
        if args.agent in ['q_learning', 'both']:
            q_agent, q_reward, q_reward_per_step = train_and_test_q_learning(
                env, device, args.episodes, args.test_episodes
            )
        
        if args.agent in ['dqn', 'both']:
            dqn_agent, dqn_reward, dqn_reward_per_step = train_and_test_dqn(
                env, device, args.episodes, args.test_episodes
            )
        if args.agent in ["hunt_and_target"]:
            dqn_agent, dqn_reward, dqn_reward_per_step = train_and_test_HAT(
                env, device, args.episodes, args.test_episodes
            )



        # Generate comparison if requested
        if args.compare and args.agent == 'both' and q_agent and dqn_agent:
            try:
                generate_comparison_plots(q_agent, dqn_agent)
                
                # Print comparative summary
                print("\n===== COMPARATIVE SUMMARY =====")
                print(f"Q-Learning - Avg reward: {q_reward:.2f}, Avg reward per step: {q_reward_per_step:.4f}")
                print(f"DQN - Avg reward: {dqn_reward:.2f}, Avg reward per step: {dqn_reward_per_step:.4f}")
                
                # Calculate improvement percentages
                if q_reward != 0:
                    reward_improvement = ((dqn_reward - q_reward) / abs(q_reward)) * 100
                    print(f"DQN reward improvement: {reward_improvement:.1f}%")
                
                if q_reward_per_step != 0:
                    step_improvement = ((dqn_reward_per_step - q_reward_per_step) / abs(q_reward_per_step)) * 100
                    print(f"DQN reward per step improvement: {step_improvement:.1f}%")
            except Exception as e:
                print(f"Error generating comparison: {e}")
                print("Try running with more episodes to collect enough data for comparison")
    
    # Play games if requested
    if args.play:
        if args.agent == 'q_learning' and q_agent:
            print("\n===== Playing games with Q-Learning agent =====")
            play_games(q_agent, env, num_games=args.play_games, delay=args.delay)
        
        elif args.agent == 'dqn' and dqn_agent:
            print("\n===== Playing games with DQN agent =====")
            play_games(dqn_agent, env, num_games=args.play_games, delay=args.delay)
        
        elif args.agent == 'both' and q_agent and dqn_agent:
            print("\n===== Playing games with Q-Learning agent =====")
            q_reward, q_steps = play_games(q_agent, env, num_games=args.play_games, delay=args.delay)
            
            print("\n===== Playing games with DQN agent =====")
            dqn_reward, dqn_steps = play_games(dqn_agent, env, num_games=args.play_games, delay=args.delay)
            
            print("\n===== Play comparison =====")
            print(f"Q-Learning - Avg reward: {q_reward:.2f}, Avg steps: {q_steps:.2f}")
            print(f"DQN - Avg reward: {dqn_reward:.2f}, Avg steps: {dqn_steps:.2f}")
        else:
            print("No trained agent available to play games with.")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()