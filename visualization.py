import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rewards(rewards, window_size=100, title='Average Rewards', filename='rewards_plot.png'):
    """
    Plot rewards over time with a moving average.
    
    Args:
        rewards: List of rewards for each episode
        window_size: Size of the moving average window
        title: Plot title
        filename: Output filename for the saved plot
    """
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw rewards')
    
    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='blue', 
                 label=f'Moving average (window={window_size})')
    
    # Customize plot
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    
    # Close the figure to free memory
    plt.close()

def plot_rewards_comparison(rewards_dict, window_size=100, title='Rewards Comparison', 
                           filename='rewards_comparison.png'):
    """
    Plot rewards comparison between different agents.
    
    Args:
        rewards_dict: Dictionary mapping agent names to lists of rewards
        window_size: Size of the moving average window
        title: Plot title
        filename: Output filename for the saved plot
    """
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    color_idx = 0
    
    # Adjust window size if needed to ensure we can plot data
    max_possible_window = min([len(rewards) for rewards in rewards_dict.values()])
    if max_possible_window < window_size:
        window_size = max(1, max_possible_window // 2)
        print(f"Adjusted window size to {window_size} based on available data")
    
    # Plot moving averages for each agent
    for agent_name, rewards in rewards_dict.items():
        # Plot raw rewards with low alpha
        plt.plot(rewards, alpha=0.1, color=colors[color_idx % len(colors)], 
                 label=f'{agent_name} Raw')
        
        # Plot moving average if possible
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                     color=colors[color_idx % len(colors)], 
                     label=f'{agent_name} (window={window_size})')
        else:
            print(f"Warning: Not enough data to calculate moving average for {agent_name}")
        
        color_idx += 1
    
    # Customize plot
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    
    # Close the figure to free memory
    plt.close()

def create_results_dir():
    """Create a results directory if it doesn't exist"""
    os.makedirs('results', exist_ok=True)
    return 'results'