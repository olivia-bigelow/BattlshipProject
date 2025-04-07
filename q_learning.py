import numpy as np

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

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Implement Q-learning algorithm for Battleship environment.
    
    Args:
        env: Gym environment
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        
    Returns:
        q_table: Learned Q-table
    """
    # Initialize Q-table
    q_table = {}
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Convert state to hashable tuple
            state_key = state_to_tuple(state)
            
            # Choose action using epsilon-greedy policy
            if state_key not in q_table:
                q_table[state_key] = [0] * env.action_space.n
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = int(np.argmax(q_table[state_key]))  # Exploit, convert to int
            
            # Use the action directly without formatting
            next_state, reward, done, _ = env.step(action)
            
            # Convert next_state to hashable tuple
            next_state_key = state_to_tuple(next_state)
            
            # Initialize next state in Q-table if not present
            if next_state_key not in q_table:
                q_table[next_state_key] = [0] * env.action_space.n
            
            # Update Q-value
            best_next_action = np.argmax(q_table[next_state_key])
            td_target = reward + gamma * q_table[next_state_key][best_next_action]
            td_error = td_target - q_table[state_key][action]
            q_table[state_key][action] += alpha * td_error
            
            state = next_state
    
    return q_table

def test_agent(env, q_table, num_episodes=100):
    """
    Test a trained Q-learning agent.
    
    Args:
        env: Gym environment
        q_table: Learned Q-table
        num_episodes: Number of episodes to test
        
    Returns:
        average_reward: Average reward over test episodes
    """
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Convert state to hashable tuple
            state_key = state_to_tuple(state)
            
            if state_key not in q_table:
                action = env.action_space.sample()  # Fallback to random action if state not in Q-table
            else:
                action = int(np.argmax(q_table[state_key]))  # Exploit, convert to int
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    
    average_reward = total_reward / num_episodes
    return average_reward
