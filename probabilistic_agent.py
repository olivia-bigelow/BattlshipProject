"""
This is the probabilistic mapping agent

The agent stores the ships sunk, the guesses made, and the target points

The agent picks actions purely probabilistically 

update: updates all of the stores of data based on a response to a new action taken

This class only runs a single game when play is called. 


"""
import numpy as np
from visualization import plot_rewards, create_results_dir
from probMap import probabilityMap


class ProbAgent:
    """
    Hunt and target agent 
    """
    
    def __init__(self, action_dim):
        """
        Initialize a Hunt and Target agent.
        
        Args:
            action_dim: Dimension of the action space
        """
        self.action_dim = action_dim
        
        self.prob = probabilityMap()



    def update (self, state, action, reward):
        '''
        this method updates the targets and sunk based on the current state
        '''

        print(state)

        #if the reward is 100, the game has won, no need to update
        if reward == 100:
          return
        
        self.prob.updateMap(action,reward)





    def play_game(self, env):

      '''
      this method plays a game of battleship, and returns the number of actions taken
      '''
      state = env.reset()
      num_actions = 0
      done = False

      while not done:
          action = self.prob.guessPoint()
          num_actions = num_actions+1
          state, reward, done, _ = env.step(action)
          self.update(state, action, reward)

      return num_actions


