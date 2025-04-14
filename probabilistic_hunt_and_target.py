"""
This is the hunt and target agent with probabilistic mapping


The agent stores the ships sunk, the guesses made, and the target points

The agent picks actions probabilistically while hunting, 
and then guesses probabilistically around a target point (hit) until fully explored


update: updates all of the stores of data based on a response to a new action taken

This class only runs a single game when play is called. 







"""
import numpy as np
from visualization import plot_rewards, create_results_dir
from probMap import probabilityMap


class ProbHATAgent:
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
        
        # Dictionary to store targets and sunk ships
        self.targets = []

        self.prob = probabilityMap()




    def select_action(self, env, state):
        """
        Select action using hunt and target strategy 
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        #if no targets, send a random guess
        if len(self.targets) < 1:

          #take a random probabilistic choice
          return self.prob.guessPoint()


         #if there are targets, gather all possible points
        else:
          possible = []
          for target in self.targets:
            neighbors = self.prob.findAllUnguessedAdgacents(target)
            for neighbor in neighbors:
               possible.append(neighbor)
          
          #then if no possbile neighbors are found, return a random choice
          if len(possible) < 1:
            return self.prob.guessPoint()
          
          else:
            return self.prob.guessPoint(points = possible)
             
    

    def update (self, state, action, reward):
        '''
        this method updates the targets and sunk based on the current state
        '''

        print(state)

        #if the reward is 100, the game has won, no need to update
        if reward == 100:
          return
        
        self.prob.updateMap(action,reward)

        #if the reward is greater than 0, add to targets
        if reward > 0:
            self.targets.append(action)




    def play_game(self, env):

      '''
      this method plays a game of battleship, and returns the number of actions taken
      '''
      state = env.reset()
      num_actions = 0
      done = False

      while not done:
          action = self.select_action(env, state)
          num_actions = num_actions+1
          state, reward, done, _ = env.step(action)
          self.update(state, action, reward)

      
      return num_actions


