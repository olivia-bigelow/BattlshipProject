"""
This is the hunt and target agent


The agent stores the ships sunk, the guesses made, and the target points

The agent picks actions randomly while hunting, and then guesses around a target point (hit) until fully explored

This agent contains a few helper methods, some of which may be better suited for a util.py file
findAllAdjacents: finds all legal adjacent points on the board
findAllUnguessedAdjacents finds all legal adjacent points not previously guessed
parityguess: Allows for more optimal hunting, it hunts based on the current largest undiscovered ship
update: updates all of the stores of data based on a response to a new action taken



This class only runs a single game when play is called. 

This needs to implement means to determine sunk ships, if thats implemented, the parity can offer a much better choice. 
It also could have smarter guessing, implementing more heuristics on ship placement






"""
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

class HATAgent:
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

        self.sunk = {2:0, 3:0, 4:0, 5:0}
        self.guesses = []



    def findAllAdjacents(self, point):
      #compute all points in 4 directions

      #on the left edge
      if point % 10 ==0:
        points = [point+1, point+10, point-10]

      #on the right edge
      if point % 10 == 9:
        points = [point-1, point+10, point-10]
      else:
        points = [point+1,point-1, point+10, point-10]
      keep = []
      for p in points:
        if ((p < 0)or (p > 99)):
          continue
        keep.append(p)
      return keep
        

    def findAllUnguessedAdgacents(self, point):
      points = self.findAllAdjacents(point)

      keep = []
      for p in points:
        if p in self.guesses:
          continue
        keep.append(p)
      
      #return 
      return keep 



    def parityGuess(self, implementedShipSunk = False):
      """
      this function creates a set of parity states based on the highest unsunk ship
      """
      points = [0,2,4,6,8, 
                11,13,15,17,19,
                20,22,24,26,28,
                31,33,35,37,39,
                40,42,44,46,48,
                51,53,55,57,59,
                60,62,64,66,68,
                71,73,75,77,79,
                80,82,84,86,88,
                91,93,95,97,99]
      if implementedShipSunk:
        if self.sunk[5] < 1:
          points = [0,5,
                    11,16,
                    22, 27,
                    33, 38,
                    44, 49,
                    50, 55,
                    61, 66, 
                    72, 77, 
                    83, 88,
                    94, 99]

        elif self.sunk[4] < 1:
            points = [0,4,8,
                      13, 17, 
                      22, 26, 
                      31, 35, 39, 
                      40, 44, 48, 
                      53, 57, 
                      62, 66, 
                      71, 75, 79, 
                      80, 84, 88, 
                      93, 97]
        elif self.sunk[3]<2:
          points = [0,3,6,9, 
                    11, 14, 17, 
                    22, 25, 28, 
                    30, 33, 36, 39, 
                    41, 44, 47, 
                    52, 55, 58, 
                    66, 63, 67, 
                    71, 74, 77, 
                    82, 85, 88, 
                    90, 93, 96, 99]


      ret = []
      for p  in points:
        if p in self.guesses: 
           continue
        ret.append(p)
      
      return ret 

    

    




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
          return random.choice(self.parityGuess())


        targetPoint = self.targets[len(self.targets)-1]

        possible = self.findAllUnguessedAdgacents(targetPoint)

        while len(possible) < 1:
          self.targets.pop()
          if len(self.targets) < 1:
            return random.choice(self.parityGuess())

          targetPoint = self.targets[len(self.targets)-1]

          possible = self.findAllUnguessedAdgacents(targetPoint)
        

        return random.choice(possible)
    






    def update (self, state, action, reward):
        '''
        this method updates the targets and sunk based on the current state
        '''

        print(state)

        #if the reward is 100, the game has won, no need to update
        if reward == 100:
          return
        
        #add the action to the list of guesses
        self.guesses.append(action)

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



