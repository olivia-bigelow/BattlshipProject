import numpy as np
import random

"""
This class implements a probability map for the guessing of ship locations
to use this class, create a probability map with map = probabilityMap()
to access probabilities use map.map[action]
whenever an action is taken, call map.updateMap(action, reward)

"""


class probabilityMap:

  def __init__(self):
        """
        Initialize a Hunt and Target agent.
        
        Args:
            action_dim: Dimension of the action space
        """
        self.map = {}
        for i in range(0,100):
          self.map[i] = 0

        self.sunk = {2:0, 3:0, 4:0, 5:0}
        self.hits = []
        self.guesses = []
        self.setWeights()


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


  def setWeights(self):
    
    for ship_size in [2,3,3,4,5]:
          use_size = ship_size - 1
          # check where a ship will fit on the board
          for row in range(10):
            for col in range(10):
                if ((row*10) + (col)) in self.guesses:
                  continue
                # get potential ship endpoints
                endpoints = []
                # add 1 to all endpoints to compensate for python indexing
                if row - use_size >= 0:
                  #start row, startCole, endrow, end coll
                  endpoints.append(((row - use_size, col), (row + 1, col + 1)))
                if row + use_size <= 9:
                  endpoints.append(((row, col), (row + use_size + 1, col + 1)))
                if col - use_size >= 0:
                  endpoints.append(((row, col - use_size), (row + 1, col + 1)))
                if col + use_size <= 9:
                  endpoints.append(((row, col), (row + 1, col + use_size + 1)))

                for (start_row, start_col), (end_row, end_col) in endpoints:
                  guessed = False
                  for subRow in range(start_row, end_row):
                    for subCol in range(start_col, end_col):
                      point = (subRow * 10) + (subCol)

                      if point in self.guesses:
                        guessed = True
                        break
                      self.map[point] += 1
                    if guessed:
                       break
                  if guessed:
                    continue
    
    for guess in self.guesses:
      self.map[guess] = 0



  def inflateHitNeighbors(self, extraweight = 5):
    """
    this function inflates the ungessed neighbors of all hits
    """
    for hit in self.hits:
      for neighbor in self.findAllUnguessedAdgacents(hit):
        self.map[neighbor] = self.map[neighbor] + extraweight


  def updateMap(self, action, reward):
    #add the action to the list of guesses
    self.guesses.append(action)

    #if the reward is greater than 0, add to targets
    if reward > 0:
      self.hits.append(action)



    #update the probability mapping
    self.setWeights()

    #inflate hit nieghbors
    self.inflateHitNeighbors()
        


  def guessPoint(self, points = None, max = False):
    """
    This function guesses a point using the probability map
    The arguments determine the way in which a point is guessed

    points, if points is defined, the function will return an action sampled from the set of points provided
    max, if max is true, the function returns the point that maximizes liklihood, otherwise the function samples points based on their weights
    """


    #check if max
    if max:
      if points is None:
        #if points are undefined, and max is true, return the value taht maximizes weight
        return max(self.map, key=self.map.get)
      else:
        max = -1
        bestPoint = points[0]
        for point in points:
          if self.map[point] > max:
            max = self.map[point]
            bestPoint = point
        return bestPoint
    else:
      if points is None:
        total_weight = sum(self.map.values())
        random_num = random.uniform(0, total_weight)
        cumulative_weight = 0

        for key, weight in self.map.items():
          cumulative_weight += weight
          if random_num < cumulative_weight:
            return key
      else:
        total_weight = 0
        for point in points:
          total_weight = total_weight + self.map[point]
        random_num = random.uniform(0, total_weight)
        cumulative_weight = 0

        for point in points:
          cumulative_weight += self.map[point]
          if random_num < cumulative_weight:
            return point

      
