"""
Policy Gradient REINFORCE Agent for Pacman
Implemented using TensorFlow

"""

import random
import numpy as np
import tensorflow as tf
from pgutils import * 

#from game import Agent

NUM_EPISODES = 100

class ReinforceAgent(object):
    """
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        # currently random choice model
        choice = random.choice(legalMoves)
        print self.evaluationNetwork(gameState, choice)
        return choice
    """

    def oneHotVector(self, state):
        one_hot_state = []
        for s in state:
            if s == 0:
                v = [0,0,0,0]
            elif s == 1:
                v = [1,0,0,0]
            elif s == 2:
                v = [0,1,0,0]
            elif s == 3:
                v = [0,0,1,0]
            elif s == 4:
                v = [0,0,0,1]
            else:
                raise ValueError('Invalid State: %d' %s)
            one_hot_state.append(v)
        flattened_onehot = lambda x: [i for sub in x for i in sub]
        return flattened_onehot(one_hot_state)

    def createState(self, pac_pos, ghost_pos, cap_pos, food_pos):
        """
        0 : None | 1 : Pacman | 2 : Ghost | 3 : Capsule | 4 : Food
        Creates 1-D vector ready as NN input

        """
        j, k = 0, 0
        current_state = np.array(food_pos, dtype=int)
        for i, e in enumerate(food_pos):
            if e:
                current_state[i] = 4
            else:
                current_state[i] = 0
        current_state[gridToArray(pac_pos)] = 1
        while j < len(ghost_pos):
            current_state[gridToArray(ghost_pos[j])] = 2
            j += 1
        while k < len(cap_pos):
            current_state[gridToArray(cap_pos[k])] = 3
            k += 1
        #return current_state
        return self.oneHotVector(current_state)

    def evaluationNetwork(self, currentGameState):
        #successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacmanPosition = currentGameState.getPacmanPosition()
        ghostPositions = currentGameState.getGhostPositions()
        capsulePositions = currentGameState.getCapsules()
        currScore = currentGameState.getScore()
        isOver = currentGameState.isWin() or currentGameState.isLose()

        def vanilla_ret_foods():
            # We can improve performance with memoize
            # only update the food location in loc reachable by pacman
            i = 0
            food_list = []
            while i < WORLD_WIDTH:
                j = 0
                while j < WORLD_HEIGHT:
                    food_list.append(currentGameState.getFood()[i][j])
                    j += 1
                i += 1
            return food_list
        game_state = self.createState(pacmanPosition, ghostPositions, capsulePositions, vanilla_ret_foods())

        return game_state, currScore, isOver
