# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from decimal import Decimal
pos_inf = Decimal('Infinity')
neg_inf = Decimal('-Infinity')
import random, util
from game import Agent
import time


class ReflexAgent(Agent):

    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        "*** YOUR CODE HERE ***"
        if newGhostStates:
            closest_ghost_distance, closest_ghost = min(
                (util.manhattanDistance(newPos, ghost.getPosition()), ghost) for ghost in newGhostStates)
            if closest_ghost.scaredTimer == 0:
                ghost_distance_variable = (-2.0 if closest_ghost_distance == 0 else -closest_ghost_distance ** -1)
            else:
                ghost_distance_variable = 0
        else:
            ghost_distance_variable = 0
        if newFood.asList():
            closest_food_distance = min(util.manhattanDistance(newPos, food) for food in newFood.asList())
            if closest_food_distance == 0:
                closest_food_variable = 2
            else:
                closest_food_variable = closest_food_distance ** -1
        else:
            closest_food_variable = 0  # no food

        uneaten_food_variable = -len(newFood.asList())

        weights = uneaten_food_weight, ghost_distance_weight, closest_food_weight = 2.0, 3.0, 1.0
        variables = uneaten_food_variable, ghost_distance_variable, closest_food_variable
        return sum([i*j for (i, j) in zip(weights, variables)])

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        v, action = self.value(gameState, self.index, 0)
        return action

    def value(self, state, agent, depth):
        agent = agent % state.getNumAgents()
        actions = state.getLegalActions(agent)
        if depth == self.depth and agent == 0 or not actions:
            return self.evaluationFunction(state), None
        successors = [(state.generateSuccessor(agent, action), action) for action in actions]
        if agent == 0:
            value, action = max((self.value(successor, agent + 1, depth + 1), action) for successor, action in successors)
        else:
            value, action = min((self.value(successor, agent + 1, depth), action) for successor, action in successors)
        return value[0], action


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        v, action = self.value(gameState, self.index, 0, neg_inf, pos_inf)
        return action

    def value(self, state, agent, depth, a, b):
        agent = agent % state.getNumAgents()
        actions = state.getLegalActions(agent)
        if depth == self.depth and agent == 0 or not actions:
            return self.evaluationFunction(state), None
        elif agent == 0:
            value = (neg_inf, None)
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                next_value, next_action = self.value(successor, agent+1, depth+1, a, b)
                value = max((next_value, action), value)
                a = max(a, value[0])
                if value[0] > b:
                    break
        else:
            value = (pos_inf, None)
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                next_value, next_action = self.value(successor, agent + 1, depth, a, b)
                value = min((next_value, action), value)
                b = min(b, value[0])
                if value[0] < a:
                    break
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        #t0 = time.time()
        v, action = self.value(gameState, self.index, 0)
        #t1 = time.time()
        #total = t1 - t0
        #print total
        return action

    def value(self, state, agent, depth):
        agent = agent % state.getNumAgents()
        actions = state.getLegalActions(agent)
        if depth == self.depth and agent == 0 or not actions:
            return self.evaluationFunction(state), None
        successors = [(state.generateSuccessor(agent, action), action) for action in actions]
        if agent == 0:
            value, action = max((self.value(successor, agent + 1, depth + 1), action) for successor, action in successors)
        else:
            values = []
            for successor, action in successors:
                next_value, action = self.value(successor, agent + 1, depth)
                values.append(next_value)
            value = (sum(values)/float(len(values)), None)
        return value[0], action

def betterEvaluationFunction(currentGameState):
    """
      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    if newGhostStates:
        closest_ghost_distance, closest_ghost = min(
            (util.manhattanDistance(newPos, ghost.getPosition()), ghost) for ghost in newGhostStates)
        if closest_ghost.scaredTimer == 0:
            ghost_distance_variable = (-2.0 if closest_ghost_distance == 0 else -closest_ghost_distance ** -1)
        else:
            ghost_distance_variable = 0
    else:
        ghost_distance_variable = 0
    if newFood.asList():
        closest_food_distance = min(util.manhattanDistance(newPos, food) for food in newFood.asList())
        if closest_food_distance == 0:
            closest_food_variable = 2
        else:
            closest_food_variable = closest_food_distance ** -1
    else:
         closest_food_variable = 0    # no food

    uneaten_food_variable = -len(newFood.asList())
    power_pellet_variable = -len(newGhostStates)
    score_variable = currentGameState.getScore()/500

    weights = uneaten_food_weight, ghost_distance_weight, closest_food_weight, power_pellet_weight, score_weight = 2.0, 3.0, 1.0, 1.0, 10.0
    variables = uneaten_food_variable, ghost_distance_variable, closest_food_variable, power_pellet_variable, score_variable
    return sum([i * j for (i, j) in zip(weights, variables)])


# Abbreviation
better = betterEvaluationFunction