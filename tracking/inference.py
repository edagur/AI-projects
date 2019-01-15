# inference.py
# ------------
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


import itertools
import util
import random
import busters
import game

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"

        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
    #    print noisyDistance
     #   print self.beliefs
      #  print emissionModel

        allPossible = util.Counter()
        for position in self.legalPositions:
            if noisyDistance == None:
                allPossible[position] = 0
            else:
                trueDistance = util.manhattanDistance(pacmanPosition, position)
                allPossible[position] = self.beliefs[position] * emissionModel[trueDistance]

        if noisyDistance == None:
            allPossible[self.getJailPosition()] = 1
        "*** END YOUR CODE HERE ***"

        allPossible.normalize()
        self.beliefs = allPossible
    def elapseTime(self, gameState):

        "*** YOUR CODE HERE ***"
        allPossible = util.Counter()
        for oldPos in self.legalPositions:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            for newPos, prob in newPosDist.items():
                allPossible[newPos] += self.beliefs[oldPos] * prob
        allPossible.normalize()
        self.beliefs = allPossible


    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        "*** YOUR CODE HERE ***"
        locations = self.legalPositions
        self.particles = []
        a = self.numParticles % len(locations)
        for n in range(self.numParticles, len(locations), -len(locations)):
            self.particles += locations
        self.particles += locations[0:a-1]

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        "*** YOUR CODE HERE ***"
        if noisyDistance == None:
            self.particles = [self.getJailPosition()] * len(self.particles)
        else:
            sampleSpace = util.Counter()
            for particle in self.particles:
                trueDistance = util.manhattanDistance(pacmanPosition, particle)
                sampleSpace[particle] += emissionModel[trueDistance]
            sampleSpace.normalize()
            if sampleSpace.totalCount() == 0:
                #print "now"
                self.initializeUniformly(gameState)
            else:
                for i in range(0, len(self.particles)):
                    self.particles[i] = util.sample(sampleSpace)

    def elapseTime(self, gameState):

        "*** YOUR CODE HERE ***"
        newParticles = []
        for oldPos in self.particles:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            newParticles.append(util.sample(newPosDist))
        self.particles = newParticles

    def getBeliefDistribution(self):
        "*** YOUR CODE HERE ***"
        self.beliefs = util.Counter()
        for p in self.particles:
            self.beliefs[p] += 1
        self.beliefs.normalize()
        return self.beliefs

class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1:
            jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        "*** YOUR CODE HERE ***"
        self.particles = list(itertools.product(self.legalPositions, self.legalPositions))

    def addGhostAgent(self, agent):
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):

        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts:
            return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

        "*** YOUR CODE HERE ***"
        for g in range(self.numGhosts):
            if noisyDistances[g] == None:
                for i in range(0, len(self.particles)):
                    self.particles[i] = self.getParticleWithGhostInJail(self.particles[i], g)
            else:
                sampleSpace = util.Counter()
                for particle in self.particles:
                    trueDistance = util.manhattanDistance(pacmanPosition, particle[g])
                    sample = emissionModels[g][trueDistance]
                    sampleSpace[particle] += sample
                sampleSpace.normalize()

                if sampleSpace.totalCount() == 0:
                    #print 'now'
                    self.initializeParticles()
                    for i in range(0, len(self.particles)):
                        self.particles[i] = self.getParticleWithGhostInJail(self.particles[i], g)
                else:
                    Particles = []
                    for i in range(0, self.numParticles):
                        Particles.append(util.sample(sampleSpace))
                    self.particles = Particles

    def getParticleWithGhostInJail(self, particle, ghostIndex):

        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)
            "*** YOUR CODE HERE ***"
            for i in range(0, self.numGhosts):
                newGameState = setGhostPositions(gameState, newParticle)
                ghostIndex = i
                ghostAgent = self.ghostAgents[i]
                ghostPosDist = getPositionDistributionForGhost(newGameState, ghostIndex, ghostAgent)
                newParticle[i] = util.sample(ghostPosDist)

            "*** END YOUR CODE HERE ***"
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

    def getBeliefDistribution(self):
        "*** YOUR CODE HERE ***"
        self.belief = util.Counter()
        for particle in self.particles:
            self.belief[particle] += 1
        self.belief.normalize()
        return self.belief


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState
