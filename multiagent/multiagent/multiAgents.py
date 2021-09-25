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
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # 1. Consider the distance to the nearest food; reciprocal dist. to food weigh more than that to ghost
        # 2. Consider the score
        # 3. Consider food left, more points if less food afterwards
        # 4. Consider a penalty (deduct a lot of points) for nearby ghosts
        # use the getScore() function

        # manhattan distance btw new pacman position and closest new ghost position .getGhostPosition()
        # manhattan dist. to closest food, - points?
        # + points if new pac pos has food
        # + points if decrease amount of food left; len(newFood.asList()) vs. len(currFood.asList())
        # - points if manhattan distance to ghost very close
        # + points if manhattan distance < newscaredtime, then can eat ghost
        
        score = 0 # add to successorGameState.getScore()

        min_manhat_food_dist = 1000000000000
        newFood_list = newFood.asList()
        for newFoodPos in newFood_list:
            curr_dist = util.manhattanDistance(newPos, newFoodPos)
            if curr_dist < min_manhat_food_dist:
                min_manhat_food_dist = curr_dist
        if len(newFood_list) > 0:
            score -= min_manhat_food_dist
        
        if len(newFood_list) < len(currentGameState.getFood().asList()):
            score += 100
        
        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Minimax Agent - adversarial search agent 
        # Minimax agent should work with any number of ghosts
        # Minimax tree will have multiple min layers (one for each ghost) for every max layer

        # Goal:
            # Find the optimal move at each state for PacMan (maximizer) against the ghosts (minimizers)
        
        # 1. Check if the current game state is a winning state or a losing state
        # 2. Check if it is Pacman's turn
            # For every possible action, recurse using minimax alg. to get the maximum reward
        # 3. Check if it is the ghosts' (>=1) turn
            # Get the number of agents in the game
            # For every agent, recurse using minimax alg. to get the minimum reward for the ghosts
                # If all the agents moved -> add 1 to the depth and change the agent's index to 0 to signify PacMan's turn

        # buggy because we were only passing 1 arg into min and max
        #not returning anything, need to return an action, keep track of [action, score]

        def minimax_algorithm(depth, agentIndex, gameState):
            
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                score = self.evaluationFunction(gameState)
                return ['', score]

            if (agentIndex == 0): # PacMan's turn
                # PacMan's turn
                v_for_max = ['', float('-inf')]
                for pacMan_possible_action in gameState.getLegalActions(0):
                    poss_action_score = minimax_algorithm(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, pacMan_possible_action))[1]
                    if poss_action_score > v_for_max[1]:
                        v_for_max = [pacMan_possible_action, poss_action_score]
                return v_for_max

            if (agentIndex >= 1): # Ghost's turn
                nextGhostIndex = 1 + agentIndex

                if (nextGhostIndex == gameState.getNumAgents()): # Check if we checked all the ghosts
                    nextGhostIndex = 0 # It's now PacMan's turn
                
                if (nextGhostIndex == 0): # If all the ghosts moved..
                    depth += 1 # check the next min layer
                v_for_min = ['', float('inf')]
                for ghost_possible_action in gameState.getLegalActions(agentIndex):
                    if nextGhostIndex == 0:
                        poss_action_score = minimax_algorithm(depth, nextGhostIndex, gameState.generateSuccessor(agentIndex, ghost_possible_action))[1]
                        if poss_action_score < v_for_min[1]:
                            v_for_min = [ghost_possible_action, poss_action_score]
                    else:
                        poss_action_score = minimax_algorithm(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, ghost_possible_action))[1]
                        if poss_action_score < v_for_min[1]:
                            v_for_min = [ghost_possible_action, poss_action_score]
                return v_for_min

        minimax_action, _ = minimax_algorithm(0, self.index, gameState)
        return minimax_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta_algorithm(depth, agentIndex, gameState, alpha, beta):
            
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                score = self.evaluationFunction(gameState)
                return ['', score]

            if (agentIndex == 0): # PacMan's turn
                # PacMan's turn
                v_for_max = ['', float('-inf')]
                for pacMan_possible_action in gameState.getLegalActions(0):
                    poss_action_score = alphabeta_algorithm(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, pacMan_possible_action), alpha, beta)[1]
                    if poss_action_score > v_for_max[1]:
                        v_for_max = [pacMan_possible_action, poss_action_score]
                    if v_for_max[1] > beta:
                        return v_for_max
                    alpha = max(alpha, v_for_max[1])
                return v_for_max

            if (agentIndex >= 1): # Ghost's turn
                nextGhostIndex = 1 + agentIndex

                if (nextGhostIndex == gameState.getNumAgents()): # Check if we checked all the ghosts
                    nextGhostIndex = 0 # It's now PacMan's turn
                
                if (nextGhostIndex == 0): # If all the ghosts moved..
                    depth += 1 # check the next min layer
                v_for_min = ['', float('inf')]
                for ghost_possible_action in gameState.getLegalActions(agentIndex):
                    if nextGhostIndex == 0:
                        poss_action_score = alphabeta_algorithm(depth, nextGhostIndex, gameState.generateSuccessor(agentIndex, ghost_possible_action), alpha, beta)[1]
                        if poss_action_score < v_for_min[1]:
                            v_for_min = [ghost_possible_action, poss_action_score]
                        if v_for_min[1] < alpha:
                            return v_for_min
                        beta = min(beta, v_for_min[1])
                    else:
                        poss_action_score = alphabeta_algorithm(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, ghost_possible_action), alpha, beta)[1]
                        if poss_action_score < v_for_min[1]:
                            v_for_min = [ghost_possible_action, poss_action_score]
                        if v_for_min[1] < alpha:
                            return v_for_min
                        beta = min(beta, v_for_min[1])
                return v_for_min

        alpha = float('-inf')
        beta = float('inf')
        alphabeta_action, _ = alphabeta_algorithm(0, self.index, gameState, alpha, beta)
        return alphabeta_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        
        def expectimax_action(depth, agentIndex, gameState):
            
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                score = self.evaluationFunction(gameState)
                return ['', score]

            if (agentIndex == 0): # PacMan's turn
                # PacMan's turn
                v_for_max = ['', float('-inf')]
                for pacMan_possible_action in gameState.getLegalActions(0):
                    poss_action_score = expectimax_action(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, pacMan_possible_action))[1]
                    if poss_action_score > v_for_max[1]:
                        v_for_max = [pacMan_possible_action, poss_action_score]
                return v_for_max

            if (agentIndex >= 1): # Ghost's turn
                nextGhostIndex = 1 + agentIndex

                if (nextGhostIndex == gameState.getNumAgents()): # Check if we checked all the ghosts
                    nextGhostIndex = 0 # It's now PacMan's turn
                
                if (nextGhostIndex == 0): # If all the ghosts moved..
                    depth += 1 # check the next min layer
                
                expectimax_for_min = ['', 0]

                for ghost_possible_action in gameState.getLegalActions(agentIndex):
                    avg_action_score = expectimax_for_min[1]

                    if nextGhostIndex == 0: 
                        avg_action_score += (expectimax_action(depth, nextGhostIndex, gameState.generateSuccessor(agentIndex, ghost_possible_action))[1] / len(gameState.getLegalActions(agentIndex)))
                        expectimax_for_min = [ghost_possible_action, avg_action_score]

                    else:
                        avg_action_score += (expectimax_action(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, ghost_possible_action))[1] / len(gameState.getLegalActions(agentIndex)))

                        if avg_action_score > expectimax_for_min[1]:
                            expectimax_for_min = [ghost_possible_action, avg_action_score]
                return expectimax_for_min

        expectimax_action, _ = expectimax_action(0, self.index, gameState)
        return expectimax_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
        # 1. Consider the distance to the nearest food; reciprocal dist. to food weigh more than that to ghost
        # 2. Consider the score
        # 3. Consider food left, more points if less food afterwards
        # 4. Consider a penalty (deduct a lot of points) for nearby ghosts
        # use the getScore() function

        # manhattan distance btw new pacman position and closest new ghost position .getGhostPosition()
        # manhattan dist. to closest food, - points?
        # + points if new pac pos has food
        # + points if decrease amount of food left; len(newFood.asList()) vs. len(currFood.asList())
        # - points if manhattan distance to ghost very close
        # + points if manhattan distance < newscaredtime, then can eat ghost
    """
    score = 0

    currPos = currentGameState.getPacmanPosition()
    currFoodList = currentGameState.getFood().asList()

    min_manhat_food_dist = 1000000000000
    for currFood in currFoodList:
        curr_dist = util.manhattanDistance(currPos, currFood)
        if curr_dist < min_manhat_food_dist:
            min_manhat_food_dist = curr_dist

    if len(currFoodList) > 0:
        score -= min_manhat_food_dist
    
    if len(currFoodList) < len(currentGameState.getFood().asList()):
        score += 100

    return score + currentGameState.getScore()
    
# Abbreviation
better = betterEvaluationFunction
