# multi_agents.py
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


from util import manhattan_distance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        get_action() chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best







        "Add more of your code here if you want to"
        





        

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current games state (from which you 
        can retrieve proposed successor GameStates) (pacman.py) and returns a number, 
        where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (all_food) and Pacman position after moving (pacman_position).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        pacman_position = successor_game_state.get_pacman_position()
        all_food = successor_game_state.get_food()
        ghost_states = successor_game_state.get_ghost_states()
        scared_times = [ghost_state.scared_timer for ghost_state in ghost_states]


        "distance to food score"
        potential_score = successor_game_state.get_score()
        food_list = all_food.as_list()
        food_distance = []
        if len(food_list) > 0:
            for food in food_list:
                food_distance.append(manhattan_distance(pacman_position, food))
            potential_score += 5/min(food_distance)
        
        "ghost positions"
        ghost_positions = successor_game_state.get_ghost_positions()
        ghost_distances = []
        for ghost in ghost_positions:
            ghost_distances.append(manhattan_distance(pacman_position, ghost))
            "killing ghost boost"
            if  max(scared_times) > 0:
                potential_score += 50/min(ghost_distances)
            else:
                "if too close run"
                if min(ghost_distances) < 3:
                    potential_score -=100
        
        return potential_score
        util.raise_not_defined()
        

        

def score_evaluation_function(current_game_state: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):

        num_agents = game_state.get_num_agents()

        def minimax_function(game_state, depth, agent_index):
            if depth == self.depth or game_state.is_win() or game_state.is_lose():
                return self.evaluation_function(game_state)
            
            if agent_index == 0:
                return max_value(game_state, depth, agent_index)
            else:
                return min_value(game_state, depth, agent_index)

        def max_value(game_state, depth, agent_index):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)
            best_score = float('-inf')
            best_action = None
            for action in actions:
                successor_state = game_state.generate_successor(agent_index, action)
                value = minimax_function(successor_state, depth, 1)  # Next agent is ghost 1
                if value > best_score:
                    best_score = value
                    best_action = action
            if depth == 0:
                return best_action
            return best_score
        
        def min_value(game_state, depth, agent_index):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)
            best_score = float('inf')
            next_agent = agent_index + 1
            for action in actions:
                successor_state = game_state.generate_successor(agent_index, action)
                if next_agent < num_agents: 
                    value = minimax_function(successor_state, depth, next_agent)
                else:
                    value = minimax_function(successor_state, depth + 1, 0)
                best_score = min(best_score, value)
            return best_score
        return minimax_function(game_state, 0, 0)

        
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):

        def alpha_beta(state, depth, agent_index, alpha, beta):

            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)
            
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:  
                return self.evaluation_function(state)
            
            if agent_index == 0: 
                best_value = float("-inf")
                best_action = None
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = alpha_beta(successor, depth, 1, alpha, beta) 
                    if value > best_value:
                        best_value = value
                        best_action = action
                    alpha = max(alpha, best_value)
                    if alpha > beta: 
                        break
                return best_action if depth == 0 else best_value
            else: 
                best_value = float("inf")
                next_agent = agent_index + 1
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    if next_agent < state.get_num_agents(): 
                        value = alpha_beta(successor, depth, next_agent, alpha, beta)
                    else:  
                        value = alpha_beta(successor, depth + 1, 0, alpha, beta)
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta < alpha:
                        break
                return best_value
            
        return alpha_beta(game_state, 0, 0, float("-inf"), float("inf"))
        util.raise_not_defined()
         
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state: GameState):


        def expectimax(state, depth, agent_index):


            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions: 
                return self.evaluation_function(state)

            if agent_index == 0:
                best_value = float("-inf")
                best_action = None
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = expectimax(successor, depth, 1)
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_action if depth == 0 else best_value
            else: 
                total_value = 0
                next_agent = agent_index + 1
                num_actions = len(legal_actions)
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    if next_agent < state.get_num_agents():
                        value = expectimax(successor, depth, next_agent)
                    else: 
                        value = expectimax(successor, depth + 1, 0)
                    total_value += value
                return total_value / num_actions
             
        return expectimax(game_state, 0, 0)
        util.raise_not_defined()        
        
        

def better_evaluation_function(game_state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    I pretty much copied my method from Question 1, but I added more incentive to eat close 
    capsules over close food. I also subtracted score based on how much food was remaining
    """
 
    pacman_position = game_state.get_pacman_position()
    food = game_state.get_food()
    capsules = game_state.get_capsules()
    score = game_state.get_score()
    ghost_states = game_state.get_ghost_states()

    food_list = food.as_list()
    if len(food_list) > 0: 
        food_distances = [manhattan_distance(pacman_position, food_pos) for food_pos in food_list]
        score += 5 / min(food_distances)  

    capsule_distances = []
    for capsule in capsules:
        capsule_distances.append(manhattan_distance(pacman_position, capsule))
        score += 20 / min(capsule_distances)

    ghost_distances = []
    for ghost in ghost_states:
        ghost_pos = ghost.get_position()
        ghost_distances.append(manhattan_distance(pacman_position, ghost_pos))
        if ghost.scared_timer > 0:
            score += 50 / min(ghost_distances) 
        else:
            if min(ghost_distances) < 3:
                score -= 100 
    remaining_food = len(food_list)
    score -= 10 * remaining_food 

    return score

    util.raise_not_defined()


# Abbreviation
better = better_evaluation_function
