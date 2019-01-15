# search.py
# ---------
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

import util

class SearchProblem:

    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):

        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    return 0

def getActionsOfPath(path):
    actions = map(lambda node: node[1], path)[1:]
    return actions

def generalSearch(problem, search_algorithm, heuristic=nullHeuristic):
    current_node = (problem.getStartState(), [], 0)
    current_path = [current_node]
    explored = []
    if problem.isGoalState(current_node[0]):
        return current_node[1]

    if search_algorithm is "bfs":
        frontier = util.Queue()
        frontier.push(current_path)
    elif search_algorithm is "dfs":
        frontier = util.Stack()
        frontier.push(current_path)
    elif search_algorithm is "ucs" or search_algorithm is "ass":
        current_path = (current_path, 0)
        frontier = util.PriorityQueue()
        frontier.push(current_path, 0)

    while not frontier.isEmpty():
        current_path = frontier.pop()
        if search_algorithm is "bfs" or search_algorithm is "dfs":
            current_node = current_path[- 1]
        elif search_algorithm is "ucs" or search_algorithm is "ass":
            current_node = current_path[0][-1]

        if problem.isGoalState(current_node[0]):
            if search_algorithm is "ucs" or search_algorithm is "ass":
                current_path = current_path[0]
            return getActionsOfPath(current_path)

        if not current_node[0] in explored:
            explored.append(current_node[0])
            for child_node in problem.getSuccessors(current_node[0]):
                if child_node[0] not in explored:
                    if search_algorithm is "bfs" or search_algorithm is "dfs":
                        next_path = current_path+[child_node]
                        frontier.push(next_path)
                    elif search_algorithm is "ucs" or search_algorithm is "ass":
                        next_path = current_path[0]+[child_node]
                        actions = getActionsOfPath(next_path)
                        cost = problem.getCostOfActions(actions)
                        next_path = (next_path, cost)
                        if search_algorithm is "ucs":
                            frontier.push(next_path, cost)
                        elif search_algorithm is "ass":
                            frontier.push(next_path, cost + heuristic(child_node[0], problem))
    return []
def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    return generalSearch(problem, "dfs")

def breadthFirstSearch(problem):
    return generalSearch(problem, "bfs")

def uniformCostSearch(problem):
    return generalSearch(problem, "ucs")

def aStarSearch(problem, heuristic=nullHeuristic):
     return generalSearch(problem, "ass", heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
