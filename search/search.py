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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import math
from util import Stack
from util import Queue
from util import PriorityQueue
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def mediumClassicSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [e,e,e,e,n,n,n,n,e,e,e,e,e,s,s,s,s,w,w,w,n,n,w,w,w,w,w,w,w,w,w,w,w,w,n,n,n,n,e,n,n,w,w,w,s,s,s,s,s,s,s,s,e,e,e,n,n,e,e,n,n,n,n,n,n,e,e,e,e,e,e,e,s,s,e,e,n,n,e,e,e,s,s,s,s,w,w,n,n,w,w,w,w,w,w,w,w,w,w,w,w,w,s,s,w,e,e,e,e,s,s,s,s,e,e,e,e,e,e,e,n,n,n,n,n,s,e,e,e,s,s]

def mediumMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [n,w,s,e,w,w,s,s,e,e,e,e,s,s,w,w,w,w,s,s,e,e,e,e,s,s,w,w,w,w,s,s,e,e,e,e,s,s,s,w,w,w,w,w,w,w,n,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,s,w,w,w,w,w,w,w,w,w]


def mySearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    startState=problem.getStartState()
    childStates=problem.getSuccessors(startState)
    leftChild=childStates[0]
    print(startState)
    print(childStates)
    print(leftChild)
    return [s]

def depthFirstSearch(problem):
    stack=Stack()
    explored=set()
    initialState= problem.getStartState()
    print(initialState)
    stack.push((initialState,[]))
    while(not stack.isEmpty()):
        state = stack.pop()
        node = state[0]
        path  = state[1]
        if node not in explored:
            explored.add(node)
        if(problem.isGoalState(node)):
            return path
        children=problem.getSuccessors(node)
        #print(children)
        for child in children:
            print(child)
            childNode=child[0]
            childPath = child[1]
            fullPath=path+[childPath]
            if childNode not in explored and childNode not in (element[0] for element in stack.list):
                stack.push((childNode,fullPath))
    return []
    util.raiseNotDefined()

    
def getActionFromTriplet(triple):
    return triple[1]

def breadthFirstSearch(problem):
    queue=Queue()
    explored=set()
    initialState= problem.getStartState()
    queue.push((initialState,[]))
    while(not queue.isEmpty()):
        state = queue.pop()
        node = state[0]
        path  = state[1]
        if node not in explored:
            explored.add(node)
        if(problem.isGoalState(node)):
            return path
        children=problem.getSuccessors(node)
        for child in children:
            childNode=child[0]
            childPath = child[1]
            fullPath=path+[childPath]
            if childNode not in explored and childNode not in (element[0] for element in queue.list):
                queue.push((childNode,fullPath))
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    pq=PriorityQueue()
    explored=set()
    initialState= problem.getStartState()
    pq.push((initialState,[],0),0)
    while(not pq.isEmpty()):
        state = pq.pop()
        node = state[0]
        path  = state[1]
        cost=state[2]
        if node not in explored:
            explored.add(node)
        if(problem.isGoalState(node)):
            return path
        children=problem.getSuccessors(node)
        for child in children:
            childNode=child[0]
            childPath = child[1]
            childCost=child[2]
            fullPath=path+[childPath]
            fullCost=cost+childCost
            if childNode not in explored and childNode not in (element[0] for element in pq.heap):
                pq.push((childNode,fullPath,fullCost),fullCost)
    return []
    return[]
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def manHattanHeuristic(state,problem=None):
    #Distance= abs(x2-x1) + abs(y2-y1)
    #return goal.path-state[2]
    """x and y coordinate of node"""
    node=state
    xOfNode=state[0]
    yOfNode=state[1]
    """x and y coordinate of goal node"""
    goalNode=problem.goal
    xOfGoalNode=goalNode[0]
    yOfGoalNode=goalNode[1] 
    heuristicCost=abs( xOfGoalNode-xOfNode) + abs(yOfGoalNode- yOfNode)
    return heuristicCost

def euclideanDistance (state,problem=None):
    #Distance= sqrt((x2-x1)^2+(y2-y1)^2)
    #return goal.path-state[2]
    """x and y coordinate of node"""
    node=state
    xOfNode=state[0]
    yOfNode=state[1]
    """x and y coordinate of goal node"""
    goalNode=problem.goal
    xOfGoalNode=goalNode[0]
    yOfGoalNode=goalNode[1]
    heuristicCost=math.sqrt(pow(( xOfGoalNode-xOfNode),2) + pow((yOfGoalNode- yOfNode),2))
    return heuristicCost
    
def aStarSearch(problem, heuristic=manHattanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq=PriorityQueue()
    explored=set()
    initialState= problem.getStartState()
    heuristicValue=heuristic(initialState,problem)
    initialPathCost=0
    pq.push((initialState,[],0),initialPathCost+heuristicValue)
    while(not pq.isEmpty()):
        state = pq.pop()
        node = state[0]
        path  = state[1]
        cost=state[2]
        if node not in explored:
            explored.add(node)
        if(problem.isGoalState(node)):
            return path
        children=problem.getSuccessors(node)
        for child in children:
            childNode=child[0]
            childPath = child[1]
            childCost=child[2]
            fullPath=path+[childPath]
            fullPathCost=cost+childCost
            heuristicValue=manHattanHeuristic(node,problem)
            #heuristicValue=euclideanDistance(node,problem)
            if childNode not in explored and childNode not in (element[0] for element in pq.heap):
                pq.push((childNode,fullPath,fullPathCost),fullPathCost+heuristicValue)
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
