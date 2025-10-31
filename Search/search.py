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


def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    fringe = util.Stack()
    visited = set()
    
    start_state = problem.getStartState()
    fringe.push((start_state, []))  # state, path

    while not fringe.isEmpty():
        state, path = fringe.pop()

        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, step_cost in problem.getSuccessors(state):
            if successor not in visited:
                fringe.push((successor, path + [action]))

    return []

def breadthFirstSearch(problem: SearchProblem):

    # Initialize the fringe and visited set
    fringe = util.Queue()
    visited = set()

    # Start with the initial state and an empty path
    start_state = problem.getStartState()
    fringe.push((start_state, []))
    visited.add(start_state)

    while not fringe.isEmpty():
        current_state, path = fringe.pop()

        # If the goal is reached, return the path
        if problem.isGoalState(current_state):
            return path

        # Expand successors
        for successor, action, cost in problem.getSuccessors(current_state):
            if successor not in visited:
                visited.add(successor)  # mark as visited when added
                fringe.push((successor, path + [action]))

    return []

def uniformCostSearch(problem: SearchProblem):
    # Priority queue: (state, path, total_cost)
    fringe = util.PriorityQueue()
    start_state = problem.getStartState()
    fringe.push((start_state, [], 0), 0)

    # Visited dictionary to store the lowest cost to reach each state
    visited = {}

    while not fringe.isEmpty():
        current_state, path, current_cost = fringe.pop()

        # If current_state already visited with lower cost, skip
        if current_state in visited and visited[current_state] <= current_cost:
            continue

        visited[current_state] = current_cost

        # Goal check
        if problem.isGoalState(current_state):
            return path

        # Expand successors
        for successor, action, step_cost in problem.getSuccessors(current_state):
            total_cost = current_cost + step_cost
            if successor not in visited or total_cost < visited.get(successor, float('inf')):
                fringe.push((successor, path + [action], total_cost), total_cost)

    return []

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    fringe = util.PriorityQueue()
    start_state = problem.getStartState()
    fringe.push((start_state, [], 0), heuristic(start_state, problem))

    visited = {}  # maps state -> lowest cost (g) seen so far

    while not fringe.isEmpty():
        current_state, path, g_cost = fringe.pop()

        if current_state in visited and visited[current_state] <= g_cost:
            continue

        visited[current_state] = g_cost

        if problem.isGoalState(current_state):
            return path

        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_g = g_cost + step_cost
            f = new_g + heuristic(successor, problem)

            if successor not in visited or new_g < visited.get(successor, float('inf')):
                fringe.push((successor, path + [action], new_g), f)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
