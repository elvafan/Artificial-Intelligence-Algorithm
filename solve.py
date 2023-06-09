from copy import deepcopy
from queue import PriorityQueue
from board import *

def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    frontier = PriorityQueue()
    init_state = State(init_board,hfn,hfn(init_board),0,None)
    frontier.put((init_state.f,(init_state.id,(0, init_state))))
    explored = set()
    while(not frontier.empty()):
        curstate = frontier.get()[1][1][1]
        if curstate.board not in explored:
            explored.add(curstate.board)
            if is_goal(curstate):
                return (get_path(curstate), curstate.depth)
            successors_lst = get_successors(curstate)
            #print("list")
            for successor in successors_lst:
                #print(successor.f)
                #print(successor.id)
                successor.depth = curstate.depth + 1
                successor.f = hfn(successor.board) + successor.depth
                frontier.put((successor.f,(successor.id,(successor.parent.id, successor))))
        ''''   
        print("queue")
        while (not frontier.empty()): 
            f =frontier.get()
            print(f)
        '''
    return ([], -1)



def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    frontier = []
    init_state = State(init_board,zero_heuristic,0,0,None)
    frontier.append(init_state)
    explored = set()
    while (len(frontier)> 0):
        curstate = frontier.pop()
        if curstate.board not in explored:
            explored.add(curstate.board)
            ''''
            print("curstate")
            curstate.board.display()
            print(state.id)
            '''
            if is_goal(curstate):
                return (get_path(curstate), curstate.depth)
            successors_lst = get_successors(curstate)
            successors_lst = sorted(successors_lst, key = lambda state: state.id, reverse= True)
            #print("list")
            for successor in successors_lst:
                #print(successor.id)
                #successor.board.display()
                successor.depth = curstate.depth + 1
                frontier.append(successor)
    return ([], -1)



def newstate(oldstate, oldcar, newcoord):
    #print("new state")
    carlst = []
    for car in oldstate.board.cars:
        if car.fix_coord == oldcar.fix_coord and car.var_coord == oldcar.var_coord and car.orientation == oldcar.orientation:
            #print("same car")
            newcar = deepcopy(oldcar)
            newcar.set_coord(newcoord)
        else:
            newcar = deepcopy(car)
        carlst.append(newcar)
        ''''
        print("car" + car.orientation)
        print(car.fix_coord)
        print(car.var_coord)
        '''
    newboard = Board(oldstate.board.name,oldstate.board.size,carlst)
    #newboard.display()
    newstate = State(newboard,oldstate.hfn,oldstate.f,oldstate.depth,oldstate)
    return newstate

def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """
    successor_list = []
    for car in state.board.cars:
        if car.orientation == 'h':
            #move left
            newcoord = car.var_coord - 1
            while newcoord >= 0:
                if state.board.grid[car.fix_coord][newcoord] == '.':
                    #print("move left")
                    suc = newstate(state,car,newcoord)
                    successor_list.append(suc)
                    newcoord -= 1
                else:
                    break
            #move right
            newcoord = car.var_coord + 1
            rightmost = car.var_coord + car.length
            while rightmost < 6:
                if state.board.grid[car.fix_coord][rightmost] == '.':
                    #print("move right")
                    suc = newstate(state,car,newcoord)                
                    successor_list.append(suc)
                    newcoord += 1
                    rightmost += 1
                else:
                    break
        else:
            #move up
            newcoord = car.var_coord - 1
            while newcoord >= 0:
                if state.board.grid[newcoord][car.fix_coord] == '.':
                    #print("move up")
                    suc = newstate(state,car,newcoord)
                    successor_list.append(suc)
                    newcoord -= 1
                else:
                    break
            #move down
            newcoord = car.var_coord + 1
            rightmost = car.var_coord + car.length
            while rightmost < 6:
                if state.board.grid[rightmost][car.fix_coord] == '.':
                    #print("move down")
                    suc = newstate(state,car,newcoord)                
                    successor_list.append(suc)
                    newcoord += 1
                    rightmost += 1
                else:
                    break
    return successor_list

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    for car in state.board.cars:
        if car.is_goal:
            return car.var_coord == 4
    return False

def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    # can we assume state is not null
    list = []
    curstate = state
    while curstate.parent != None:
        list.insert(0,curstate)
        curstate = curstate.parent
    list.insert(0,curstate)
    return list


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    # if is goal is changed  the following three need to be changed
    goalx = 0 #the left most possible coord to blocking the car
    for car in board.cars:
        if car.is_goal:
            if car.var_coord == 4:
                return 0
            goalx = car.var_coord
    goalx += 2
    hv = 1
    for car in board.cars:
        if car.orientation == 'v' and car.fix_coord >= goalx:
            if car.var_coord <= 2 and car.var_coord + car.length > 2:
                ''''
                print(car.var_coord)
                print(car.fix_coord)
                print(car.orientation)
                print(car.length)
                print("--")
                '''
                hv += 1
    return hv
            


def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError
