from rl_provided import *
import numpy as np
from typing import Tuple, List


def get_transition_prob(n_sa, n_sas, curr_state: State, dir_intended: int, next_state: State) -> float:
    """
    Determine the transition probability based on counts in n_sa and n_sas'.
    curr_state is s. dir_intended is a. next_state is s'.

    @return: 0 if we haven't visited the state-action pair yet (i.e. N_sa = 0).
      Otherwise, return N_sas' / N_sa.
    """
    N_sa = n_sa[curr_state[0], curr_state[1],dir_intended]
    N_sas = n_sas[curr_state[0], curr_state[1],dir_intended,next_state[0],next_state[1]]
    if N_sa == 0:
        return 0
    else:
        return  N_sas/ N_sa


def exp_utils(world, utils, n_sa, n_sas, curr_state: State) -> List[float]:
    """
    @return: The expected utility values for all four possible actions.
    i.e. calculates sum_s'( P(s' | s, a) * U(s')) for all four possible actions.

    The returned list contains the expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the expected utility
    if the agent ends up going down from the current state.
    """
    explst = np.zeros(world.num_actions)
    for i in range(4):
        next_states = get_next_states(world.grid, curr_state)
        state_set = set(next_states)
        sum = 0
        for next in state_set:
            sum += get_transition_prob(n_sa, n_sas, curr_state,i,next)* utils[next[0], next[1]]
        explst[i] = sum
    return explst
    
    


def optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> List[float]:
    """
    @return: The optimistic expected utility values for all four possible actions.
    i.e. calculates f ( sum_s'( P(s' | s, a) * U(s')), N(s, a) ) for all four possible actions.

    The returned list contains the optimistic expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the optimistic expected utility
    if the agent ends up going down from the current state.
    """
    exputils =  exp_utils(world, utils, n_sa, n_sas, curr_state)
    oplst = np.zeros(world.num_actions)
    for i in range(4):
        Nsa = n_sa[curr_state[0], curr_state[1],i]
        if Nsa < n_e:
            oplst[i] = r_plus
        else:
            oplst[i] = exputils[i]
    return oplst



def update_utils(world, utils, n_sa, n_sas, n_e: int, r_plus: float) -> np.ndarray:
    """
    Update the utility values via value iteration until they converge.
    Call `utils_converged` to check for convergence.
    @return: The updated utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    cur = utils
    while 1:
        n_rows, n_cols = world.grid.shape
        next_utils = np.zeros(world.grid.shape, dtype=float)
        for x in range(n_rows):
            for y in range(n_cols):
                state = (x, y)
                if not_goal_nor_wall(world.grid, state):
                    opt_lst = optimistic_exp_utils(world, utils, n_sa, n_sas, state, n_e, r_plus)
                    best_action = get_best_action(world, utils, n_sa, n_sas, state, n_e, r_plus)
                    ut = world.reward + world.discount*opt_lst[best_action]
                    next_utils[x,y] = ut
                else:
                    next_utils[x,y] = utils[x,y]
        if utils_converged(cur, next_utils):
            return next_utils
        cur = next_utils
        


def get_best_action(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> int:
    """
    @return: The best action, based on the agent's current understanding of the world, to perform in `curr_state`.
    """
    opt_utils_for_a = optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)
    return np.argmax(opt_utils_for_a)


def adpa_move(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> Tuple[State, np.ndarray]:
    """
    Execute ADP for one move. This function performs the following steps.
        1. Choose best action based on the utility values (utils).
        2. Make a move by calling `make_move_det`.
        3. Update the counts.
        4. Update the utility values (utils) via value iteration.
        5. Return the new state and the new utilities.

    @return: The state the agent ends up in after performing what it thinks is the best action + the updated
      utilities after performing this action.
    @rtype: A tuple (next_state, next_utils), where:
     - next_utils is an `np.ndarray` of size `world.grid.shape` of type `float`
    """
    action = get_best_action(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)
    next = world.make_move_det(action, n_sa)
    n_sa[curr_state[0],curr_state[1],action] += 1
    n_sas[curr_state[0], curr_state[1],action,next[0],next[1]] += 1
    new_utils = update_utils(world, utils, n_sa, n_sas, n_e, r_plus)
    return next, new_utils



def utils_to_policy(world, utils, n_sa, n_sas) -> np.ndarray:
    """
    @return: The optimal policy derived from the given utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `int`.
    """
    # Initialize the policy.
    policy = np.zeros(world.grid.shape, dtype=int)

    n_rows, n_cols = world.grid.shape
    for x in range(n_rows):
        for y in range(n_cols):
            state = (x, y)
            if not_goal_nor_wall(world.grid, state):
                exp = exp_utils(world, utils, n_sa, n_sas, state)
                policy[x,y] = np.argmax(exp)
            else:
                policy[x,y] = utils[x,y]
    return policy




def is_done_exploring(n_sa, grid, n_e: int) -> bool:
    """
    @return: True when the agent has visited each state-action pair at least `n_e` times.
    """
    n_rows, n_cols = grid.shape
    for x in range(n_rows):
        for y in range(n_cols):
            state = (x, y)
            if not_goal_nor_wall(grid, state):
                for sa in n_sa[x,y]:
                    if sa < n_e:
                        return False
    return True


def adpa(world_name: str, n_e: int, r_plus: float) -> np.ndarray:
    """
    Perform active ADP. Runs a certain number of moves and returns the learned utilities and policy.
    Stops when the agent is done exploring the world and the utility values have converged.
    Call `utils_converged` to check for convergence.

    Note: By convention, our tests expect the utility of a "wall" state to be 0.

    @param world_name: The name of the world we wish to explore.
    @param n_e: The minimum number of times we wish to see each state-action pair.
    @param r_plus: The maximum reward we can expect to receive in any state.
    @return: The learned utilities.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    # Initialize the world
    world = read_world(world_name)
    grid = world.grid
    num_actions = world.num_actions

    # Initialize persistent variable
    utils = np.zeros(grid.shape)
    n_sa = np.zeros((*grid.shape, num_actions))
    n_sas = np.zeros((*grid.shape, num_actions, *grid.shape))

    n_rows, n_cols = grid.shape
    for x in range(n_rows):
        for y in range(n_cols):
            state = (x, y)
            if is_goal(grid, state):
                utils[x,y] = int(grid[x,y])

    req1 = False
    req2 = False
    while 1:
        next_state, next_utils = adpa_move(world, utils, n_sa, n_sas, world.curr_state, n_e, r_plus)
        req1 = is_done_exploring(n_sa, grid, n_e)
        req2 = utils_converged(utils, next_utils)
        if req1 and req2:
            return next_utils
        utils = next_utils

            

