# version 1.0

import numpy as np
from typing import List, Dict

from numpy.core.defchararray import mod

from utils_soln import *

def create_observation_matrix(env: Environment):
    '''
    Creates a 2D numpy array containing the observation probabilities for each state. 

    Entry (i,j) in the array is the probability of making an observation type j in state i.

    Saves the matrix in env.observe_matrix and returns nothing.
    '''

    #### Your Code Here ####
    matrix = np.zeros((len(env.state_types), env.num_observe_types))
    for i in range(len(env.state_types)):
        for j in range(env.num_observe_types):
            matrix[i][j] = env.observe_probs[env.state_types[i]][j]
    env.observe_matrix = matrix

    


def create_transition_matrices(env: Environment):
    '''
    If the transition_matrices in env is not None, 
    constructs a 3D numpy array containing the transition matrix for each action.

    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.

    Saves the matrices in env.transition_matrices and returns nothing.
    '''

    if env.transition_matrices is not None:
        return

    #### Your Code Here ####
    numstate = len(env.state_types)
    
    matrix = np.zeros((len(env.action_effects),numstate,numstate))
    for i in range(len(env.action_effects)):
        #print("i", i)
        temp = {}
        for key in env.action_effects[i]:
            #print("key", key)
            if key not in range(numstate):
                newkey = key % numstate
                val = env.action_effects[i].get(key)
                temp[newkey] = val
            else:
                temp[key] = env.action_effects[i].get(key)
        env.action_effects[i] = temp
        #print(temp)
        for j in range(numstate):
            for k in range(numstate):
                offset = (k - j) % numstate
                if offset in env.action_effects[i]:
                    matrix[i][j][k] = env.action_effects[i][offset]
    env.transition_matrices = matrix
    return


def forward_recursion(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform the filtering task for all the time steps.

    Calculate and return the values f_{0:0} to f_{0,t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.
    :param probs_init: The initial probabilities over the N states.

    :return: A numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the normalized values of f_{0:k} (0 <= k <= t - 1).
    '''
    
    ### YOUR CODE HERE ###
    N = len(env.state_types)
    t = len(observ)
    create_observation_matrix(env)
    create_transition_matrices(env)
    result = np.zeros((t,N))
    for k in range(t):
        if k == 0:
            observ_type = observ[0]
            f00 = np.zeros((N))
            total = 0.0
            for state in range(N):
                f00[state] = env.observe_matrix[state][observ_type] * probs_init[state]
                total += f00[state]
            f00 = f00 /total
            result[k] = f00
        else:
            observ_type = observ[k]
            action = actions[k-1]
            totalsk = np.zeros((N))
            for state in range(N):
                totalsk += env.transition_matrices[action][state] * result[k-1][state]
            total = 0.0
            f0k = np.zeros((N))
            for state in range(N):
                f0k[state] = env.observe_matrix[state][observ_type] * totalsk[state]
                total += f0k[state]
            f0k = f0k /total
            result[k] = f0k
    return result



def backward_recursion(env: Environment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform the smoothing task for each time step.

    Calculate and return the values b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.

    :return: A numpy array with shape (t+1, N), (N is the number of states)
            the k'th row represents the values of b_{k:t-1} (1 <= k <= t - 1),
            while the k=0 row is meaningless and we will NOT test it.
    '''

    ### YOUR CODE HERE ###
    N = len(env.state_types)
    t = len(observ)
    create_observation_matrix(env)
    create_transition_matrices(env)
    result = np.zeros((t+1,N))
    for i in range(t, 0, -1):
        k = i -1
        if k == t - 1:
            result[i] = [1.0]*N
        else:
            action = actions[i-1]
            observ_type = observ[i]
            totalbk = np.zeros((N))
            for pre in range(N):
                for nexts in range (N):
                    totalbk[pre] += result[i+1][nexts] * env.transition_matrices[action][pre][nexts]* env.observe_matrix[nexts][observ_type]
            result[i] = totalbk
    return result






def fba(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array with shape (t,N) where t = len(observ) and N is the number of states.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env: The environment.
    :param actions: A list of agent's past actions.
    :param observ: A list of observations.
    :param probs_init: The agent's initial beliefs over states
    :return: A numpy array with shape (t, N)
        the k'th row represents the normalized smoothed probability distribution over all the states for time k.
    '''

    ### YOUR CODE HERE ###
    N = len(env.state_types)
    t = len(observ)
    result = np.zeros((t,N))
    f = forward_recursion(env,actions,observ,probs_init)
    b = backward_recursion(env, actions,observ)
    for k in range(t):
        total = 0.0
        for state in range(N):
            result[k][state] = f[k][state]*b[k+1][state]
            total += result[k][state]
        result[k] = result[k]/total
    return result