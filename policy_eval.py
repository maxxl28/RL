import numpy as np

def policy_eval(S, A, P, R, gamma, policy):
    '''
    Params:
    S: number of states
    A: number of actions
    P: numpy array in the shape of A, S, S
    R: numpy array of shape S, A
    gamma: discount factor
    policy: 
    '''
    theta = 1e-6
    