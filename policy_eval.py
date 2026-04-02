# Note: used LLMs to search up numpy syntax

import numpy as np

def policy_eval(S, A, P, R, gamma, policy):
    '''
    Params:
    S: number of states
    A: number of actions
    P: numpy array in the shape of A, S, S
    R: numpy array of shape S, A
    gamma: discount factor
    policy: numpy array of shape S, A
    returns:
    value state function after following the policy
    '''
    theta = 1e-6
    V = np.zeros(S)
    while True:
        prev = V.copy()
        # loop over each state
        for v in range(0, S):
            total = 0
            # loop over each action
            for a in range(0, A):
                # store policy[v][a]
                discounted_future = 0
                reward = R[v][a]
                action = policy[v][a]
                for s_prime in range(0, S):
                    discounted_future += gamma*P[a][v][s_prime]*prev[s_prime]
                total += (discounted_future + reward) * action
            V[v] = total
        if np.max(np.abs(V - prev)) < theta:
            break
    
    return V

# testing on small gridworld (generated from lecture by llm)

# Gridworld setup
# States 0-14 are non-terminal, state 15 is terminal
# Actions: 0=up, 1=down, 2=left, 3=right
S = 16
A = 4
gamma = 1.0

P = np.zeros((A, S, S))

# states 1-14 are nonterminal, 0 and 15 are terminal
for s in range(1, 15):
    row, col = s // 4, s % 4
    neighbors = [
        s - 4 if row > 0 else s,  # up
        s + 4 if row < 3 else s,  # down
        s - 1 if col > 0 else s,  # left
        s + 1 if col < 3 else s,  # right
    ]
    for a, s_prime in enumerate(neighbors):
        P[a, s, s_prime] = 1.0

# both terminal states absorb
P[:, 0, 0] = 1.0
P[:, 15, 15] = 1.0

# reward -1 everywhere except terminals
R = np.full((S, A), -1.0)
R[0, :] = 0.0
R[15, :] = 0.0

policy = np.ones((S, A)) / A

V = policy_eval(S, A, P, R, gamma, policy)
print(V.reshape(4, 4))