import numpy as np

def policy_improvement(S, A, P, R, gamma, policy):
    theta = 1e-6
    while True:    
        V = policy_eval(S, A, P, R, gamma, policy)
        # put this inside a while loop
        policy_new = np.zeros((S, A))
        # iterate over all the states
        for s in range(0, S):
            # find the best action
            action = 0
            max_q = -float("inf")
            for a in range(0, A):
                reward = R[s][a]
                discounted_future = 0
                for s_prime in range(0, S):
                    discounted_future += gamma*P[a][s][s_prime]*V[s_prime]
                q = reward + discounted_future
                if q > max_q:
                    action = a
                    max_q = q
            policy_new[s][action] = 1
        if np.max(np.abs(policy_new - policy)) < theta:
            break
        policy = policy_new
        
    return V, policy



# helper
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


# AI Generated test case
S = 16
A = 4
gamma = 1.0

P = np.zeros((A, S, S))
for s in range(1, 15):
    row, col = s // 4, s % 4
    neighbors = [
        s - 4 if row > 0 else s,
        s + 4 if row < 3 else s,
        s - 1 if col > 0 else s,
        s + 1 if col < 3 else s,
    ]
    for a, s_prime in enumerate(neighbors):
        P[a, s, s_prime] = 1.0

P[:, 0, 0] = 1.0
P[:, 15, 15] = 1.0

R = np.full((S, A), -1.0)
R[0, :] = 0.0
R[15, :] = 0.0

policy_init = np.ones((S, A)) / A
V_opt, policy_opt = policy_improvement(S, A, P, R, gamma, policy_init)

print("Optimal Value Function:")
print(V_opt.reshape(4, 4))

print("\nOptimal Policy (0=up, 1=down, 2=left, 3=right):")
print(np.argmax(policy_opt, axis=1).reshape(4, 4))

action_names = ['up', 'down', 'left', 'right']
print("\nOptimal Policy (readable):")
for s in range(S):
    row, col = s // 4, s % 4
    best = np.argmax(policy_opt[s])
    print(f"  state ({row},{col}): {action_names[best]}")