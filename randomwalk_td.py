import numpy as np

def silver_random_walk_prediction(policy, alpha, gamma):
    '''
    Params:
    policy: arbitrary as this is random walk
    alpha: learning rate
    gamma: discount rate
    returns:
    value state vector
    '''
    # initialize states to 0
    values = np.zeros(5)
    for episode in range(0, 10000):
        # always start at state C
        state = 2
        while state >= 0 and state <= 4: 
            # simulate random walk
            movement = np.random.choice([1, -1], p=[0.5, 0.5])
            # terminal states
            if state + movement > 4:
                values[state] = values[state] + alpha*(1 - values[state])
            elif state + movement < 0:
                values[state] = values[state] + alpha*(0 - values[state])

            else:
                values[state] = values[state] + alpha*(0 + gamma*values[state+movement]-values[state])
            state += movement
    return values

print(silver_random_walk_prediction(1, .1, 1))

