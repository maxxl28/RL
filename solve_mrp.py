# Note: used LLMs to search up numpy syntax

# imports
import numpy as np

def solve_mrp(r, gamma, transition_matrix):
    '''
    Params: r, reward vector, gamma, discount, transition_matrix (namesake)
    Returns: solved mrp
    '''
    n = transition_matrix.shape[0]
    i = np.eye(n)
    a = np.linalg.inv((i-gamma*transition_matrix))
    return a @ r

if __name__ == '__main__':
    test_gamma = 0.9
    test_r = np.array([1.0, 0.0, 0.0]) 
    test_t_matrix = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])


    result = solve_mrp(test_r, test_gamma, test_t_matrix)
    print(result)
        