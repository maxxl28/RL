import numpy as np
import matplotlib.pyplot as plt

def prediction(policy):
    '''
    Params:
    policy: a 3 dimensional vector
    First dimension: ace yes or no
    Second dimension: your hands
    Third dimension: dealer's hands
    returns:
    value state vector
    '''
    visits = np.zeros((2, 10, 10))
    values = np.zeros((2, 10, 10))
    for episode in range(0, 10000):
        '''
        pseudo code: 
        Helper: pick a random state 
        go to the goal (by playing)
        keep track of the states visited
        reach a terminal condition
        then add the value to the states, so first state gets length of states visited  * 1 or 0 or -1, etc
        '''
        states_visited, goal = play_episode(policy)
        states_visited = set(states_visited) # for first visit mc
        for states in states_visited:
            visits[states[0], states[1], states[2]] += 1
            values[states[0], states[1], states[2]] += goal # using gamma of 1

    return values/visits

def play_episode(policy):
    '''
    used LLMs to help debug
    '''
    # initialize state randomly
    player_sum = np.random.randint(12, 22) # 12-21
    dealer_card = np.random.randint(1, 11) # ace(1) to 10
    usable_ace = np.random.randint(0, 2) # 0 or 1

    states_visited = []

    while True:
        state = (usable_ace, player_sum - 12, dealer_card - 1)
        states_visited.append(state)

        action = policy[usable_ace, player_sum - 12, dealer_card - 1]

        if action == 0:  # twist
            new_card = np.random.randint(1, 11)
            player_sum += new_card
            if player_sum > 21:
                return states_visited, -1  # bust
        else:  # stick
            dealer_sum = dealer_card
            while dealer_sum < 17:
                dealer_sum += np.random.randint(1, 11)
            if player_sum > dealer_sum or dealer_sum > 21:
                return states_visited, 1
            elif player_sum == dealer_sum:
                return states_visited, 0
            else:
                return states_visited, -1
            
# test on policy from lecture (LLM generated fully)
if __name__ == "__main__":

    # policy from lecture: stick (1) if sum >= 20, otherwise twist (0)
    # dimensions: (usable_ace, player_sum_idx, dealer_card_idx)
    policy = np.zeros((2, 10, 10), dtype=int)
    policy[:, 8:, :] = 1  # sum indices 8,9 correspond to sums 20,21

    V = prediction(policy)

    # plot value function for no usable ace
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ace in enumerate(['No Usable Ace', 'Usable Ace']):
        ax = axes[i]
        im = ax.imshow(V[i], origin='lower',
                       extent=[1, 10, 12, 21],
                       aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(f'{ace}')
        plt.colorbar(im, ax=ax)

    plt.suptitle('MC Policy Evaluation after 1000 episodes')
    plt.tight_layout()
    plt.show()