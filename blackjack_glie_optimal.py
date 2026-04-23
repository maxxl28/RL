import numpy as np
import matplotlib.pyplot as plt
import random

def optimize_glie():
    # 0 for twist, 1 for stick
    N = np.zeros((2, 10, 10, 2)) # useable ace, player_sum, dealer_card, action
    Q = np.zeros((2, 10, 10, 2)) # useable ace, player_sum, dealer_card, action
    for episode in range(1, 1000000):
        epsilon = 1/episode
        
        shape = (2, 10, 10) # 200 total elements
        options = ["greedy", "random"]
        # only 2 options in blackjack
        weights = [1-epsilon+epsilon/2, epsilon/2]
        flat_samples = np.random.choice(options, size=np.prod(shape), p=weights)
        policy = flat_samples.reshape(shape)
        states_visited, goal = play_episode(policy, Q)
        for state in states_visited:
            N[state[0], state[1], state[2], state[3]] += 1
            Q[state[0], state[1], state[2], state[3]] += 1/N[state[0], state[1], state[2], state[3]]*(goal-Q[state[0], state[1], state[2], state[3]])
    return Q
        


def play_episode(policy, value_func):
    '''
    used LLMs to help debug
    '''
    # initialize state randomly
    player_sum = np.random.randint(12, 22) # 12-21
    dealer_card = np.random.randint(1, 11) # ace(1) to 10
    usable_ace = np.random.randint(0, 2) # 0 or 1

    states_visited = []
    while True:
        if policy[usable_ace, player_sum - 12, dealer_card - 1] == "random":
            if value_func[usable_ace, player_sum - 12, dealer_card - 1, 0] > value_func[usable_ace, player_sum - 12, dealer_card - 1, 1]:
                action = 1
            elif value_func[usable_ace, player_sum - 12, dealer_card - 1, 0] == value_func[usable_ace, player_sum - 12, dealer_card - 1, 1]:
                action = np.random.randint(0, 2)
            else:
                action = 0


        else:
            if value_func[usable_ace, player_sum - 12, dealer_card - 1, 0] < value_func[usable_ace, player_sum - 12, dealer_card - 1, 1]:
                action = 1
            else:
                action = 0
        
        state = (usable_ace, player_sum - 12, dealer_card - 1, action)
        states_visited.append(state)

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
            



if __name__ == "__main__":
    Q = optimize_glie()
    
    policy_optimal = np.argmax(Q, axis=3)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, ace in enumerate(['No Usable Ace', 'Usable Ace']):
        ax = axes[i]
        
        # plot each cell manually as a colored rectangle
        for p_idx in range(10):        # player sum 12-21
            for d_idx in range(10):    # dealer card 1-10
                action = policy_optimal[i, p_idx, d_idx]
                color = '#4A90D9' if action == 1 else '#E84040'  # blue=stick, red=twist
                rect = plt.Rectangle((d_idx, p_idx), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=0.8)
                ax.add_patch(rect)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks(np.arange(10) + 0.5)
        ax.set_xticklabels(np.arange(1, 11))
        ax.set_yticks(np.arange(10) + 0.5)
        ax.set_yticklabels(np.arange(12, 22))
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(f'Optimal Policy - {ace}')
        
        # manual legend
        from matplotlib.patches import Patch
        legend = [Patch(facecolor='#4A90D9', label='Stick'),
                  Patch(facecolor='#E84040', label='Twist')]
        ax.legend(handles=legend, loc='upper right')
    
    plt.suptitle('GLIE Monte Carlo Control - Optimal Policy', fontsize=14)
    plt.tight_layout()
    plt.show()