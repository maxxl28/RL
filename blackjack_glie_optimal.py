import numpy as np
import matplotlib.pyplot as plt
import random

def optimize_glie():
    # 0 for twist, 1 for stick
    N = np.zeros((2, 10, 10, 2)) # useable ace, player_sum, dealer_card, action
    Q = np.zeros((2, 10, 10, 2)) # useable ace, player_sum, dealer_card, action
    for episode in range(1, 2000000):
        epsilon = 1/episode
        
        shape = (2, 10, 10) # 200 total elements
        options = ["greedy", "random"]
        # only 2 options in blackjack
        weights = [1-epsilon+epsilon/2, epsilon/2]
        flat_samples = np.random.choice(options, size=np.prod(shape), p=weights)

        # fix: do not calculate the policy all at once
 
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
    # From Claude
    Q = optimize_glie()
    
    policy_optimal = np.argmax(Q, axis=3)  # 0=twist, 1=stick

    # Game-theoretic optimal policy (basic strategy, hit/stand only, no splits/doubles)
    # 0 = twist/hit, 1 = stick/stand
    # Axes: [usable_ace, player_sum_idx (0=12,...,9=21), dealer_card_idx (0=A,...,9=10)]
    game_theoretic = np.zeros((2, 10, 10), dtype=int)

    # --- No usable ace (hard totals: 12-21) ---
    for p_idx in range(10):
        player_sum = p_idx + 12
        for d_idx in range(10):
            dealer_card = d_idx + 1  # 1=Ace, 2-10
            if player_sum >= 17:
                action = 1  # stand
            elif player_sum >= 13:
                action = 1 if dealer_card in range(2, 7) else 0  # stand vs 2-6
            elif player_sum == 12:
                action = 1 if dealer_card in range(4, 7) else 0  # stand vs 4-6
            else:
                action = 0  # hit (shouldn't occur since min is 12)
            game_theoretic[0, p_idx, d_idx] = action

    # --- Usable ace (soft totals: soft 12 = A+12? No: player_sum 12-21 with usable ace)
    # With a usable ace, player_sum is the total counting ace as 11.
    # Soft 19+ -> stand; Soft 18 -> stand vs 2-8, hit vs 9,10,A; Soft 17 or less -> hit
    for p_idx in range(10):
        player_sum = p_idx + 12  # e.g. soft 12 means A+A which is unusual; soft 18 = A+7
        for d_idx in range(10):
            dealer_card = d_idx + 1
            if player_sum >= 19:
                action = 1  # stand
            elif player_sum == 18:
                # stand vs dealer 2-8, hit vs 9, 10, Ace
                action = 1 if dealer_card in range(2, 9) else 0
            else:
                action = 0  # hit
            game_theoretic[1, p_idx, d_idx] = action

    # Compare and compute accuracy
    correct = (policy_optimal == game_theoretic)
    total_states = correct.size  # 200
    pct_correct = correct.mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    titles = ['No Usable Ace', 'Usable Ace']
    for i, ace_label in enumerate(titles):
        for col, (data, plot_title) in enumerate([
            (policy_optimal[i], f'Agent Policy - {ace_label}'),
            (correct[i],        f'Correctness vs Optimal - {ace_label}')
        ]):
            ax = axes[i][col]
            for p_idx in range(10):
                for d_idx in range(10):
                    if col == 0:
                        # Original policy plot: blue=stick, red=twist
                        action = data[p_idx, d_idx]
                        color = '#4A90D9' if action == 1 else '#E84040'
                    else:
                        # Correctness plot: green=correct, red=wrong
                        color = '#5DBB63' if data[p_idx, d_idx] else '#E84040'
                    rect = plt.Rectangle((d_idx, p_idx), 1, 1,
                                         facecolor=color, edgecolor='black', linewidth=0.8)
                    ax.add_patch(rect)

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xticks(np.arange(10) + 0.5)
            ax.set_xticklabels(['A'] + list(range(2, 11)))
            ax.set_yticks(np.arange(10) + 0.5)
            ax.set_yticklabels(range(12, 22))
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(plot_title)

            from matplotlib.patches import Patch
            if col == 0:
                legend = [Patch(facecolor='#4A90D9', label='Stick'),
                          Patch(facecolor='#E84040', label='Twist')]
            else:
                legend = [Patch(facecolor='#5DBB63', label='Correct'),
                          Patch(facecolor='#E84040', label='Wrong')]
            ax.legend(handles=legend, loc='upper right')

    plt.suptitle(
        f'GLIE Monte Carlo Control — Agent vs. Game-Theoretic Optimal\n'
        f'Accuracy: {pct_correct:.1f}% ({int(correct.sum())}/{total_states} states correct)',
        fontsize=13
    )
    plt.tight_layout()
    plt.show()