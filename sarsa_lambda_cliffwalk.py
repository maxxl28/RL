import numpy as np
import random
import matplotlib.pyplot as plt

def sarsa(gamma, alpha, lambda1):
    Q = np.zeros((4, 12, 4)) 
    for z in range(1, 1000):
        E = np.zeros((4, 12, 4))
        state_pos = (3,0)
        action = random.randint(0, 3)
        epsilon = 1
        for x in range(1, 1000):
            epsilon = 1/z
            new_state_pos, reward = take_action(state_pos, action)
            
            weights = [1-epsilon, epsilon]
            choice = np.random.choice([0, 1], p=weights)
            if choice == 0:
                action_next = np.argmax(Q[state_pos[0], state_pos[1], :])
            else:
                action_next = np.random.randint(0, 4)

            delta = reward + gamma*Q[new_state_pos[0], new_state_pos[1], action_next] - Q[state_pos[0], state_pos[1], action]
            E[state_pos[0], state_pos[1], action] += 1
            Q += alpha*delta*E
            E *= gamma*lambda1
            state_pos = new_state_pos
            action = action_next
            if state_pos == (3, 11):
                break
    return Q
        
            
# Below code is from Claude
def take_action(state, action):
    row, col = state
    
    if action == 0:  # up
        new_row, new_col = row - 1, col
    elif action == 1:  # down
        new_row, new_col = row + 1, col
    elif action == 2:  # left
        new_row, new_col = row, col - 1
    elif action == 3:  # right
        new_row, new_col = row, col + 1
    
    # clip to grid boundaries — stay in place if out of bounds
    new_row = np.clip(new_row, 0, 3)
    new_col = np.clip(new_col, 0, 11)
    
    new_state = (new_row, new_col)
    
    # check cliff
    if new_row == 3 and 1 <= new_col <= 10:
        return (3, 0), -100  # back to start
    
    # check goal
    if new_state == (3, 11):
        return new_state, -1  # terminal
    
    return new_state, -1  # normal step


if __name__ == "__main__":
    Q = sarsa(gamma=1, alpha=0.1, lambda1=0.9)
    
    # extract greedy policy
    policy = np.argmax(Q, axis=2)  # shape (4, 12)
    
    # simulate greedy path from start
    state = (3, 0)
    path = [state]
    for _ in range(1000):  # max steps to avoid infinite loop
        action = policy[state[0], state[1]]
        state, _ = take_action(state, action)
        path.append(state)
        if state == (3, 11):
            break
    
    # draw grid
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # draw cells
    for row in range(4):
        for col in range(12):
            # color cliff red
            if row == 3 and 1 <= col <= 10:
                color = '#E84040'
            else:
                color = '#f0f0f0'
            rect = plt.Rectangle((col, 3-row), 1, 1,
                                facecolor=color, edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
    
    # label start, goal, cliff
    ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(11.5, 0.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(5.5, 0.5, 'CLIFF', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # draw path
    path_cols = [p[1] + 0.5 for p in path]
    path_rows = [3 - p[0] + 0.5 for p in path]
    ax.plot(path_cols, path_rows, 'b-', linewidth=2, zorder=5)
    ax.plot(path_cols[0], path_rows[0], 'go', markersize=10, zorder=6)   # start
    ax.plot(path_cols[-1], path_rows[-1], 'b*', markersize=15, zorder=6)  # end
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(np.arange(12))
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(np.arange(3, -1, -1))
    ax.set_title('Sarsa(λ) - Learned Path')
    
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()