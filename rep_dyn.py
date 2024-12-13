# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: research
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)

# %%
def replicator_dynamics(A, B, x_init, y_init, max_iters=1000, learning_rate=0.01):
    """
    Solve a 2-player Nash equilibrium using replicator dynamics.

    Parameters:
    - A: Payoff matrix for player 1
    - B: Payoff matrix for player 2
    - x_init: Initial strategy for player 1 (numpy array)
    - y_init: Initial strategy for player 2 (numpy array)
    - iterations: Number of iterations
    - learning_rate: Step size for updates

    Returns:
    - x: Final strategy for player 1
    - y: Final strategy for player 2
    """
    
    x = x_init
    y = y_init
    
    for _ in range(max_iters):
        # Compute expected payoffs
        payoff_x = np.dot(A, y)
        payoff_y = np.dot(x, B)
        
        # Update strategies
        x = x + learning_rate * x * (payoff_x - np.dot(x, payoff_x))
        y = y + learning_rate * y * (payoff_y - np.dot(y, payoff_y))
        
        # Normalize strategies
        x = x / np.sum(x)
        y = y / np.sum(y)
        
    return x, y

# %%
DATASET_FILE = "mixed_nash_equilibrium_dataset.csv"
data = pd.read_csv(DATASET_FILE)

# %%
OUTPUT_FILE = "replicator_dynamics_results.csv"
NUM_GAMES = len(data)

game_ids = []
exp_payoffs_p1 = []
exp_payoffs_p2 = []
strategy_p1_list = []
strategy_p2_list = []
epsilon_1_list = []
epsilon_2_list = []

def process_game(i):
    game_id = data['Game ID'][i]
    payoff_p1 = np.array(eval(data['Payoff Matrix P1'][i]))
    payoff_p2 = np.array(eval(data['Payoff Matrix P2'][i]))
    
    p1_strategy = np.random.rand(payoff_p1.shape[0])
    p2_strategy = np.random.rand(payoff_p1.shape[1])
    p1_strategy = p1_strategy / np.sum(p1_strategy)
    p2_strategy = p2_strategy / np.sum(p2_strategy)

    x, y = replicator_dynamics(payoff_p1, payoff_p2, p1_strategy, p2_strategy)
    p1_strategy = x
    p2_strategy = y
    exp_payoff_p1 = np.dot(x, np.dot(payoff_p1, y))
    exp_payoff_p2 = np.dot(x, np.dot(payoff_p2, y))

    player_1_matrix = np.dot(payoff_p1, p2_strategy)
    max_payoff_player_1 = player_1_matrix.max()
    player_2_matrix = np.dot(payoff_p2.T, p1_strategy)
    max_payoff_player_2 = player_2_matrix.max()

    epsilon_1 = max_payoff_player_1 - exp_payoff_p1
    epsilon_2 = max_payoff_player_2 - exp_payoff_p2

    return game_id, exp_payoff_p1, exp_payoff_p2, p1_strategy, p2_strategy, epsilon_1, epsilon_2

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_game, range(NUM_GAMES)), total=NUM_GAMES))

for game_id, exp_payoff_p1, exp_payoff_p2, p1_strategy, p2_strategy, epsilon_1, epsilon_2 in results:
    game_ids.append(game_id)
    exp_payoffs_p1.append(exp_payoff_p1)
    exp_payoffs_p2.append(exp_payoff_p2)
    strategy_p1_list.append(p1_strategy)
    strategy_p2_list.append(p2_strategy)
    epsilon_1_list.append(epsilon_1)
    epsilon_2_list.append(epsilon_2)
    
with open(OUTPUT_FILE, "w") as f:
    f.write("game_id,strategy_p1,strategy_p2,exp_payoff_p1,exp_payoff_p2,epsilon_1,epsilon_2\n")
    for i in range(NUM_GAMES):
        # Convert strategies to properly formatted strings
        strategy_p1 = ",".join(map(str, strategy_p1_list[i]))
        strategy_p2 = ",".join(map(str, strategy_p2_list[i]))
        # Write the formatted row
        f.write(
            f"{game_ids[i]},\"[{strategy_p1}]\",\"[{strategy_p2}]\","
            f"{exp_payoffs_p1[i]},{exp_payoffs_p2[i]},{epsilon_1_list[i]},{epsilon_2_list[i]}\n"
        )
