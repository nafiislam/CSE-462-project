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
RESULTS_FILE = "nash_results.csv"
data = pd.read_csv(DATASET_FILE)
result = pd.read_csv(RESULTS_FILE)

assert len(data) == len(result)

# %%
OUTPUT_FILE = "replicator_dynamics_results.csv"

exp_payoffs_p1 = []
exp_payoffs_p2 = []
best_exp_payoffs_p1 = []
best_exp_payoffs_p2 = []

def process_game(i):
    game_id = data['Game ID'][i]
    p1_strategy = np.array(eval(data['Player 1 Strategy'][i]))
    p2_strategy = np.array(eval(data['Player 2 Strategy'][i]))
    payoff_p1 = np.array(eval(data['Payoff Matrix P1'][i]))
    payoff_p2 = np.array(eval(data['Payoff Matrix P2'][i]))

    x, y = replicator_dynamics(payoff_p1, payoff_p2, p1_strategy, p2_strategy)
    exp_payoff_p1 = np.dot(x, np.dot(payoff_p1, y))
    exp_payoff_p2 = np.dot(x, np.dot(payoff_p2, y))

    best_exp_payoff_p1 = result[result['Game ID'] == game_id]['Expected payoff of player1 real'].values[0]
    best_exp_payoff_p2 = result[result['Game ID'] == game_id]['Expected payoff of player2 real'].values[0]

    return exp_payoff_p1, exp_payoff_p2, best_exp_payoff_p1, best_exp_payoff_p2

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_game, range(len(data))), total=len(data)))

for exp_payoff_p1, exp_payoff_p2, best_exp_payoff_p1, best_exp_payoff_p2 in results:
    exp_payoffs_p1.append(exp_payoff_p1)
    exp_payoffs_p2.append(exp_payoff_p2)
    best_exp_payoffs_p1.append(best_exp_payoff_p1)
    best_exp_payoffs_p2.append(best_exp_payoff_p2)
    
with open(OUTPUT_FILE, "w") as f:
    f.write("RD Payoff P1, RD Payoff P2, Best Payoff P1, Best Payoff P2\n")
    for i in range(len(data)):
        f.write(f"{exp_payoffs_p1[i]},{exp_payoffs_p2[i]},{best_exp_payoffs_p1[i]},{best_exp_payoffs_p2[i]}\n")
