import numpy as np
import nashpy as nash
import pandas as pd
import warnings
import random


np.random.seed(42)
random.seed(42)


def normalize_payoffs(payoff_matrix):
    min_val = payoff_matrix.min()  # Find the minimum value
    shifted_matrix = payoff_matrix - min_val  # Shift to make all values positive
    normalized_matrix = shifted_matrix / shifted_matrix.max()  # Scale to [0, 1]
    return normalized_matrix


def round_to_6_decimal_places(matrix):
    return np.round(matrix, 6)


def generate_benchmark_dataset(num_actions_player1, num_actions_player2, game_id=0):
    # Catch RuntimeWarnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)  # Catch all RuntimeWarnings

        # Generate random payoff matrices for Player 1 and Player 2
        payoff_player1 = np.random.randint(-10, 10, (num_actions_player1, num_actions_player2))
        payoff_player2 = np.random.randint(-10, 10, (num_actions_player1, num_actions_player2))

        # Normalize the payoff matrices
        payoff_player1 = normalize_payoffs(payoff_player1)
        payoff_player2 = normalize_payoffs(payoff_player2)

        # Create a NashPy game

        game = nash.Game(payoff_player1, payoff_player2)

        # Compute equilibria
        equilibria = list(game.support_enumeration())

        # Check for warnings
        for w in caught_warnings:
            if issubclass(w.category, RuntimeWarning):
                raise RuntimeError(str(w.message))


    # Format the equilibria for storage
    formatted_equilibria = [
        {  
            "Game ID": game_id+i,
            "Player 1 Strategy": round_to_6_decimal_places(equilibria[i][0]).tolist(),
            "Player 2 Strategy": round_to_6_decimal_places(equilibria[i][1]).tolist(),
            "Payoff Matrix P1": round_to_6_decimal_places(payoff_player1).tolist(),
            "Payoff Matrix P2": round_to_6_decimal_places(payoff_player2).tolist(),
        }
        for i in range(len(equilibria))
    ]

    # Convert to DataFrame for easier analysis
    return pd.DataFrame(formatted_equilibria)


global_dataset = pd.DataFrame()
# Parameters for dataset generation
for i in range(10000):
    while True:
        try:
            # Number of actions for Player 1
            num_actions_player1 = np.random.randint(2, 10)
            # Number of actions for Player 2
            num_actions_player2 = np.random.randint(2, 10)
            # Generate the dataset
            dataset = generate_benchmark_dataset(num_actions_player1, num_actions_player2, len(global_dataset)+1)
            print(num_actions_player1, num_actions_player2)
            break
        except RuntimeError as e:
            print("Error:", e)
            continue
    
    global_dataset = pd.concat([global_dataset, dataset], axis=0)

    print("Dataset size:", global_dataset.shape)
    print("Iteration:", i+1)

# Save to a CSV file
global_dataset.to_csv("mixed_nash_equilibrium_dataset_nashpy.csv", index=False)
