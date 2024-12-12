import numpy as np
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
    # Generate random payoff matrices for Player 1 and Player 2
    payoff_player1 = np.random.randint(-10, 10, (num_actions_player1, num_actions_player2))
    payoff_player2 = np.random.randint(-10, 10, (num_actions_player1, num_actions_player2))

    # Normalize the payoff matrices
    payoff_player1 = normalize_payoffs(payoff_player1)
    payoff_player2 = normalize_payoffs(payoff_player2)

    # Format the equilibria for storage
    formatted_equilibria = [
        {  
            "Game ID": game_id,
            "Payoff Matrix P1": round_to_6_decimal_places(payoff_player1).tolist(),
            "Payoff Matrix P2": round_to_6_decimal_places(payoff_player2).tolist(),
        }
    ]

    # Convert to DataFrame for easier analysis
    return pd.DataFrame(formatted_equilibria)


# Initialize global parameters
output_file = "mixed_nash_equilibrium_dataset.csv"
header_written = False  # Track if the header has been written
cnt = 0
# Parameters for dataset generation
for i in range(2, 100+1):
    for j in range(2, 100+1):
        cnt += 1
        # Number of actions for Player 1
        num_actions_player1 = i
        # Number of actions for Player 2
        num_actions_player2 = j
        # Generate the dataset
        dataset = generate_benchmark_dataset(num_actions_player1, num_actions_player2, cnt)
        print(num_actions_player1, num_actions_player2)
        
        # Append to CSV file
        dataset.to_csv(output_file, mode='a', header=not header_written, index=False)
        header_written = True  # After first write, set header_written to True
        
        print("Dataset size:", len(dataset))
        print("Iteration:", i+1)
