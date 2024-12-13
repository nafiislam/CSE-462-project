import numpy as np
import pandas as pd
import warnings
import random
import ast
import utilty
import time
import sys
import io
from tqdm import tqdm


np.random.seed(42)
random.seed(42)


def load_dataset(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    # dataframe["Player 1 Strategy"] = dataframe["Player 1 Strategy"].apply(ast.literal_eval)
    # dataframe["Player 2 Strategy"] = dataframe["Player 2 Strategy"].apply(ast.literal_eval)
    dataframe["Payoff Matrix P1"] = dataframe["Payoff Matrix P1"].apply(ast.literal_eval)
    dataframe["Payoff Matrix P2"] = dataframe["Payoff Matrix P2"].apply(ast.literal_eval)
    return dataframe


def calculate_expected_payoff(payoff_matrix, player_1_strategy, player_2_strategy):
    return np.sum(np.multiply(payoff_matrix, np.outer(player_1_strategy, player_2_strategy)))


def nash_eq_one_two_approx(payoff_matrix_p1, payoff_matrix_p2):
    payoff_matrix_p1 = np.asarray(payoff_matrix_p1)
    payoff_matrix_p2 = np.asarray(payoff_matrix_p2)
    
    # Random response for player 1
    player_1_first_strategy = np.random.randint(0, payoff_matrix_p1.shape[0])
    
    # Best response for player 2
    player_2_first_strategy = np.argmax(payoff_matrix_p2[player_1_first_strategy])
    
    # Best response for player 1
    player_1_second_strategy = np.argmax(payoff_matrix_p1[:, player_2_first_strategy])
    
    # Player 1's mixed strategy
    if player_1_first_strategy != player_1_second_strategy:
        player_1_strategy = np.zeros(payoff_matrix_p1.shape[0])
        player_1_strategy[player_1_first_strategy] = 0.5
        player_1_strategy[player_1_second_strategy] = 0.5
    else:
        player_1_strategy = np.zeros(payoff_matrix_p1.shape[0])
        player_1_strategy[player_1_first_strategy] = 1.0
    
    # Player 2's pure strategy
    player_2_strategy = np.zeros(payoff_matrix_p2.shape[1])
    player_2_strategy[player_2_first_strategy] = 1.0

    return player_1_strategy, player_2_strategy


def round_to_6_decimal_places(value):
    return round(value, 6)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python nash_equilibrium_1_2.py <dataset_path> <output_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2]

    # Open the dataset to count total rows for progress bar
    with open(dataset_path, 'r') as csv_file:
        total_rows = sum(1 for _ in csv_file) - 1  # Count rows excluding the header

    # Open dataset for reading
    with open(dataset_path, 'r') as csv_file:
        # Read header
        header = csv_file.readline()

        # Write the output CSV header
        with open(output_path, 'w') as output_file:
            output_file.write("Game ID,Expected payoff of player1 from approx,epsilon player1,"
                              "Expected payoff of player2 from approx,epsilon player2,"
                              "player_1_strategy,player_2_strategy,time\n")

        # Process rows with progress bar
        for line in tqdm(csv_file, total=total_rows, desc="Processing Games"):
            # Load a single row
            row = pd.read_csv(io.StringIO(header + line), converters={
                "Payoff Matrix P1": ast.literal_eval,
                "Payoff Matrix P2": ast.literal_eval
            })

            game_id = row['Game ID'].iloc[0]
            payoff_matrix_p1 = row["Payoff Matrix P1"].iloc[0]
            payoff_matrix_p2 = row["Payoff Matrix P2"].iloc[0]

            # Start timing
            start_time = time.time()

            # Compute Nash equilibrium approximation
            player_1_strategy, player_2_strategy = nash_eq_one_two_approx(payoff_matrix_p1, payoff_matrix_p2)
            time_taken = time.time() - start_time

            # Calculate maximum and expected payoffs
            player_1_matrix = np.dot(payoff_matrix_p1, player_2_strategy)
            max_payoff_player_1 = player_1_matrix.max()

            player_2_matrix = np.dot(np.array(payoff_matrix_p2).T, player_1_strategy)
            max_payoff_player_2 = player_2_matrix.max()

            expected_payoff_p1_approx = calculate_expected_payoff(payoff_matrix_p1, player_1_strategy, player_2_strategy)
            expected_payoff_p2_approx = calculate_expected_payoff(payoff_matrix_p2, player_1_strategy, player_2_strategy)
            epsilon_p1 = max_payoff_player_1 - expected_payoff_p1_approx
            epsilon_p2 = max_payoff_player_2 - expected_payoff_p2_approx

            # Round to 6 decimal places
            expected_payoff_p1_approx = round_to_6_decimal_places(expected_payoff_p1_approx)
            epsilon_p1 = round_to_6_decimal_places(epsilon_p1)
            expected_payoff_p2_approx = round_to_6_decimal_places(expected_payoff_p2_approx)
            epsilon_p2 = round_to_6_decimal_places(epsilon_p2)

            if epsilon_p1 > 0.75 or epsilon_p2 > 0.75:
                print(f"Game ID: {game_id} performed contradictorily")

            # Write result immediately to the output CSV
            with open(output_path, 'a') as output_file:
                output_file.write(f"{game_id},{expected_payoff_p1_approx},{epsilon_p1},"
                                  f"{expected_payoff_p2_approx},{epsilon_p2},"
                                  f"\"{player_1_strategy.tolist()}\",\"{player_2_strategy.tolist()}\",{time_taken}\n")

            # Free memory explicitly
            del row, payoff_matrix_p1, payoff_matrix_p2, player_1_strategy, player_2_strategy
            del player_1_matrix, player_2_matrix, expected_payoff_p1_approx, expected_payoff_p2_approx
            del epsilon_p1, epsilon_p2