import numpy as np
import pandas as pd
import warnings
import random
import ast
import time
import sys
from tqdm import tqdm
import gc
import io

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


def nash_eq_three_four_approx(payoff_matrix_p1, payoff_matrix_p2):
    payoff_matrix_p1 = np.array(payoff_matrix_p1)
    payoff_matrix_p2 = np.array(payoff_matrix_p2)
    # find the best response for player 1
    best_response_p1 = np.unravel_index(np.argmax(payoff_matrix_p1), payoff_matrix_p1.shape)
    # find the best response for player 2
    best_response_p2 = np.unravel_index(np.argmax(payoff_matrix_p2), payoff_matrix_p2.shape)
    # print(best_response_p1)
    # print(best_response_p2)
    
    if best_response_p1[0] != best_response_p2[0]:
        player_1_pure_strategy_index = [best_response_p1[0], best_response_p2[0]]
    else:
        player_1_pure_strategy_index = [best_response_p1[0]]
    
    if best_response_p1[1] != best_response_p2[1]:
        player_2_pure_strategy_index = [best_response_p1[1], best_response_p2[1]]
    else:
        player_2_pure_strategy_index = [best_response_p1[1]]
    
    # player 1's strategy
    player_1_strategy = np.zeros(payoff_matrix_p1.shape[0])
    player_1_strategy[player_1_pure_strategy_index] = 1/len(player_1_pure_strategy_index)

    # player 2's strategy
    player_2_strategy = np.zeros(payoff_matrix_p2.shape[1])
    player_2_strategy[player_2_pure_strategy_index] = 1/len(player_2_pure_strategy_index)

    return player_1_strategy, player_2_strategy


def round_to_6_decimal_places(value):
    return round(value, 6)



if __name__ == "__main__":    
    if len(sys.argv) != 3:
        print("Usage: python nash_equilibrium_3_4.py <dataset_path> <output_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2]

    # Create an empty DataFrame to store results temporarily
    result_columns = [
        "Game_ID", "Expected_payoff_of_player1_from_approx", "epsilon_player1",
        "Expected_payoff_of_player2_from_approx", "epsilon_player2",
        "player_1_strategy", "player_2_strategy", "time"
    ]
    results_df = pd.DataFrame(columns=result_columns)

    # Open dataset and process row by row
    with open(dataset_path, 'r') as csv_file:
        total_rows = sum(1 for _ in csv_file) - 1  # Count rows excluding header

    with open(dataset_path, 'r') as csv_file:
        # Skip header
        header = csv_file.readline()

        # Process each row with a progress bar
        for line in tqdm(csv_file, total=total_rows, desc="Processing Games"):
            # Load a single row
            row = pd.read_csv(io.StringIO(header + line), converters={
                "Payoff Matrix P1": ast.literal_eval,
                "Payoff Matrix P2": ast.literal_eval
            })


            game_id = row['Game ID'].iloc[0]
            payoff_matrix_p1 = row["Payoff Matrix P1"].iloc[0]
            payoff_matrix_p2 = row["Payoff Matrix P2"].iloc[0]

            # Process Nash equilibrium approximation
            start_time = time.time()
            player_1_strategy, player_2_strategy = nash_eq_three_four_approx(payoff_matrix_p1, payoff_matrix_p2)
            time_taken = time.time() - start_time

            player_1_matrix = np.dot(payoff_matrix_p1, player_2_strategy)
            max_payoff_player_1 = player_1_matrix.max()

            player_2_matrix = np.dot(np.array(payoff_matrix_p2).T, player_1_strategy)
            max_payoff_player_2 = player_2_matrix.max()

            expected_payoff_p1_approx = calculate_expected_payoff(payoff_matrix_p1, player_1_strategy, player_2_strategy)
            expected_payoff_p2_approx = calculate_expected_payoff(payoff_matrix_p2, player_1_strategy, player_2_strategy)
            epsilon_p1 = max_payoff_player_1 - expected_payoff_p1_approx
            epsilon_p2 = max_payoff_player_2 - expected_payoff_p2_approx

            expected_payoff_p1_approx = round_to_6_decimal_places(expected_payoff_p1_approx)
            epsilon_p1 = round_to_6_decimal_places(epsilon_p1)
            expected_payoff_p2_approx = round_to_6_decimal_places(expected_payoff_p2_approx)
            epsilon_p2 = round_to_6_decimal_places(epsilon_p2)

            if epsilon_p1 > 0.75 or epsilon_p2 > 0.75:
                print(f"Game ID: {game_id} performed contradictorily")

            # Append the result to the DataFrame
            results_df = pd.DataFrame([{
                "Game_ID": game_id,
                "Expected_payoff_of_player1_from_approx": expected_payoff_p1_approx,
                "epsilon_player1": epsilon_p1,
                "Expected_payoff_of_player2_from_approx": expected_payoff_p2_approx,
                "epsilon_player2": epsilon_p2,
                "player_1_strategy": player_1_strategy.tolist(),
                "player_2_strategy": player_2_strategy.tolist(),
                "time": time_taken
            }])

            # Write to CSV immediately
            results_df.to_csv(output_path, mode='a', header=not pd.io.common.file_exists(output_path), index=False)

            # Free memory explicitly
            del row, payoff_matrix_p1, payoff_matrix_p2, player_1_strategy, player_2_strategy
            del player_1_matrix, player_2_matrix, expected_payoff_p1_approx, expected_payoff_p2_approx
            del epsilon_p1, epsilon_p2