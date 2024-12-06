import numpy as np
import pandas as pd
import warnings
import random
import ast
import utilty


np.random.seed(42)
random.seed(42)


def load_dataset():
    dataframe = pd.read_csv("mixed_nash_equilibrium_dataset.csv")
    dataframe["Player 1 Strategy"] = dataframe["Player 1 Strategy"].apply(ast.literal_eval)
    dataframe["Player 2 Strategy"] = dataframe["Player 2 Strategy"].apply(ast.literal_eval)
    dataframe["Payoff Matrix P1"] = dataframe["Payoff Matrix P1"].apply(ast.literal_eval)
    dataframe["Payoff Matrix P2"] = dataframe["Payoff Matrix P2"].apply(ast.literal_eval)
    return dataframe


def calculate_expected_payoff(payoff_matrix, player_1_strategy, player_2_strategy):
    return np.sum(np.multiply(payoff_matrix, np.outer(player_1_strategy, player_2_strategy)))


def nash_eq_one_two_approx(payoff_matrix_p1, payoff_matrix_p2):
    payoff_matrix_p1 = np.array(payoff_matrix_p1)
    payoff_matrix_p2 = np.array(payoff_matrix_p2)
    
    # find the response for player 1 randomly not the best response
    player_1_first_strategy = np.random.randint(0, payoff_matrix_p1.shape[0])
    # find the best response for player 2
    for i in range(payoff_matrix_p2.shape[1]):
        if payoff_matrix_p2[player_1_first_strategy][i] == np.max(payoff_matrix_p2[player_1_first_strategy]):
            player_2_first_strategy = i
            break
    
    # find the best response for player 1
    for i in range(payoff_matrix_p1.shape[0]):
        if payoff_matrix_p1[i][player_2_first_strategy] == np.max(payoff_matrix_p1[:, player_2_first_strategy]):
            player_1_second_strategy = i
            break
    
    if player_1_first_strategy != player_1_second_strategy:
        player_1_pure_strategy_index = [player_1_first_strategy, player_1_second_strategy]
    
    else:
        player_1_pure_strategy_index = [player_1_first_strategy]
    
    player_2_strategy = np.zeros(payoff_matrix_p2.shape[1])
    player_2_strategy[player_2_first_strategy] = 1

    player_1_strategy = np.zeros(payoff_matrix_p1.shape[0])
    player_1_strategy[player_1_pure_strategy_index] = 1/len(player_1_pure_strategy_index)

    return player_1_strategy, player_2_strategy


def round_to_6_decimal_places(value):
    return round(value, 6)


df = load_dataset()

with open("nash_results_1_2.csv", 'w') as f:
    f.write("Game ID,Expected payoff of player1 from approx,Expected payoff of player1 real,epsilon player1,Expected payoff of player2 from approx,Expected payoff of player2 real,epsilon player2,player_1_strategy,player_2_strategy\n")

    for i in range(df.shape[0]):
        print(f"Processing Game ID: {df['Game ID'].iloc[i]}")
        player_1_strategy, player_2_strategy = nash_eq_one_two_approx(df["Payoff Matrix P1"].iloc[i], df["Payoff Matrix P2"].iloc[i])

        player_1_matrix = np.dot(df["Payoff Matrix P1"].iloc[i], player_2_strategy)

        combinations = utilty.generate_combinations(len(player_1_strategy))
        max_payoff_player_1 = -1
        for combination in combinations:
            strategy = utilty.solve_equation(combination, player_1_matrix)
            if calculate_expected_payoff(df["Payoff Matrix P1"].iloc[i], strategy, player_2_strategy) > max_payoff_player_1:
                max_payoff_player_1 = calculate_expected_payoff(df["Payoff Matrix P1"].iloc[i], strategy, player_2_strategy)
        
        player_2_matrix = np.dot(np.array(df["Payoff Matrix P2"].iloc[i]).T, player_1_strategy)

        combinations = utilty.generate_combinations(len(player_2_strategy))
        max_payoff_player_2 = -1
        for combination in combinations:
            strategy = utilty.solve_equation(combination, player_2_matrix)
            if calculate_expected_payoff(df["Payoff Matrix P2"].iloc[i], player_1_strategy, strategy) > max_payoff_player_2:
                max_payoff_player_2 = calculate_expected_payoff(df["Payoff Matrix P2"].iloc[i], player_1_strategy, strategy)
            
        expected_payoff_p1_approx = calculate_expected_payoff(df["Payoff Matrix P1"].iloc[i], player_1_strategy, player_2_strategy)
        expected_payoff_p1_real = max_payoff_player_1
        expected_payoff_p2_approx = calculate_expected_payoff(df["Payoff Matrix P2"].iloc[i], player_1_strategy, player_2_strategy)
        expected_payoff_p2_real = max_payoff_player_2
        epsilon_p1 =  expected_payoff_p1_real - expected_payoff_p1_approx
        epsilon_p2 =  expected_payoff_p2_real - expected_payoff_p2_approx

        expected_payoff_p1_approx = round_to_6_decimal_places(expected_payoff_p1_approx)
        expected_payoff_p1_real = round_to_6_decimal_places(expected_payoff_p1_real)
        epsilon_p1 = round_to_6_decimal_places(epsilon_p1)
        expected_payoff_p2_approx = round_to_6_decimal_places(expected_payoff_p2_approx)
        expected_payoff_p2_real = round_to_6_decimal_places(expected_payoff_p2_real)
        epsilon_p2 = round_to_6_decimal_places(epsilon_p2)

        if epsilon_p1>0.75 or epsilon_p2>0.75:
            print(f"Game ID: {df['Game ID'].iloc[i]} performed contradictorily")

        f.write(f"{df['Game ID'].iloc[i]},{expected_payoff_p1_approx},{expected_payoff_p1_real},{epsilon_p1},{expected_payoff_p2_approx},{expected_payoff_p2_real},{epsilon_p2},\"{player_1_strategy.tolist()}\",\"{player_2_strategy.tolist()}\"\n")
