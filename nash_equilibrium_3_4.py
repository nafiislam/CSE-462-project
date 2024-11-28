import numpy as np
import pandas as pd
import warnings
import random
import ast


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


df = load_dataset()
# print(df["Player 1 Strategy"].iloc[0])
# print(df["Player 2 Strategy"].iloc[0])
# print(df["Payoff Matrix P1"].iloc[0])
# print(df["Payoff Matrix P2"].iloc[0])

# player_1_strategy, player_2_strategy = nash_eq_three_four_approx(df["Payoff Matrix P1"].iloc[0], df["Payoff Matrix P2"].iloc[0])
# print(player_1_strategy)
# print(player_2_strategy)
# print(calculate_expected_payoff(df["Payoff Matrix P1"].iloc[0], player_1_strategy, player_2_strategy))
# print(calculate_expected_payoff(df["Payoff Matrix P1"].iloc[0], df["Player 1 Strategy"].iloc[0], df["Player 2 Strategy"].iloc[0]))
# print(calculate_expected_payoff(df["Payoff Matrix P2"].iloc[0], player_1_strategy, player_2_strategy))
# print(calculate_expected_payoff(df["Payoff Matrix P2"].iloc[0], df["Player 1 Strategy"].iloc[0], df["Player 2 Strategy"].iloc[0]))

with open("nash_results.csv", 'w') as f:
    f.write("Game ID, Expected payoff of player1 from approx, Expected payoff of player1 real, epsilon player1, Expected payoff of player2 from approx, Expected payoff of player2 real, epsilon player2, player_1_strategy, player_2_strategy\n")

    for i in range(df.shape[0]):
        player_1_strategy, player_2_strategy = nash_eq_three_four_approx(df["Payoff Matrix P1"].iloc[i], df["Payoff Matrix P2"].iloc[i])
        expected_payoff_p1_approx = calculate_expected_payoff(df["Payoff Matrix P1"].iloc[i], player_1_strategy, player_2_strategy)
        expected_payoff_p1_real = calculate_expected_payoff(df["Payoff Matrix P1"].iloc[i], df["Player 1 Strategy"].iloc[i], df["Player 2 Strategy"].iloc[i])
        expected_payoff_p2_approx = calculate_expected_payoff(df["Payoff Matrix P2"].iloc[i], player_1_strategy, player_2_strategy)
        expected_payoff_p2_real = calculate_expected_payoff(df["Payoff Matrix P2"].iloc[i], df["Player 1 Strategy"].iloc[i], df["Player 2 Strategy"].iloc[i])
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

        f.write(f"{df['Game ID'].iloc[i]}, {expected_payoff_p1_approx}, {expected_payoff_p1_real}, {epsilon_p1}, {expected_payoff_p2_approx}, {expected_payoff_p2_real}, {epsilon_p2}, \"{player_1_strategy.tolist()}\", \"{player_2_strategy.tolist()}\"\n")
