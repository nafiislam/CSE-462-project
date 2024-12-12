import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import random
import ast
import utilty


np.random.seed(42)
random.seed(42)


def load_dataset_1():
    dataframe = pd.read_csv("nash_results.csv")
    dataframe["player_1_strategy"] = dataframe["player_1_strategy"].apply(ast.literal_eval)
    dataframe["player_2_strategy"] = dataframe["player_2_strategy"].apply(ast.literal_eval)
    return dataframe

def load_dataset_2():
    dataframe = pd.read_csv("nash_results_1_2.csv")
    dataframe["player_1_strategy"] = dataframe["player_1_strategy"].apply(ast.literal_eval)
    dataframe["player_2_strategy"] = dataframe["player_2_strategy"].apply(ast.literal_eval)
    return dataframe

df = load_dataset_1()

epsilon_player_one = df["epsilon player1"].values
epsilon_player_two = df["epsilon player2"].values

strategy_counts = df["player_1_strategy"].apply(lambda x: len(x)).values
strategy_counts_2 = df["player_2_strategy"].apply(lambda x: len(x)).values

plt.scatter(strategy_counts, epsilon_player_one)
plt.xlabel("Number of strategies")
plt.ylabel("Epsilon")
plt.title("Epsilon vs Number of strategies for player 1")
plt.show()

plt.scatter(strategy_counts_2, epsilon_player_two)
plt.xlabel("Number of strategies")
plt.ylabel("Epsilon")
plt.title("Epsilon vs Number of strategies for player 2")
plt.show()

time = df["time"].values
plt.scatter(strategy_counts, time)
plt.xlabel("Number of strategies")
plt.ylabel("Time (sec)")
plt.title("Time vs Number of strategies for player 1")
plt.tight_layout()
plt.show()

plt.scatter(strategy_counts_2, time)
plt.xlabel("Number of strategies")
plt.ylabel("Time (sec)")
plt.title("Time vs Number of strategies for player 2")
plt.tight_layout()
plt.show()

between_0_0_0_1 = 0
between_0_1_0_2 = 0
between_0_2_0_3 = 0
between_0_3_0_4 = 0
between_0_4_0_5 = 0
between_0_5_0_6 = 0
between_0_6_0_7 = 0
between_0_7_0_8 = 0

for i in range(len(epsilon_player_one)):
    if 0.0 <= epsilon_player_one[i] < 0.1:
        between_0_0_0_1 += 1
    elif 0.1 <= epsilon_player_one[i] < 0.2:
        between_0_1_0_2 += 1
    elif 0.2 <= epsilon_player_one[i] < 0.3:
        between_0_2_0_3 += 1
    elif 0.3 <= epsilon_player_one[i] < 0.4:
        between_0_3_0_4 += 1
    elif 0.4 <= epsilon_player_one[i] < 0.5:
        between_0_4_0_5 += 1
    elif 0.5 <= epsilon_player_one[i] < 0.6:
        between_0_5_0_6 += 1
    elif 0.6 <= epsilon_player_one[i] < 0.7:
        between_0_6_0_7 += 1
    elif 0.7 <= epsilon_player_one[i] < 0.75:
        between_0_7_0_8 += 1

epsilon_ranges = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.75"]
game_counts = [between_0_0_0_1, between_0_1_0_2, between_0_2_0_3, between_0_3_0_4, between_0_4_0_5, between_0_5_0_6, between_0_6_0_7, between_0_7_0_8]

plt.bar(epsilon_ranges, game_counts)
plt.xlabel("Epsilon ranges")
plt.ylabel("Game counts")
plt.title("Game counts vs Epsilon ranges for player 1")
plt.tight_layout()
plt.show()

between_0_0_0_1 = 0
between_0_1_0_2 = 0
between_0_2_0_3 = 0
between_0_3_0_4 = 0
between_0_4_0_5 = 0
between_0_5_0_6 = 0
between_0_6_0_7 = 0
between_0_7_0_8 = 0

for i in range(len(epsilon_player_two)):
    if 0.0 <= epsilon_player_two[i] < 0.1:
        between_0_0_0_1 += 1
    elif 0.1 <= epsilon_player_two[i] < 0.2:
        between_0_1_0_2 += 1
    elif 0.2 <= epsilon_player_two[i] < 0.3:
        between_0_2_0_3 += 1
    elif 0.3 <= epsilon_player_two[i] < 0.4:
        between_0_3_0_4 += 1
    elif 0.4 <= epsilon_player_two[i] < 0.5:
        between_0_4_0_5 += 1
    elif 0.5 <= epsilon_player_two[i] < 0.6:
        between_0_5_0_6 += 1
    elif 0.6 <= epsilon_player_two[i] < 0.7:
        between_0_6_0_7 += 1
    elif 0.7 <= epsilon_player_two[i] < 0.75:
        between_0_7_0_8 += 1

epsilon_ranges = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.75"]
game_counts = [between_0_0_0_1, between_0_1_0_2, between_0_2_0_3, between_0_3_0_4, between_0_4_0_5, between_0_5_0_6, between_0_6_0_7, between_0_7_0_8]

plt.bar(epsilon_ranges, game_counts)
plt.xlabel("Epsilon ranges")
plt.ylabel("Game counts")
plt.title("Game counts vs Epsilon ranges for player 2")
plt.tight_layout()
plt.show()

df = load_dataset_2()

time_2 = df["time"].values

plt.scatter(strategy_counts, time, label="3/4-approximation")
plt.scatter(strategy_counts, time_2, label="1/2-approximation")
plt.xlabel("Number of strategies")
plt.ylabel("Time (sec)")
plt.title("Time vs Number of strategies for player 1")
plt.legend()
plt.tight_layout()
plt.show()

plt.scatter(strategy_counts_2, time, label="3/4-approximation")
plt.scatter(strategy_counts_2, time_2, label="1/2-approximation")
plt.xlabel("Number of strategies")
plt.ylabel("Time (sec)")
plt.title("Time vs Number of strategies for player 2")
plt.legend()
plt.tight_layout()
plt.show()