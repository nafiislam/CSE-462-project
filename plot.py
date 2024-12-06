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

# plot epsilon values with respect to the number of iterations
plt.figure()
plt.plot(epsilon_player_one)
plt.xlabel("Iterations")
plt.ylabel("Epsilon")
plt.title("Epsilon values of player 1 with respect to the number of iterations")
plt.show()

plt.figure()
plt.plot(epsilon_player_two)
plt.xlabel("Iterations")
plt.ylabel("Epsilon")
plt.title("Epsilon values of player 2 with respect to the number of iterations")
plt.show()


df = load_dataset_2()

epsilon_player_one = df["epsilon player1"].values
epsilon_player_two = df["epsilon player2"].values

# plot epsilon values with respect to the number of iterations
plt.figure()
plt.plot(epsilon_player_one)
plt.xlabel("Iterations")
plt.ylabel("Epsilon")
plt.title("Epsilon values of player 1 with respect to the number of iterations")
plt.show()

plt.figure()
plt.plot(epsilon_player_two)
plt.xlabel("Iterations")
plt.ylabel("Epsilon")
plt.title("Epsilon values of player 2 with respect to the number of iterations")
plt.show()
