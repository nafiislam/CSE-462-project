import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('replicator_dynamics_results.csv')

epsilon_player_one = df["epsilon_1"].values
epsilon_player_two = df["epsilon_2"].values

strategy_counts = df['strategy_p1'].apply(lambda x: np.array(eval(x))).apply(lambda x: len(x))
strategy_counts_2 = df['strategy_p2'].apply(lambda x: np.array(eval(x))).apply(lambda x: len(x))

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

plt.figure(figsize=(8, 6))
plt.violinplot([epsilon_player_one, epsilon_player_two], showmeans=True)
plt.xticks([1, 2], ["Player 1", "Player 2"])  # Set x-axis categories
plt.ylabel("Epsilon values")
plt.title("Epsilon Distribution for Both Players")
plt.show()