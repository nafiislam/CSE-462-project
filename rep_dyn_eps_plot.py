import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('replicator_dynamics_results.csv')

epsilon_player_one = df["epsilon_1"].values
epsilon_player_two = df["epsilon_2"].values

strategy_counts = df['strategy_p1'].apply(lambda x: np.array(eval(x))).apply(lambda x: len(x))
strategy_counts_2 = df['strategy_p2'].apply(lambda x: np.array(eval(x))).apply(lambda x: len(x))

# Set global Seaborn style
sns.set_theme(style="whitegrid")

# Scatter plot for player 1
plt.figure(figsize=(8, 6))
sns.scatterplot(x=strategy_counts, y=epsilon_player_one, hue=epsilon_player_one, palette="viridis", s=100, legend=False)
# Add colorbar manually
norm = plt.Normalize(vmin=epsilon_player_one.min(), vmax=epsilon_player_one.max())  # Normalize epsilon values
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)  # Create ScalarMappable
sm.set_array([])  # Empty array to satisfy ScalarMappable requirements
plt.colorbar(sm, ax=plt.gca(), label="Epsilon Intensity")  # Add colorbar with label
plt.xlabel("Number of Strategies (Player 1)")
plt.ylabel("Epsilon (Player 1)")
plt.title("Epsilon vs Number of Strategies for Player 1")
plt.show()

# Scatter plot for player 2
plt.figure(figsize=(8, 6))
sns.scatterplot(x=strategy_counts_2, y=epsilon_player_two, hue=epsilon_player_two, palette="magma", s=100, legend=False)
norm = plt.Normalize(vmin=epsilon_player_two.min(), vmax=epsilon_player_two.max())  # Normalize epsilon values
sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)  # Create ScalarMappable
sm.set_array([])  # Empty array to satisfy ScalarMappable requirements
plt.colorbar(sm, ax=plt.gca(), label="Epsilon Intensity")  # Add colorbar with label
plt.xlabel("Number of Strategies (Player 2)")
plt.ylabel("Epsilon (Player 2)")
plt.title("Epsilon vs Number of Strategies for Player 2")
plt.show()

# Violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=[epsilon_player_one, epsilon_player_two], palette="muted")
plt.xticks([0, 1], ["Player 1", "Player 2"])  # Set x-axis categories
plt.ylabel("Epsilon Values")
plt.title("Epsilon Distribution for Both Players")
plt.show()

# Additional Plots
# Histogram of epsilon values
plt.figure(figsize=(8, 6))
sns.histplot(epsilon_player_one, color="blue", kde=True, label="Player 1", alpha=0.6)
sns.histplot(epsilon_player_two, color="orange", kde=True, label="Player 2", alpha=0.6)
plt.xlabel("Epsilon Values")
plt.ylabel("Frequency")
plt.title("Histogram of Epsilon Values")
plt.legend()
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=[epsilon_player_one, epsilon_player_two], palette="Set3")
plt.xticks([0, 1], ["Player 1", "Player 2"])
plt.ylabel("Epsilon Values")
plt.title("Box Plot of Epsilon Distribution")
plt.show()
