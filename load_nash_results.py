import numpy as np
import pandas as pd
import warnings
import random
import ast
import utilty


np.random.seed(42)
random.seed(42)


def load_dataset():
    dataframe = pd.read_csv("nash_results.csv")
    dataframe["player_1_strategy"] = dataframe["player_1_strategy"].apply(ast.literal_eval)
    dataframe["player_2_strategy"] = dataframe["player_2_strategy"].apply(ast.literal_eval)
    return dataframe

df = load_dataset()
print(df.head())