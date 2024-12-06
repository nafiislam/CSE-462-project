import numpy as np


def generate_combinations(m):
    combinations = []
    # Iterate over the range of 1 to 2^m - 1 (excluding empty set)
    for i in range(1, 2**m):
        # Get the binary representation of the index as a list of 0s and 1s
        binary_combination = [(i >> j) & 1 for j in range(m)]
        combinations.append(binary_combination)
    
    return combinations

def linear_solve(A, b):
    return np.linalg.solve(A, b)



def solve_equation(strategy, payoff_matrix):
    # print(strategy, payoff_matrix)
    for i in range(len(payoff_matrix)):
        if payoff_matrix[i] == 0:
            payoff_matrix[i] = 1e-16
    a = []
    j = -1
    for i in range(len(strategy)):
        if strategy[i] == 1:
            temp = [0] * len(strategy)
            temp[i] = payoff_matrix[i]
            for j in range(i+1, len(strategy)):
                if strategy[j] == 1:
                    temp[j] = -payoff_matrix[j]
                    a.append(temp)
                    break
            if j == len(strategy) - 1:
                break
    for i in range(len(strategy)):
        if strategy[i] == 0:
            temp = [0] * len(strategy)
            temp[i] = 1
            a.append(temp)
    a.append([1] * len(strategy))
    a = np.array(a)
    b = np.zeros(len(a))
    b[-1] = 1
    # print(a)
    # print(b)
    solve = linear_solve(a, b)

    return solve


# a = np.array([0, 1, 0, 1])
# b = np.array([4, 2, 5, 7])
# print(solve_equation(a, b))

# m = 3  # For example, with m = 3
# combinations = generate_combinations(m)
# print(combinations)

