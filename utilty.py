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
    pos_count = len([i for i in strategy if i == 1])
    temp_x = -1
    temp_y = 0
    j = -1
    for i in range(len(strategy)):
        if strategy[i] == 1:
            temp = [0] * len(strategy)
            if temp_x == -1:
                temp_x = i
            temp[i] = payoff_matrix[i]
            for j in range(i+1, len(strategy)):
                if strategy[j] == 1:
                    temp[j] = -payoff_matrix[j]
                    temp_y = j
                    a.append(temp)
                    break
            if j == len(strategy) - 1:
                # if pos_count > 2:
                #     print(temp_x, temp_y)
                #     temp = [0] * len(strategy)
                #     temp[temp_y] = payoff_matrix[temp_y]
                #     temp[temp_x] = -payoff_matrix[temp_x]
                #     a.append(temp)
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


# a = np.array([1, 1, 0, 1])
# b = np.array([4, 2, 5, 7])
# print(solve_equation(a, b))

# m = 3  # For example, with m = 3
# combinations = generate_combinations(m)
# print(combinations)

