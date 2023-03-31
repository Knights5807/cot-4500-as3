import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)


# Question 1
def euler_method(function_euler, range_euler, iter_euler, initial_euler):
    h_euler = (range_euler[1] - range_euler[0]) / iter_euler
    t_prev_euler = range_euler[0]
    y_prev_euler = initial_euler

    for i in range(iter_euler):
        t_curr_euler = t_prev_euler + h_euler
        y_curr_euler = y_prev_euler + h_euler * function_euler(t_prev_euler, y_prev_euler)
        t_prev_euler = t_curr_euler
        y_prev_euler = y_curr_euler

    return y_curr_euler


def function_euler(t, y):
    return t - y ** 2


range_euler = (0, 2)
iter_euler = 10
initial_euler = 1

print(euler_method(function_euler, range_euler, iter_euler, initial_euler))
print()


# Question 2
def runge_kutta(function_rk, range_rk, iter_rk, initial_rk):
    h_rk = (range_rk[1] - range_rk[0]) / iter_rk
    t_prev_rk = range_rk[0]
    y_prev_rk = initial_rk

    for i in range(iter_rk):
        k1 = h_rk * function_rk(t_prev_rk, y_prev_rk)
        k2 = h_rk * function_rk(t_prev_rk + h_rk / 2, y_prev_rk + k1 / 2)
        k3 = h_rk * function_rk(t_prev_rk + h_rk / 2, y_prev_rk + k2 / 2)
        k4 = h_rk * function_rk(t_prev_rk + h_rk, y_prev_rk + k3)

        t_curr_rk = t_prev_rk + h_rk
        y_curr_rk = y_prev_rk + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t_prev_rk = t_curr_rk
        y_prev_rk = y_curr_rk

    return y_curr_rk


def function_rk(t, y):
    return t - y ** 2


range_rk = (0, 2)
iter_rk = 10
initial_rk = 1

print(runge_kutta(function_rk, range_rk, iter_rk, initial_rk))
print()


# Question 3
g_array = np.array([[2, -1, 1, 6],
                    [1, 3, 1, 0],
                    [-1, 5, 4, -3]])

for i in range(g_array.shape[0]):
    max_row = i + np.argmax(np.abs(g_array[i:, i]))
    g_array[[i, max_row]] = g_array[[max_row, i]]
    for j in range(i + 1, g_array.shape[0]):
        factor = g_array[j, i] / g_array[i, i]
        g_array[j, i:] = g_array[j, i:] - factor * g_array[i, i:]

x = np.zeros(g_array.shape[0])
for i in range(g_array.shape[0] - 1, -1, -1):
    x[i] = (g_array[i, -1] - np.dot(g_array[i, :-1], x)) / g_array[i, i]
x = x.astype(int)

print(x)
print()

# Question 4
lu_array = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])


def lu_factorization(array):
    n = array.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1
        for j in range(k, n):
            U[k, j] = array[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k + 1, n):
            L[i, k] = (array[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


L, U = lu_factorization(lu_array)
determinant = np.linalg.det(U)

print(determinant)
print()
print(L)
print()
print(U)
print()


# Question 5
def diag_dom(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True


matrix = [
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8],
]
print(diag_dom(matrix))
print()


# Question 6
def pos_def(def_matrix):
    if def_matrix.shape[0] != def_matrix.shape[1]:
        return False
    if not np.allclose(def_matrix, def_matrix.T):
        return False
    eigenvalues = np.linalg.eigvals(def_matrix)
    return np.all(eigenvalues > 0)


matrix = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2],
])

print(pos_def(matrix))
