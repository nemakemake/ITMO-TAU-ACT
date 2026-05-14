import numpy as np
import scipy.linalg as la

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -0.7560, 0],
    [0, 0, 0, 1],
    [0, 0, 4.8626, 0]
])
B = np.array([[0], [0.0041], [0], [-0.0019]])

def format_matrix(M):
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for row in M:
        rows.append(", ".join([f"{x:.4f}" for x in row]))
    return "$ mat(" + "; ".join(rows) + ") $"

# Sylvester K: AP - P G = -B Y
G = np.diag([-3, -4, -5, -6])
Y = np.array([[1, 1, 1, 1]])

# Solving for P in AP - PG = -BY -> AP + P(-G) = -BY
P = la.solve_sylvester(A, -G, -B @ Y)
K = Y @ la.inv(P)

print("G_K = " + format_matrix(G))
print("Y_K = " + format_matrix(Y))
print("P_K = " + format_matrix(P))
print("K_norm = " + format_matrix(K))

# Sylvester L: G Q - Q A = Y C  -> Q A - G Q = -Y C -> Q A + (-G) Q = -Y C
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
G_L = np.diag([-15, -16, -17, -18])
Y_L = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [1, -1]
])
# Q A - G_L Q = -Y_L C <=> Q A + (-G_L) Q = -Y_L C -> A^T Q^T - Q^T G_L^T = -C^T Y_L^T -> A^T Q^T + Q^T (-G_L^T) = -C^T Y_L^T
QT = la.solve_sylvester(A.T, -G_L.T, -C.T @ Y_L.T)
Q = QT.T
L = la.inv(Q) @ Y_L

print("G_L = " + format_matrix(G_L))
print("Y_L = " + format_matrix(Y_L))
print("Q_L = " + format_matrix(Q))
print("L_norm = " + format_matrix(L))
