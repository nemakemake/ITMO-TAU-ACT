import numpy as np

# Параметры (вариант 24)
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -0.7560, 0],
    [0, 0, 0, 1],
    [0, 0, 4.8626, 0]
])
B = np.array([[0], [0.0041], [0], [-0.0019]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

def format_matrix(M):
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for row in M:
        rows.append(", ".join([f"{x:.4f}" for x in row]))
    return "$ mat(" + "; ".join(rows) + ") $"

AB = A @ B
A2B = A @ AB
A3B = A @ A2B
Co = np.hstack([B, AB, A2B, A3B])

print("A B = " + format_matrix(AB))
print("A^2 B = " + format_matrix(A2B))
print("A^3 B = " + format_matrix(A3B))
print("Co = " + format_matrix(Co))

CA = C @ A
CA2 = CA @ A
CA3 = CA2 @ A
Ob = np.vstack([C, CA, CA2, CA3])

print("C A = " + format_matrix(CA))
print("C A^2 = " + format_matrix(CA2))
print("C A^3 = " + format_matrix(CA3))
print("Ob = " + format_matrix(Ob))
