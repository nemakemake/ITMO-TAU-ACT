import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import control as ct
import matplotlib.pyplot as plt
import os

os.makedirs('images', exist_ok=True)

# 1.4 Выбор параметров
np.random.seed(24)
M = np.random.randint(100000, 1000001) / 1000 / np.sqrt(2)
m1 = np.random.randint(1000, 10001) / 1000 * np.sqrt(3)
m2 = np.random.randint(1000, 10001) / 1000 * np.sqrt(3)
l = np.random.randint(100, 1001) / np.sqrt(5) / 100
g = 9.81

print("=== ПАРАМЕТРЫ СИСТЕМЫ ===")
print(f"M = {M:.4f} кг")
print(f"m1 = {m1:.4f} кг")
print(f"m2 = {m2:.4f} кг")
print(f"l = {l:.4f} м")

M_t = M + m1 + m2
L_t = (m1 / 3 + m2) * l**2
S_t = (m1 / 2 + m2) * l

denom = M_t * L_t - S_t**2

a23 = - S_t**2 * g / denom
a43 = M_t * S_t * g / denom

b2 = L_t / denom
b4 = - S_t / denom

d2 = - S_t / denom
d4 = M_t / denom

A = np.array([
    [0, 1, 0, 0],
    [0, 0, a23, 0],
    [0, 0, 0, 1],
    [0, 0, a43, 0]
])

B = np.array([
    [0],
    [b2],
    [0],
    [b4]
])

D = np.array([
    [0],
    [d2],
    [0],
    [d4]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

print("\n=== МАТРИЦЫ ЛИНЕАРИЗОВАННОЙ МОДЕЛИ ===")
print("A = \n", np.round(A, 4))
print("B = \n", np.round(B, 4))
print("C = \n", np.round(C, 4))
print("D = \n", np.round(D, 4))

# 2.1 Анализ матриц
eigvals, eigvecs = la.eig(A)
print("\n=== СОБСТВЕННЫЕ ЧИСЛА A ===")
print(np.round(eigvals, 4))

Co = ct.ctrb(A, B)
rank_Co = np.linalg.matrix_rank(Co)
print(f"Ранг матрицы управляемости: {rank_Co}")

Ob = ct.obsv(A, C)
rank_Ob = np.linalg.matrix_rank(Ob)
print(f"Ранг матрицы наблюдаемости: {rank_Ob}")

# 2.2 Передаточные функции
sys_u = ct.ss(A, B, C, np.zeros((2,1)))
print("\n=== ПЕРЕДАТОЧНАЯ ФУНКЦИЯ u -> y1 ===")
print(ct.tf(sys_u[0, 0]))
print("\n=== ПЕРЕДАТОЧНАЯ ФУНКЦИЯ u -> y2 ===")
print(ct.tf(sys_u[1, 0]))

sys_f = ct.ss(A, D, C, np.zeros((2,1)))
print("\n=== ПЕРЕДАТОЧНАЯ ФУНКЦИЯ f -> y1 ===")
print(ct.tf(sys_f[0, 0]))
print("\n=== ПЕРЕДАТОЧНАЯ ФУНКЦИЯ f -> y2 ===")
print(ct.tf(sys_f[1, 0]))

# 2.3 Моделирование
from scipy.integrate import solve_ivp

def nonlinear_sys(t, x):
    u = 0
    f = 0
    x1, x2, x3, x4 = x
    
    den = M_t * L_t - S_t**2 * np.cos(x3)**2
    
    dx1 = x2
    dx2 = (L_t * (u + S_t * x4**2 * np.sin(x3)) - S_t * np.cos(x3) * (f + S_t * g * np.sin(x3))) / den
    dx3 = x4
    dx4 = (M_t * (f + S_t * g * np.sin(x3)) - S_t * np.cos(x3) * (u + S_t * x4**2 * np.sin(x3))) / den
    
    return [dx1, dx2, dx3, dx4]

def linear_sys(t, x):
    dx = A @ x
    return dx

x0_sets = [
    [0.1, 0, 0, 0],
    [0, 0.1, 0, 0],
    [0, 0, 0.1, 0],
    [0, 0, 0, 0.1],
    [0.05, 0.05, 0.05, -0.05]
]

t_span = (0, 2)
t_eval = np.linspace(0, 2, 500)

for i, x0 in enumerate(x0_sets):
    sol_nl = solve_ivp(nonlinear_sys, t_span, x0, t_eval=t_eval)
    sol_l = solve_ivp(linear_sys, t_span, x0, t_eval=t_eval)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sol_l.t, sol_l.y[0], '--', label=r'Linear $a(t)$')
    plt.plot(sol_nl.t, sol_nl.y[0], '-', label=r'Nonlinear $a(t)$')
    plt.plot(sol_l.t, sol_l.y[2], '--', label=r'Linear $\varphi(t)$')
    plt.plot(sol_nl.t, sol_nl.y[2], '-', label=r'Nonlinear $\varphi(t)$')
    plt.title(f'Сравнение свободного движения при x0 = {x0}')
    plt.xlabel('Время $t$, с')
    plt.ylabel('Координаты')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/free_resp_{i+1}.png', dpi=300)
    plt.close()

# Большое время моделирования (чтобы показать расхождение или сильный рост)
t_span_long = (0, 10)
t_eval_long = np.linspace(0, 10, 1000)
x0_long = [0, 0, 0.01, 0] # небольшое отклонение по углу
sol_nl_long = solve_ivp(nonlinear_sys, t_span_long, x0_long, t_eval=t_eval_long)
sol_l_long = solve_ivp(linear_sys, t_span_long, x0_long, t_eval=t_eval_long)

plt.figure(figsize=(10, 6))
plt.plot(sol_l_long.t, sol_l_long.y[0], '--', label=r'Linear $a(t)$')
plt.plot(sol_nl_long.t, sol_nl_long.y[0], '-', label=r'Nonlinear $a(t)$')
plt.plot(sol_l_long.t, sol_l_long.y[2], '--', label=r'Linear $\varphi(t)$')
plt.plot(sol_nl_long.t, sol_nl_long.y[2], '-', label=r'Nonlinear $\varphi(t)$')
plt.title(f'Сравнение на большом отрезке времени при x0 = {x0_long}')
plt.xlabel('Время $t$, с')
plt.ylabel('Координаты')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'images/free_resp_long.png', dpi=300)
plt.close()
