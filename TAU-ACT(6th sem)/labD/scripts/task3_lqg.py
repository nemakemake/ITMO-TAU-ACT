"""
Задание 3: Синтез LQG
Вариант 24 (чётный) → детерминированные гармонические f и ξ
Таблица 1: Задание 3 → Таблица 4, №4
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
os.makedirs(out_dir, exist_ok=True)
var_dir = os.path.dirname(os.path.dirname(__file__))

# --- Исходные данные (Таблица 4, №4) ---
A = np.array([
    [ 5, -7, -5,  1],
    [-7,  5, -1,  5],
    [-5, -1,  5,  7],
    [ 1,  5,  7,  5]
])
B = np.array([
    [5, 0],
    [7, 0],
    [1, 0],
    [9, 0]
])
C = np.array([
    [0, 0, 2, 2],
    [1, 1, -1, 1]
])
D = np.array([
    [0, 4],
    [0, 2]
])

n = 4  # порядок системы
m = 2  # число входов
p = 2  # число выходов

x0 = np.array([1.0, 1.0, 1.0, 1.0])
x0_hat = np.array([0.0, 0.0, 0.0, 0.0])

print("=== Задание 3: LQG ===")
print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"C =\n{C}")
print(f"D =\n{D}")

eigs_A = np.linalg.eigvals(A)
print(f"Собственные числа A: {eigs_A}")

# --- 1. Проверка стабилизируемости ---
U_ctrl = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
rank_U = np.linalg.matrix_rank(U_ctrl)
print(f"rank(U) = {rank_U}")

hautus_ctrl = []
for lam in eigs_A:
    H = np.hstack([lam * np.eye(n) - A, B])
    hautus_ctrl.append(np.linalg.matrix_rank(H, tol=1e-6))
print(f"Ранги Хаутуса (управляемость): {hautus_ctrl}")

stabilizable = True
for lam, hr in zip(eigs_A, hautus_ctrl):
    if hr < n and np.real(lam) >= 0:
        stabilizable = False
    elif hr < n:
        print(f"  Мода λ={lam:.4f} неуправляема, но устойчива")
print(f"Стабилизируемость: {'Да' if stabilizable else 'Нет'}")

# --- 2. Проверка обнаруживаемости ---
O = np.vstack([C, C @ A, C @ A @ A, C @ A @ A @ A])
rank_O = np.linalg.matrix_rank(O)
print(f"rank(O) = {rank_O}")

hautus_obs = []
for lam in eigs_A:
    H = np.vstack([lam * np.eye(n) - A, C])
    hautus_obs.append(np.linalg.matrix_rank(H, tol=1e-6))
print(f"Ранги Хаутуса (наблюдаемость): {hautus_obs}")

detectable = True
for lam, hr in zip(eigs_A, hautus_obs):
    if hr < n and np.real(lam) >= 0:
        detectable = False
    elif hr < n:
        print(f"  Мода λ={lam:.4f} ненаблюдаема, но устойчива")
print(f"Обнаруживаемость: {'Да' if detectable else 'Нет'}")

# --- 3. Синтез регулятора K ---
QK = np.eye(n)
RK = np.eye(m)

PK = solve_continuous_are(A, B, QK, RK)
K = -np.linalg.solve(RK, B.T @ PK)

Acl_K = A + B @ K
eigs_cl_K = np.linalg.eigvals(Acl_K)
print(f"\nРегулятор:")
print(f"QK = I4, RK = I2")
print(f"PK =\n{PK}")
print(f"K =\n{K}")
print(f"σ(A+BK) = {eigs_cl_K}")

# --- 4. Синтез наблюдателя L ---
QL = np.eye(n)
RL = np.eye(p)

PL = solve_continuous_are(A.T, C.T, QL, RL)
L = -PL @ C.T @ np.linalg.inv(RL)

Acl_L = A + L @ C
eigs_cl_L = np.linalg.eigvals(Acl_L)
print(f"\nНаблюдатель:")
print(f"QL = I4, RL = I2")
print(f"PL =\n{PL}")
print(f"L =\n{L}")
print(f"σ(A+LC) = {eigs_cl_L}")

# --- 5. Моделирование LQG ---
t_end = 10.0

# Детерминированные гармонические возмущения (для чётного варианта)
def f_det(t):
    return np.array([
        np.sin(2*t),
        np.cos(2*t),
        0.5 * np.sin(3*t),
        0.5 * np.cos(3*t)
    ])

def xi_det(t):
    return np.array([
        0.2 * np.sin(4*t),
        0.2 * np.cos(4*t)
    ])

def lqg_system(t, state):
    x = state[:n]
    x_hat = state[n:]
    
    u = K @ x_hat  # управление по оценке
    f_t = f_det(t)
    xi_t = xi_det(t)
    
    y = C @ x + D @ u + xi_t  # измерение
    
    dx = A @ x + B @ u + f_t
    dx_hat = A @ x_hat + B @ u + L @ (C @ x_hat + D @ u - y)
    
    return np.concatenate([dx, dx_hat])

state0 = np.concatenate([x0, x0_hat])
sol = solve_ivp(lqg_system, [0, t_end], state0, max_step=0.005,
                method='RK45', dense_output=True)

t = sol.t
x_t = sol.y[:n]
xhat_t = sol.y[n:]
e_t = x_t - xhat_t

# Вычисление u(t)
u_t = K @ xhat_t

max_u_norm = np.max(np.linalg.norm(u_t, axis=0))
print(f"\nmax ||u(t)|| = {max_u_norm:.6f}")

# --- 6. Графики ---

# Управление u(t)
fig, ax = plt.subplots(figsize=(10, 5))
for j in range(m):
    ax.plot(t, u_t[j], label=f'$u_{j+1}(t)$')
ax.set_xlabel('t, с')
ax.set_ylabel('u(t)')
ax.set_title('Управление $u(t) = K\\hat{x}(t)$')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task3_u.png'), dpi=200)
plt.close(fig)

# Невязка e(t)
fig, ax = plt.subplots(figsize=(10, 5))
for j in range(n):
    ax.plot(t, e_t[j], label=f'$e_{j+1}(t)$')
ax.set_xlabel('t, с')
ax.set_ylabel('e(t)')
ax.set_title('Невязка наблюдателя $e(t) = x(t) - \\hat{x}(t)$')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task3_error.png'), dpi=200)
plt.close(fig)

# Сравнение x(t) и x_hat(t)
fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
for j in range(n):
    axes[j].plot(t, x_t[j], label=f'$x_{j+1}(t)$', alpha=0.8)
    axes[j].plot(t, xhat_t[j], '--', label=f'$\\hat{{x}}_{j+1}(t)$', alpha=0.8)
    axes[j].set_ylabel(f'$x_{j+1}$')
    axes[j].legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('t, с')
fig.suptitle('Сравнение $x(t)$ и $\\hat{x}(t)$', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task3_xcomp.png'), dpi=200)
plt.close(fig)

# --- 7. Сохранение переменных ---
def fmt_mat(M, fmt='.4f'):
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for row in M:
        rows.append(', '.join(f'{v:{fmt}}' for v in row))
    return 'mat(' + '; '.join(rows) + ')'

def fmt_eigs(eigs):
    parts = []
    for e in sorted(eigs, key=lambda x: np.real(x)):
        re_p = np.real(e)
        im_p = np.imag(e)
        if abs(im_p) < 1e-8:
            parts.append(f'{re_p:.4f}')
        else:
            parts.append(f'{re_p:.4f} + {im_p:.4f}i')
    return ', '.join(parts)

lines = []
lines.append('// Автоматически сгенерировано task3_lqg.py')
lines.append(f'#let T3_A = $ {fmt_mat(A, ".0f")} $')
lines.append(f'#let T3_B = $ {fmt_mat(B, ".0f")} $')
lines.append(f'#let T3_C = $ {fmt_mat(C, ".0f")} $')
lines.append(f'#let T3_D = $ {fmt_mat(D, ".0f")} $')
lines.append(f'#let T3_eigs_A = $ {fmt_eigs(eigs_A)} $')
lines.append(f'#let T3_rank_U = {rank_U}')
lines.append(f'#let T3_rank_O = {rank_O}')
lines.append(f'#let T3_PK = $ {fmt_mat(PK, ".4f")} $')
lines.append(f'#let T3_K = $ {fmt_mat(K, ".4f")} $')
lines.append(f'#let T3_eigs_cl_K = $ {fmt_eigs(eigs_cl_K)} $')
lines.append(f'#let T3_PL = $ {fmt_mat(PL, ".4f")} $')
lines.append(f'#let T3_L = $ {fmt_mat(L, ".4f")} $')
lines.append(f'#let T3_eigs_cl_L = $ {fmt_eigs(eigs_cl_L)} $')

with open(os.path.join(var_dir, 'variables_task3.typ'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("\n=== Скрипт завершён. Графики и переменные сохранены. ===")
