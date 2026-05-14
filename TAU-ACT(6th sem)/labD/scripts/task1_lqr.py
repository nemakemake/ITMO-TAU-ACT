"""
Задание 1: Исследование LQR
Вариант 24, Таблица 2 №14
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm
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

# --- Исходные данные ---
A = np.array([
    [12, -1, 14],
    [ 6,  0,  6],
    [-6, -2, -8]
])
B = np.array([[11], [7], [-7]])
x0 = np.array([1, 1, 1])
n = 3

# --- 1. Проверка стабилизируемости ---
# Матрица управляемости
U_ctrl = np.hstack([B, A @ B, A @ A @ B])
rank_U = np.linalg.matrix_rank(U_ctrl)
eigs_A = np.linalg.eigvals(A)

print("=== Задание 1: LQR ===")
print(f"A =\n{A}")
print(f"B =\n{B}")
print(f"Матрица управляемости U =\n{U_ctrl}")
print(f"rank(U) = {rank_U}")
print(f"Собственные числа A: {eigs_A}")

# Проверка Хаутуса для каждого собственного числа
hautus_ranks = []
for lam in eigs_A:
    H = np.hstack([lam * np.eye(n) - A, B])
    hautus_ranks.append(np.linalg.matrix_rank(H, tol=1e-6))
print(f"Ранги матриц Хаутуса: {hautus_ranks}")

# Проверка стабилизируемости: все неуправляемые моды в LHP
stabilizable = True
for lam, hr in zip(eigs_A, hautus_ranks):
    if hr < n and np.real(lam) >= 0:
        stabilizable = False
        print(f"  Мода λ={lam:.4f} неуправляема и неустойчива!")
print(f"Стабилизируемость: {'Да' if stabilizable else 'Нет'}")

# --- 2. Синтез LQR для 4 пар (Q, R) ---
Q0 = np.eye(n)
R0 = np.array([[1.0]])
alpha = 5.0

cases = [
    ("Q, R",          Q0,         R0),
    ("αQ, R",         alpha * Q0, R0),
    ("Q, αR",         Q0,         alpha * R0),
    ("αQ, αR",        alpha * Q0, alpha * R0),
]

results = []
t_end = 15.0

for label, Q, R in cases:
    print(f"\n--- Пара ({label}) ---")
    
    # Решение уравнения Риккати
    P = solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ P)
    
    Acl = A + B @ K
    eigs_cl = np.linalg.eigvals(Acl)
    Jmin = x0 @ P @ x0
    
    print(f"P =\n{P}")
    print(f"K = {K}")
    print(f"Jmin = {Jmin:.10f}")
    print(f"σ(A+BK) = {eigs_cl}")
    
    # Моделирование замкнутой системы
    def closed_loop(t, x):
        u = K @ x
        return (A + B @ K) @ x
    
    sol = solve_ivp(closed_loop, [0, t_end], x0, max_step=0.01, dense_output=True)
    t = sol.t
    x_t = sol.y  # shape (3, N)
    
    # Вычисление u(t) и Jexp(t)
    u_t = K @ x_t  # shape (1, N)
    
    # Интегральный функционал качества (кумулятивный)
    integrand = np.array([
        x_t[:, i] @ Q @ x_t[:, i] + u_t[:, i] @ R @ u_t[:, i]
        for i in range(len(t))
    ])
    Jexp = np.cumsum(integrand[:-1] * np.diff(t))
    Jexp = np.concatenate([[0], Jexp])
    
    results.append({
        'label': label,
        'Q': Q, 'R': R, 'P': P, 'K': K,
        'Acl': Acl, 'eigs_cl': eigs_cl,
        'Jmin': Jmin,
        't': t, 'x': x_t, 'u': u_t,
        'Jexp': Jexp
    })
    
    print(f"Jexp(t_end) = {Jexp[-1]:.10f}")

# --- 3. Графики ---
case_labels_short = ['QR', 'aQR', 'QaR', 'aQaR']
case_labels_math = [r'$(Q, R)$', r'$(\alpha Q, R)$', r'$(Q, \alpha R)$', r'$(\alpha Q, \alpha R)$']

# Графики x(t) для каждой пары — отдельные рисунки
for i, res in enumerate(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(n):
        ax.plot(res['t'], res['x'][j], label=f'$x_{j+1}(t)$')
    ax.set_xlabel('t, с')
    ax.set_ylabel('x(t)')
    ax.set_title(f'Вектор состояния для пары {case_labels_math[i]}')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'task1_x_{case_labels_short[i]}.png'), dpi=200)
    plt.close(fig)

# Сравнение u(t)
fig, ax = plt.subplots(figsize=(10, 5))
for i, res in enumerate(results):
    ax.plot(res['t'], res['u'][0], label=case_labels_math[i])
ax.set_xlabel('t, с')
ax.set_ylabel('u(t)')
ax.set_title('Сравнение сигналов управления')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task1_u_compare.png'), dpi=200)
plt.close(fig)

# Сравнение Jexp(t) с Jmin
fig, ax = plt.subplots(figsize=(10, 5))
for i, res in enumerate(results):
    ax.plot(res['t'], res['Jexp'], label=f'{case_labels_math[i]}')
    ax.axhline(y=res['Jmin'], linestyle='--', alpha=0.7,
               label=f'$J_{{min}}$ = {res["Jmin"]:.2f}')
ax.set_xlabel('t, с')
ax.set_ylabel('J(t)')
ax.set_title('Сравнение $J_{exp}(t)$ и $J_{min}$')
ax.legend(loc='center right', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task1_J_compare.png'), dpi=200)
plt.close(fig)

# --- 4. Сохранение переменных для Typst ---
def fmt_mat(M, fmt='.4f'):
    """Форматирует матрицу для Typst."""
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for row in M:
        rows.append(', '.join(f'{v:{fmt}}' for v in row))
    return 'mat(' + '; '.join(rows) + ')'

def fmt_vec(v, fmt='.4f'):
    """Форматирует вектор для Typst."""
    return 'mat(' + '; '.join(f'{x:{fmt}}' for x in v) + ')'

def fmt_eigs(eigs):
    """Форматирует собственные числа для Typst."""
    parts = []
    seen = set()
    for e in eigs:
        re_part = np.real(e)
        im_part = np.imag(e)
        key = (round(re_part, 6), round(abs(im_part), 6))
        if key in seen:
            continue
        if abs(im_part) < 1e-8:
            parts.append(f'{re_part:.4f}')
        else:
            if abs(re_part) < 1e-8:
                parts.append(f'plus.minus {abs(im_part):.4f}i')
            else:
                parts.append(f'{re_part:.4f} plus.minus {abs(im_part):.4f}i')
            seen.add(key)
            seen.add((round(re_part, 6), round(abs(im_part), 6)))
    return ', '.join(parts)

lines = []
lines.append(f'// Автоматически сгенерировано task1_lqr.py')
lines.append(f'#let T1_A = $ {fmt_mat(A, ".0f")} $')
lines.append(f'#let T1_B = $ {fmt_vec(B.flatten(), ".0f")} $')
lines.append(f'#let T1_U = $ {fmt_mat(U_ctrl, ".0f")} $')
lines.append(f'#let T1_rank_U = {rank_U}')
lines.append(f'#let T1_eigs_A = $ {fmt_eigs(eigs_A)} $')
lines.append(f'#let T1_stabilizable = {"true" if stabilizable else "false"}')

for i, res in enumerate(results):
    tag = case_labels_short[i]
    lines.append(f'#let T1_{tag}_P = $ {fmt_mat(res["P"], ".4f")} $')
    lines.append(f'#let T1_{tag}_K = $ {fmt_mat(res["K"], ".4f")} $')
    lines.append(f'#let T1_{tag}_Jmin = {res["Jmin"]:.4f}')
    lines.append(f'#let T1_{tag}_eigs_cl = $ {fmt_eigs(res["eigs_cl"])} $')
    lines.append(f'#let T1_{tag}_Jexp = {res["Jexp"][-1]:.4f}')

with open(os.path.join(var_dir, 'variables_task1.typ'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("\n=== Скрипт завершён. Графики и переменные сохранены. ===")
