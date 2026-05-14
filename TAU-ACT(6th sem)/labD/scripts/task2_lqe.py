"""
Задание 2: Исследование LQE / Фильтр Калмана
Вариант 24 (чётный) → Фильтр Калмана (случайные сигналы f, ξ)
Таблица 1: Задание 2 → Таблица 3, №14
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

# --- Исходные данные (Таблица 3, №14) ---
A = np.array([
    [-35,  11,   6,  11],
    [-56,  17,  10,  18],
    [-22,   7,   5,   6],
    [-42,  12,  10,  13]
])
C = np.array([[-1, 0, 0, 1]])  # C^T = [-1, 0, 0, 1]^T → C = [-1, 0, 0, 1]
n = 4
p = 1  # размерность выхода

x0 = np.array([1.0, 1.0, 1.0, 1.0])
x0_hat = np.array([0.0, 0.0, 0.0, 0.0])

print("=== Задание 2: LQE / Фильтр Калмана ===")
print(f"A =\n{A}")
print(f"C = {C}")

# --- 1. Проверка обнаруживаемости ---
eigs_A = np.linalg.eigvals(A)
print(f"Собственные числа A: {eigs_A}")

# Матрица наблюдаемости
O = np.vstack([C, C @ A, C @ A @ A, C @ A @ A @ A])
rank_O = np.linalg.matrix_rank(O)
print(f"Матрица наблюдаемости O =\n{O}")
print(f"rank(O) = {rank_O}")

# Хаутус для наблюдаемости
hautus_obs_ranks = []
for lam in eigs_A:
    H = np.vstack([lam * np.eye(n) - A, C])
    hautus_obs_ranks.append(np.linalg.matrix_rank(H, tol=1e-6))
print(f"Ранги матриц Хаутуса (наблюдаемость): {hautus_obs_ranks}")

detectable = True
for lam, hr in zip(eigs_A, hautus_obs_ranks):
    if hr < n and np.real(lam) >= 0:
        detectable = False
        print(f"  Мода λ={lam:.4f} ненаблюдаема и неустойчива!")
    elif hr < n:
        print(f"  Мода λ={lam:.4f} ненаблюдаема, но устойчива (Re < 0)")
print(f"Обнаруживаемость: {'Да' if detectable else 'Нет'}")

# --- 2. Синтез наблюдателя для 4 пар (Q, R) ---
Q0 = np.eye(n)
R0 = np.array([[1.0]])
alpha = 5.0

cases = [
    ("Q, R",     Q0,         R0),
    ("αQ, R",    alpha * Q0, R0),
    ("Q, αR",    Q0,         alpha * R0),
    ("αQ, αR",   alpha * Q0, alpha * R0),
]

results = []
t_end = 16.0
np.random.seed(42)  # фиксируем seed для воспроизводимости

# Генерация шума заранее (одинаковый для всех пар)
dt_noise = 0.005
t_noise = np.arange(0, t_end + dt_noise, dt_noise)
N_noise = len(t_noise)

# Интенсивности шума
sigma_f = 0.5   # стандартное отклонение возмущения состояния
sigma_xi = 0.3  # стандартное отклонение шума измерения

f_noise = sigma_f * np.random.randn(n, N_noise)
xi_noise = sigma_xi * np.random.randn(p, N_noise)

def interp_noise(t_val, noise_arr):
    """Линейная интерполяция шума."""
    idx = np.searchsorted(t_noise, t_val) - 1
    idx = max(0, min(idx, N_noise - 2))
    frac = (t_val - t_noise[idx]) / dt_noise
    return noise_arr[:, idx] * (1 - frac) + noise_arr[:, idx + 1] * frac

for label, Q, R in cases:
    print(f"\n--- Пара ({label}) ---")
    
    # Решение уравнения Риккати для наблюдателя
    # AP + PA^T + Q - PC^T R^-1 C P = 0
    P = solve_continuous_are(A.T, C.T, Q, R)
    L = -P @ C.T @ np.linalg.inv(R)
    
    Aobs = A + L @ C
    eigs_obs = np.linalg.eigvals(Aobs)
    
    print(f"P =\n{P}")
    print(f"L = {L.flatten()}")
    print(f"σ(A+LC) = {eigs_obs}")
    
    # Моделирование
    def system_with_observer(t, state):
        x = state[:n]
        x_hat = state[n:]
        
        f_t = interp_noise(t, f_noise)
        xi_t = interp_noise(t, xi_noise)
        
        y = C @ x + xi_t  # измерение
        
        dx = A @ x + f_t
        dx_hat = A @ x_hat + L @ (C @ x_hat - y)
        
        return np.concatenate([dx, dx_hat])
    
    state0 = np.concatenate([x0, x0_hat])
    sol = solve_ivp(system_with_observer, [0, t_end], state0, max_step=0.01,
                    method='RK45', dense_output=True)
    
    t = sol.t
    x_t = sol.y[:n]
    xhat_t = sol.y[n:]
    e_t = x_t - xhat_t
    
    # Метрики
    e_norm = np.linalg.norm(e_t, axis=0)
    max_e = np.max(e_norm)
    final_e = e_norm[-1]
    
    print(f"max ||e(t)|| = {max_e:.6f}")
    print(f"||e(t_end)|| = {final_e:.6f}")
    
    results.append({
        'label': label,
        'Q': Q, 'R': R, 'P': P, 'L': L,
        'Aobs': Aobs, 'eigs_obs': eigs_obs,
        't': t, 'x': x_t, 'xhat': xhat_t, 'e': e_t,
        'max_e': max_e, 'final_e': final_e,
    })

# --- 3. Графики ---
case_labels_short = ['QR', 'aQR', 'QaR', 'aQaR']
case_labels_math = [r'$(Q, R)$', r'$(\alpha Q, R)$', r'$(Q, \alpha R)$', r'$(\alpha Q, \alpha R)$']

# Сравнение x(t) и x_hat(t) для каждой пары
for i, res in enumerate(results):
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    for j in range(n):
        axes[j].plot(res['t'], res['x'][j], label=f'$x_{j+1}(t)$', alpha=0.8)
        axes[j].plot(res['t'], res['xhat'][j], '--', label=f'$\\hat{{x}}_{j+1}(t)$', alpha=0.8)
        axes[j].set_ylabel(f'$x_{j+1}$')
        axes[j].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('t, с')
    fig.suptitle(f'Сравнение x(t) и $\\hat{{x}}(t)$ для пары {case_labels_math[i]}', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'task2_xcomp_{case_labels_short[i]}.png'), dpi=200)
    plt.close(fig)

# Невязка e(t) для каждой пары
for i, res in enumerate(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(n):
        ax.plot(res['t'], res['e'][j], label=f'$e_{j+1}(t)$')
    ax.set_xlabel('t, с')
    ax.set_ylabel('e(t)')
    ax.set_title(f'Невязка наблюдателя для пары {case_labels_math[i]}')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'task2_error_{case_labels_short[i]}.png'), dpi=200)
    plt.close(fig)

# Сравнение ||e(t)|| для всех пар
fig, ax = plt.subplots(figsize=(10, 5))
for i, res in enumerate(results):
    e_norm = np.linalg.norm(res['e'], axis=0)
    ax.plot(res['t'], e_norm, label=case_labels_math[i])
ax.set_xlabel('t, с')
ax.set_ylabel('||e(t)||')
ax.set_title('Сравнение нормы невязки')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'task2_error_compare.png'), dpi=200)
plt.close(fig)

# --- 4. Сохранение переменных для Typst ---
def fmt_mat(M, fmt='.4f'):
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for row in M:
        rows.append(', '.join(f'{v:{fmt}}' for v in row))
    return 'mat(' + '; '.join(rows) + ')'

def fmt_eigs(eigs):
    parts = []
    for e in sorted(eigs, key=lambda x: (np.real(x), np.imag(x))):
        re_p = np.real(e)
        im_p = np.imag(e)
        if abs(im_p) < 1e-8:
            parts.append(f'{re_p:.4f}')
        else:
            parts.append(f'{re_p:.4f} + {im_p:.4f}i')
    return ', '.join(parts)

lines = []
lines.append(f'// Автоматически сгенерировано task2_lqe.py')
lines.append(f'#let T2_A = $ {fmt_mat(A, ".0f")} $')
lines.append(f'#let T2_C = $ {fmt_mat(C, ".0f")} $')
lines.append(f'#let T2_O = $ {fmt_mat(O, ".0f")} $')
lines.append(f'#let T2_rank_O = {rank_O}')
lines.append(f'#let T2_eigs_A = $ {fmt_eigs(eigs_A)} $')
lines.append(f'#let T2_detectable = {"true" if detectable else "false"}')

for i, res in enumerate(results):
    tag = case_labels_short[i]
    lines.append(f'#let T2_{tag}_P = $ {fmt_mat(res["P"], ".4f")} $')
    lines.append(f'#let T2_{tag}_L = $ {fmt_mat(res["L"], ".4f")} $')
    lines.append(f'#let T2_{tag}_eigs_obs = $ {fmt_eigs(res["eigs_obs"])} $')

with open(os.path.join(var_dir, 'variables_task2.typ'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("\n=== Скрипт завершён. Графики и переменные сохранены. ===")
