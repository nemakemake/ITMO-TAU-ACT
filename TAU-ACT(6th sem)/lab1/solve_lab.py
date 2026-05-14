import os
import sys
import io
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sympy as sp

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def precompute_u(A, B, Wc_inv, target, t1, n_pts=2000):
    """Precompute u(t) = B^T exp(A^T(t1-t)) Wc_inv target on a grid and return interpolant."""
    t_pts = np.linspace(0, t1, n_pts)
    u_pts = np.array([float(B.T @ la.expm(A.T * (t1 - t)) @ Wc_inv @ target) for t in t_pts])
    return interp1d(t_pts, u_pts, kind='cubic', fill_value='extrapolate')

# Настройка путей
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'images')
os.makedirs(img_dir, exist_ok=True)
vars_file = os.path.join(base_dir, 'variables.typ')

def format_num(val, tol=1e-8):
    if abs(val) < tol: return "0"
    s = f"{val:.4f}".rstrip('0').rstrip('.')
    if s == "-0": return "0"
    return s

def format_sci(val):
    import math as _math
    if abs(val) < 1e-15: return "$0$"
    exp = int(_math.floor(_math.log10(abs(val))))
    mantissa = val / 10**exp
    m = f"{mantissa:.2f}".rstrip('0').rstrip('.')
    return f"${m} times 10^({exp})$"

def mat2typst(M):
    M = np.array(M)
    if M.ndim == 1: M = M.reshape(-1, 1)
    rows = []
    for r in M:
        row_str = []
        for val in r:
            if isinstance(val, (complex, np.complex128, np.complex64)) and abs(val.imag) > 1e-4:
                r_str = format_num(val.real)
                i_str = format_num(val.imag)
                sign = "+" if val.imag > 0 else "-"
                abs_i = format_num(abs(val.imag))
                if abs_i == "1": abs_i = ""
                if r_str == "0": row_str.append(f"{'-' if sign=='-' else ''}{abs_i}i")
                else: row_str.append(f"{r_str} {sign} {abs_i}i")
            else:
                row_str.append(format_num(val.real if isinstance(val, (complex, np.complex128)) else val))
        rows.append(", ".join(row_str))
    return "$mat(" + "; ".join(rows) + ")$"

def compute_Wc(A, B, t1):
    n = A.shape[0]
    M = np.zeros((2*n, 2*n))
    M[:n, :n] = -A
    M[:n, n:] = B @ B.T
    M[n:, n:] = A.T
    expM = la.expm(M * t1)
    F12 = expM[:n, n:]
    F22 = expM[n:, n:]
    return F22.T @ F12

variables = {}
def add_var(name, val):
    variables[name] = val

# =====================================================
# ВАРИАНТ 24
# Задания 1, 2, 5: условие №14
# Задания 3, 4: условие №4
# =====================================================

# ------------------ Задание 1 ------------------
# Таблица 2, условие №14
A1 = np.array([[3, -6, 4], [4, -5, 4], [-4, 4, -5]])
B1 = np.array([[-1], [3], [1]])
x1 = np.array([[1], [0], [0]])
t1_time = 3.0
time1 = np.linspace(0, t1_time, 500)

add_var("T1_A", mat2typst(A1)); add_var("T1_B", mat2typst(B1)); add_var("T1_x1", mat2typst(x1))
AB1 = A1@B1; A2B1 = A1@AB1
add_var("T1_AB", mat2typst(AB1)); add_var("T1_A2B", mat2typst(A2B1))
U1 = np.hstack([B1, AB1, A2B1])
add_var("T1_U", mat2typst(U1)); add_var("T1_rank_U", str(np.linalg.matrix_rank(U1)))
add_var("T1_detU", format_num(np.linalg.det(U1)))

eigvals1, _ = la.eig(A1)
eigvals1 = sorted(eigvals1, key=lambda x: (x.real, x.imag))
add_var("T1_eigs", ", ".join([format_num(e) for e in eigvals1]))

H_ranks1 = []
for i, eig in enumerate(eigvals1):
    H = np.hstack([eig * np.eye(3) - A1, B1])
    H_ranks1.append(str(np.linalg.matrix_rank(H)))
    add_var(f"T1_H{i+1}", mat2typst(H))
add_var("T1_H_ranks", ", ".join(H_ranks1))

# Жорданова форма
# Собственный вектор для lambda=-1: (1, 0, -1)
# Собственный вектор для lambda=-3+2i: (3+i, 2, -2) -> Re=(3,2,-2), Im=(1,0,0)
T1_P_j = np.array([[3, 1, 1], [2, 0, 0], [-2, 0, -1]], dtype=float)
T1_P_inv_j = la.inv(T1_P_j)
T1_J_j = T1_P_inv_j @ A1 @ T1_P_j
T1_B_hat = T1_P_inv_j @ B1
add_var("T1_J", mat2typst(T1_J_j))
add_var("T1_P", mat2typst(T1_P_j))
add_var("T1_P_inv", mat2typst(T1_P_inv_j))
add_var("T1_B_hat", mat2typst(T1_B_hat))

Wc1 = compute_Wc(A1, B1, t1_time)
add_var("T1_Wc", mat2typst(Wc1))
add_var("T1_Wc_eigs", ", ".join([format_num(e) for e in la.eigvals(Wc1)]))

Wc1_inv = la.pinv(Wc1)
u1_func = precompute_u(A1, B1, Wc1_inv, x1, t1_time)

sol1 = solve_ivp(lambda t, x: A1@x + B1.flatten()*u1_func(t), [0, t1_time], np.zeros(3),
                 t_eval=time1, rtol=1e-9, atol=1e-12)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time1, sol1.y[0], label='$x_1$')
plt.plot(time1, sol1.y[1], label='$x_2$')
plt.plot(time1, sol1.y[2], label='$x_3$')
plt.axhline(x1[0,0], color='C0', linestyle='--')
plt.axhline(x1[1,0], color='C1', linestyle='--')
plt.axhline(x1[2,0], color='C2', linestyle='--')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Траектории состояний x(t)'); plt.xlabel('Время t, с')
plt.subplot(1, 2, 2)
plt.plot(time1, [u1_func(t) for t in time1], 'k', label='$u(t)$')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Управляющее воздействие u(t)'); plt.xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task1_sim.png'), dpi=300); plt.close()

# Ошибка управления
T1_err = sol1.y[:, -1] - x1.flatten()
add_var("T1_err", mat2typst(T1_err))
add_var("T1_err_norm", format_sci(np.linalg.norm(T1_err)))

# Приближённый участок (zoom)
zoom_mask = time1 >= 2.7
fig, ax = plt.subplots(figsize=(6, 4))
for i, lbl in enumerate(['$x_1$', '$x_2$', '$x_3$']):
    ax.plot(time1[zoom_mask], sol1.y[i][zoom_mask], color=f'C{i}', label=lbl)
    ax.scatter([t1_time], [x1[i, 0]], color=f'C{i}', s=80, zorder=5)
ax.legend(loc='best', framealpha=0.9); ax.grid(True)
ax.set_title('Траектории состояний x(t) (приближено)')
ax.set_xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task1_sim_zoom.png'), dpi=300); plt.close()

# ------------------ Задание 2 ------------------
# Таблица 3, условие №14
B2 = np.array([[1], [0], [0]])
x1_prime = np.array([[0], [0], [1]])
x1_dprime = np.array([[1], [0], [0]])
AB2 = A1@B2; A2B2 = A1@AB2
U2 = np.hstack([B2, AB2, A2B2])

add_var("T2_B", mat2typst(B2)); add_var("T2_x1_prime", mat2typst(x1_prime)); add_var("T2_x1_dprime", mat2typst(x1_dprime))
add_var("T2_AB", mat2typst(AB2)); add_var("T2_A2B", mat2typst(A2B2))
add_var("T2_U", mat2typst(U2)); add_var("T2_rank_U", str(np.linalg.matrix_rank(U2)))
add_var("T2_check_prime", str(np.linalg.matrix_rank(np.hstack([U2, x1_prime]))))
add_var("T2_check_dprime", str(np.linalg.matrix_rank(np.hstack([U2, x1_dprime]))))

Wc2 = compute_Wc(A1, B2, t1_time)
Wc2_inv = la.pinv(Wc2)
add_var("T2_Wc", mat2typst(Wc2))
u2_func = precompute_u(A1, B2, Wc2_inv, x1_dprime, t1_time)

sol2 = solve_ivp(lambda t, x: A1@x + B2.flatten()*u2_func(t), [0, t1_time], np.zeros(3),
                 t_eval=time1, rtol=1e-9, atol=1e-12)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time1, sol2.y[0], label='$x_1$')
plt.plot(time1, sol2.y[1], label='$x_2$')
plt.plot(time1, sol2.y[2], label='$x_3$')
plt.axhline(x1_dprime[0,0], color='C0', linestyle='--')
plt.axhline(x1_dprime[1,0], color='C1', linestyle='--')
plt.axhline(x1_dprime[2,0], color='C2', linestyle='--')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title("Траектории состояний x(t), цель $x_1''$"); plt.xlabel('Время t, с')
plt.subplot(1, 2, 2)
plt.plot(time1, [u2_func(t) for t in time1], 'k', label='$u(t)$')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Управляющее воздействие u(t)'); plt.xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task2_sim.png'), dpi=300); plt.close()

# Приближённый участок Task 2 (zoom)
fig, ax = plt.subplots(figsize=(6, 4))
for i, lbl in enumerate(['$x_1$', '$x_2$', '$x_3$']):
    ax.plot(time1[zoom_mask], sol2.y[i][zoom_mask], color=f'C{i}', label=lbl)
    ax.scatter([t1_time], [x1_dprime[i, 0]], color=f'C{i}', s=80, zorder=5)
ax.legend(loc='best', framealpha=0.9); ax.grid(True)
ax.set_title("Траектории состояний x(t) (приближено), цель $x_1''$")
ax.set_xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task2_sim_zoom.png'), dpi=300); plt.close()

# Hautus для Задания 2 (управляемость с B2)
T2_H_ranks = []
for i, eig in enumerate(eigvals1):
    H = np.hstack([eig * np.eye(3) - A1, B2])
    T2_H_ranks.append(str(np.linalg.matrix_rank(H)))
    add_var(f"T2_H{i+1}", mat2typst(H))
add_var("T2_H_ranks", ", ".join(T2_H_ranks))
add_var("T2_Wc_eigs", ", ".join([format_num(e) for e in la.eigvals(Wc2)]))

# B_hat для B2 в жордановом базисе
T2_B_hat = T1_P_inv_j @ B2
add_var("T2_B_hat", mat2typst(T2_B_hat))

# Ошибка управления (достижимая точка x1'')
T2_err = sol2.y[:, -1] - x1_dprime.flatten()
add_var("T2_err", mat2typst(T2_err))
add_var("T2_err_norm", format_sci(np.linalg.norm(T2_err)))

# Моделирование с недостижимой целью x1'
u2_prime_func = precompute_u(A1, B2, Wc2_inv, x1_prime, t1_time)
sol2_prime = solve_ivp(lambda t, x: A1@x + B2.flatten()*u2_prime_func(t), [0, t1_time], np.zeros(3),
                        t_eval=time1, rtol=1e-6, atol=1e-9)
T2_err_prime = sol2_prime.y[:, -1] - x1_prime.flatten()
add_var("T2_err_prime", mat2typst(T2_err_prime))
add_var("T2_err_norm_prime", format_sci(np.linalg.norm(T2_err_prime)))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time1, sol2_prime.y[0], label='$x_1$')
plt.plot(time1, sol2_prime.y[1], label='$x_2$')
plt.plot(time1, sol2_prime.y[2], label='$x_3$')
plt.axhline(x1_prime[0,0], color='C0', linestyle='--')
plt.axhline(x1_prime[1,0], color='C1', linestyle='--')
plt.axhline(x1_prime[2,0], color='C2', linestyle='--')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title("Траектории состояний x(t), цель $x_1'$"); plt.xlabel('Время t, с')
plt.subplot(1, 2, 2)
plt.plot(time1, [u2_prime_func(t) for t in time1], 'k', label='$u(t)$')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Управляющее воздействие u(t)'); plt.xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task2_sim_prime.png'), dpi=300); plt.close()

# ------------------ Задание 3 ------------------
# Таблица 4, условие №4
A3 = np.array([[-8, -3, -12], [-3, -2, -6], [6, 0, 7]])
C3 = np.array([[1, 0, 2]])
CA3 = C3@A3; CA2_3 = C3@A3@A3
V3 = np.vstack([C3, CA3, CA2_3])

add_var("T3_A", mat2typst(A3)); add_var("T3_C", mat2typst(C3))
add_var("T3_CA", mat2typst(CA3)); add_var("T3_CA2", mat2typst(CA2_3))
add_var("T3_V", mat2typst(V3)); add_var("T3_rank_V", str(np.linalg.matrix_rank(V3)))

eigvals3, _ = la.eig(A3)
eigvals3 = sorted(eigvals3, key=lambda x: (x.real, x.imag))
add_var("T3_eigs", ", ".join([format_num(e) for e in eigvals3]))

H_ranks3 = []
for i, eig in enumerate(eigvals3):
    H = np.vstack([eig * np.eye(3) - A3, C3])
    H_ranks3.append(str(np.linalg.matrix_rank(H)))
    add_var(f"T3_H{i+1}", mat2typst(H))
add_var("T3_H_ranks", ", ".join(H_ranks3))

# Жорданова форма для A3
# Собственный вектор для lambda=1: (-1, -1, 1)
# Собственный вектор для lambda=-2+3i: (-3+i, -1+i, 2) -> Re=(-3,-1,2), Im=(1,1,0)
T3_P_j = np.array([[-3, 1, -1], [-1, 1, -1], [2, 0, 1]], dtype=float)
T3_P_inv_j = la.inv(T3_P_j)
T3_J_j = T3_P_inv_j @ A3 @ T3_P_j
T3_C_hat = C3 @ T3_P_j
add_var("T3_J", mat2typst(T3_J_j))
add_var("T3_P", mat2typst(T3_P_j))
add_var("T3_P_inv", mat2typst(T3_P_inv_j))
add_var("T3_C_hat", mat2typst(T3_C_hat))

Wo3 = compute_Wc(A3.T, C3.T, t1_time)
add_var("T3_Wo", mat2typst(Wo3)); add_var("T3_Wo_eigs", ", ".join([format_num(e) for e in la.eigvals(Wo3)]))

t_sym = sp.Symbol('t')
f_sym = 2 * sp.exp(-2*t_sym) * sp.cos(3*t_sym) + 1 * sp.exp(-2*t_sym) * sp.sin(3*t_sym)
F3 = np.array([[float(f_sym.subs(t_sym, 0))],
               [float(sp.diff(f_sym, t_sym).subs(t_sym, 0))],
               [float(sp.diff(f_sym, t_sym, 2).subs(t_sym, 0))]])
x0_3 = la.solve(V3, F3)
add_var("T3_x0", mat2typst(x0_3))

sol3 = solve_ivp(lambda t, x: A3@x, [0, t1_time], x0_3.flatten(), t_eval=time1)
y3 = C3 @ sol3.y
f_vals = 2*np.exp(-2*time1)*np.cos(3*time1) + 1*np.exp(-2*time1)*np.sin(3*time1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time1, y3[0], label='$y(t)$', lw=2)
plt.plot(time1, f_vals, '--', label='$f(t)$')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Выход y(t) и f(t)'); plt.xlabel('Время t, с')
plt.subplot(1, 2, 2)
plt.plot(time1, y3[0] - f_vals, 'r', label='$e(t) = y(t) - f(t)$')
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Ошибка e(t)'); plt.xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task3_sim.png'), dpi=300); plt.close()

# ------------------ Задание 4 ------------------
# Таблица 5, условие №4
C4 = np.array([[0, 1, 1]])
CA4 = C4@A3; CA2_4 = C4@A3@A3
V4 = np.vstack([C4, CA4, CA2_4])
add_var("T4_C", mat2typst(C4)); add_var("T4_CA", mat2typst(CA4)); add_var("T4_CA2", mat2typst(CA2_4))
add_var("T4_V", mat2typst(V4)); add_var("T4_rank_V", str(np.linalg.matrix_rank(V4)))

x0_4_base = la.pinv(V4) @ F3
null_V4 = la.null_space(V4)
x0_4_alt1 = x0_4_base + null_V4 @ [[1]] if null_V4.size > 0 else x0_4_base
x0_4_alt2 = x0_4_base + null_V4 @ [[-1]] if null_V4.size > 0 else x0_4_base
x0_4_alt3 = x0_4_base + null_V4 @ [[2]] if null_V4.size > 0 else x0_4_base

add_var("T4_x0_base", mat2typst(x0_4_base)); add_var("T4_x0_alt1", mat2typst(x0_4_alt1)); add_var("T4_x0_alt2", mat2typst(x0_4_alt2)); add_var("T4_x0_alt3", mat2typst(x0_4_alt3))

labels4 = ['Н.У. 0', 'Н.У. 1', 'Н.У. 2', 'Н.У. 3']
sols4 = [solve_ivp(lambda t, x: A3@x, [0, t1_time], x0_val.flatten(), t_eval=time1)
         for x0_val in [x0_4_base, x0_4_alt1, x0_4_alt2, x0_4_alt3]]
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for i, sol in enumerate(sols4):
    plt.plot(time1, sol.y[0], label=f'$x_1$, {labels4[i]}', alpha=0.8)
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Компонента $x_1(t)$ при разных Н.У.'); plt.xlabel('Время t, с')
plt.subplot(1, 2, 2)
for i, sol in enumerate(sols4):
    plt.plot(time1, (C4 @ sol.y)[0], label=labels4[i], ls=['-','--','-.',':'][i], alpha=0.9)
plt.legend(loc='best', framealpha=0.9); plt.grid(True)
plt.title('Выход y(t) (совпадает при всех Н.У.)'); plt.xlabel('Время t, с')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task4_sim.png'), dpi=300); plt.close()

# Hautus для Задания 4 (наблюдаемость с C4)
T4_H_ranks = []
for i, eig in enumerate(eigvals3):
    H = np.vstack([eig * np.eye(3) - A3, C4])
    T4_H_ranks.append(str(np.linalg.matrix_rank(H)))
    add_var(f"T4_H{i+1}", mat2typst(H))
add_var("T4_H_ranks", ", ".join(T4_H_ranks))

# Грамиан наблюдаемости с C4
Wo4 = compute_Wc(A3.T, C4.T, t1_time)
add_var("T4_Wo", mat2typst(Wo4))
add_var("T4_Wo_eigs", ", ".join([format_num(e) for e in la.eigvals(Wo4)]))

# C_hat для C4 в жордановом базисе
T4_C_hat = C4 @ T3_P_j
add_var("T4_C_hat", mat2typst(T4_C_hat))

# ------------------ Задание 5 ------------------
# Таблица 6, условие №14
B5 = np.array([[-1], [3], [1]]); C5 = np.array([[0, -4, -3], [0, -8, -6]])
CB5 = C5@B5; CAB5 = C5@(A1@B5); CA2B5 = C5@(A1@A1@B5)
U5 = np.hstack([B5, A1@B5, A1@A1@B5])
Uy = C5 @ U5
add_var("T5_B", mat2typst(B5)); add_var("T5_C", mat2typst(C5))
add_var("T5_CB", mat2typst(CB5)); add_var("T5_CAB", mat2typst(CAB5)); add_var("T5_CA2B", mat2typst(CA2B5))
add_var("T5_Uy", mat2typst(Uy)); add_var("T5_rank_Uy", str(np.linalg.matrix_rank(Uy)))
D_prop = np.array([[1], [0]]) if np.linalg.matrix_rank(np.hstack([Uy, np.array([[1], [0]])])) == 2 else np.array([[0], [1]])
add_var("T5_D_proposed", mat2typst(D_prop))

# Управляемость собственных чисел с B5
T5_ctrl_ranks = []
for i, eig in enumerate(eigvals1):
    H = np.hstack([eig * np.eye(3) - A1, B5])
    T5_ctrl_ranks.append(str(np.linalg.matrix_rank(H)))
add_var("T5_H_ctrl_ranks", ", ".join(T5_ctrl_ranks))

# Наблюдаемость собственных чисел с C5
T5_obs_ranks = []
for i, eig in enumerate(eigvals1):
    H = np.vstack([eig * np.eye(3) - A1, C5])
    T5_obs_ranks.append(str(np.linalg.matrix_rank(H)))
add_var("T5_H_obs_ranks", ", ".join(T5_obs_ranks))

# Проверка D
Uy_D = np.hstack([Uy, D_prop])
add_var("T5_rank_Uy_D", str(np.linalg.matrix_rank(Uy_D)))

# B_hat и C_hat для Task 5 в жордановом базисе
T5_B_hat = T1_P_inv_j @ B5
T5_C_hat = C5 @ T1_P_j
add_var("T5_B_hat", mat2typst(T5_B_hat))
add_var("T5_C_hat", mat2typst(T5_C_hat))

# Моделирование: реакция на u(t)=1, сравнение без D и с D
time5 = np.linspace(0, 5, 500)
sol5 = solve_ivp(lambda t, x: A1@x + B5.flatten()*1.0, [0, 5], np.zeros(3), t_eval=time5)
y5_no_d = C5 @ sol5.y
y5_with_d = C5 @ sol5.y + D_prop * 1.0

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time5, y5_no_d[0], label='$y_1$'); plt.plot(time5, y5_no_d[1], label='$y_2$')
plt.legend(); plt.grid(True); plt.title('Выход y(t) при D=0, u(t)=1')
plt.xlabel('Время t, с'); plt.ylabel('y(t)')
plt.subplot(1, 2, 2)
plt.plot(time5, y5_with_d[0], label='$y_1$'); plt.plot(time5, y5_with_d[1], label='$y_2$')
plt.legend(); plt.grid(True); plt.title('Выход y(t) с матрицей D, u(t)=1')
plt.xlabel('Время t, с'); plt.ylabel('y(t)')
plt.tight_layout(); plt.savefig(os.path.join(img_dir, 'task5_sim.png'), dpi=300); plt.close()

# Запись переменных в Typst
with open(vars_file, "w", encoding="utf-8") as f:
    for k, v in variables.items():
        f.write(f"#let {k} = {v}\n" if str(v).startswith("$") else f'#let {k} = "{v}"\n')
print("Расчеты завершены! Изображения сохранены в images/, переменные сохранены в variables.typ")
