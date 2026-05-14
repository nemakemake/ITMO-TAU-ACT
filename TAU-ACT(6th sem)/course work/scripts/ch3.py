import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import control as ct
import matplotlib.pyplot as plt
import os

os.makedirs('images', exist_ok=True)

# Параметры из первой главы
np.random.seed(24)
M = np.random.randint(100000, 1000001) / 1000 / np.sqrt(2)
m1 = np.random.randint(1000, 10001) / 1000 * np.sqrt(3)
m2 = np.random.randint(1000, 10001) / 1000 * np.sqrt(3)
l = np.random.randint(100, 1001) / np.sqrt(5) / 100
g = 9.81

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

A = np.array([[0, 1, 0, 0], [0, 0, a23, 0], [0, 0, 0, 1], [0, 0, a43, 0]])
B = np.array([[0], [b2], [0], [b4]])
D = np.array([[0], [d2], [0], [d4]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

# 3.1 Модальный регулятор по состоянию
def place_poles(A, B, poles):
    res = sig.place_poles(A, B, poles)
    return res.gain_matrix

poles_norm = [-3, -4, -5, -6]
poles_weak = [-0.5, -0.6, -0.7, -0.8]
poles_strong = [-10, -12, -15, -20]

K_norm = place_poles(A, B, poles_norm)
K_weak = place_poles(A, B, poles_weak)
K_strong = place_poles(A, B, poles_strong)

print("K (нормальный спектр):", np.round(K_norm, 2))
print("K (слабый спектр):", np.round(K_weak, 2))
print("K (сильный спектр):", np.round(K_strong, 2))

# 3.2 Наблюдатель
def place_observer(A, C, poles):
    # L^T = place_poles(A^T, C^T, poles)
    res = sig.place_poles(A.T, C.T, poles)
    return res.gain_matrix.T

obs_poles_norm = [-15, -16, -17, -18]
obs_poles_slow = [-2, -2.5, -3, -3.5]
obs_poles_fast = [-30, -40, -50, -60]

L_norm = place_observer(A, C, obs_poles_norm)
L_slow = place_observer(A, C, obs_poles_slow)
L_fast = place_observer(A, C, obs_poles_fast)

print("\nL (нормальный наблюдатель):\n", np.round(L_norm, 2))

# Наблюдатель пониженной размерности (нормальный)
# Полюса для -L_red = [-15, -16]
L_red_norm = np.array([[15, 0], [0, 16]])
L_red_slow = np.array([[2, 0], [0, 3]])
L_red_fast = np.array([[30, 0], [0, 40]])

# Моделирование
from scipy.integrate import solve_ivp

def get_nonlinear_sys_state_fb(K, x0):
    def sys(t, x):
        u = - (K @ x)[0]
        f = 0
        x1, x2, x3, x4 = x
        den = M_t * L_t - S_t**2 * np.cos(x3)**2
        dx1 = x2
        dx2 = (L_t * (u + S_t * x4**2 * np.np.sin(x3)) - S_t * np.cos(x3) * (f + S_t * g * np.sin(x3))) / den
        dx3 = x4
        dx4 = (M_t * (f + S_t * g * np.sin(x3)) - S_t * np.cos(x3) * (u + S_t * x4**2 * np.sin(x3))) / den
        return [dx1, dx2, dx3, dx4]
    return sys

def sim_state_fb(K, x0, t_end=5):
    def sys(t, x):
        u = - (K @ x)[0]
        # Ограничение управления для реалистичности
        if u > 500: u = 500
        if u < -500: u = -500
        f = 0
        x1, x2, x3, x4 = x
        den = M_t * L_t - S_t**2 * np.cos(x3)**2
        dx1 = x2
        dx2 = (L_t * (u + S_t * x4**2 * np.sin(x3)) - S_t * np.cos(x3) * (f + S_t * g * np.sin(x3))) / den
        dx3 = x4
        dx4 = (M_t * (f + S_t * g * np.sin(x3)) - S_t * np.cos(x3) * (u + S_t * x4**2 * np.sin(x3))) / den
        return [dx1, dx2, dx3, dx4]
    sol = solve_ivp(sys, (0, t_end), x0, t_eval=np.linspace(0, t_end, 500))
    return sol

# 3.1.1 Исследование влияния начальных условий
x0_list = [
    [0.1, 0, 0, 0],        # небольшое по тележке
    [1.0, 0, 0, 0],        # большое по тележке
    [0, 0, 0.2, 0],        # небольшой угол
    [0, 0, 1.0, 0],        # большой угол (около 60 градусов)
    [1.0, 0, 1.0, 0]       # одновременно большое отклонение
]

for i, x0 in enumerate(x0_list):
    sol = sim_state_fb(K_norm, x0, 8)
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, sol.y[0], label='$a(t)$')
    plt.plot(sol.t, sol.y[2], label='$\\varphi(t)$')
    plt.title(f'Регулятор по состоянию (нормальный), x0 = {x0}')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/ch3_ic_{i+1}.png', dpi=300)
    plt.close()

# 3.1.2 Влияние желаемого спектра
# Возьмем x0 = [0.5, 0, 0.3, 0]
x0_test = [0.5, 0, 0.3, 0]
for name, K_mat in [("weak", K_weak), ("norm", K_norm), ("strong", K_strong)]:
    sol = sim_state_fb(K_mat, x0_test, 10)
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, sol.y[0], label='$a(t)$')
    plt.plot(sol.t, sol.y[2], label='$\\varphi(t)$')
    plt.title(f'Регулятор ({name}), x0 = {x0_test}')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/ch3_spec_{name}.png', dpi=300)
    plt.close()

# 3.2 и 3.3 Регулятор по выходу с наблюдателем (полный и пониженный)
# Чтобы была видна работа наблюдателя, дадим внешнее гармоническое воздействие g(t) = 0.5 * sin(5t) (на объект как момент f)
def sim_output_fb(K, L, x0, obs0, t_end=10, is_reduced=False, L_red=None):
    def sys(t, state):
        x = state[:4]
        if is_reduced:
            z = state[4:6]
            y = np.array([x[0], x[2]])
            x_hat_b = z + L_red @ y
            x_hat = np.array([x[0], x_hat_b[0], x[2], x_hat_b[1]])
        else:
            x_hat = state[4:8]
            y = C @ x
        
        u = - (K @ x_hat)[0]
        if u > 500: u = 500
        if u < -500: u = -500
        
        f = 0.5 * np.sin(5*t)
        
        x1, x2, x3, x4 = x
        den = M_t * L_t - S_t**2 * np.cos(x3)**2
        dx1 = x2
        dx2 = (L_t * (u + S_t * x4**2 * np.sin(x3)) - S_t * np.cos(x3) * (f + S_t * g * np.sin(x3))) / den
        dx3 = x4
        dx4 = (M_t * (f + S_t * g * np.sin(x3)) - S_t * np.cos(x3) * (u + S_t * x4**2 * np.sin(x3))) / den
        dx = [dx1, dx2, dx3, dx4]
        
        if is_reduced:
            # dot_z = -L_red z + A_21 y + (-L_red)*L_red y + B_2 u
            # A21 = [[0, a23], [0, a43]], B2 = [[b2], [b4]]
            A21 = np.array([[0, a23], [0, a43]])
            B2 = np.array([b2, b4])
            dz = -L_red @ z + A21 @ y - L_red @ (L_red @ y) + B2 * u
            dstate = np.concatenate((dx, dz))
        else:
            dx_hat = A @ x_hat + B.flatten() * u + L @ (y - C @ x_hat)
            dstate = np.concatenate((dx, dx_hat))
            
        return dstate
        
    if is_reduced:
        y0 = np.array([x0[0], x0[2]])
        # obs0 for reduced is z0 = x_hat_b0 - L_red y0
        z0 = np.array([obs0[1], obs0[3]]) - L_red @ y0
        state0 = np.concatenate((x0, z0))
    else:
        state0 = np.concatenate((x0, obs0))
        
    sol = solve_ivp(sys, (0, t_end), state0, t_eval=np.linspace(0, t_end, 1000))
    return sol

x0_obs = [0.1, 0.0, 0.1, 0.0]
obs0_init = [0.0, 0.0, 0.0, 0.0]

# Полный порядок - нормальный наблюдатель
sol_full = sim_output_fb(K_norm, L_norm, x0_obs, obs0_init)
plt.figure(figsize=(10, 4))
plt.plot(sol_full.t, sol_full.y[2], '-', label='Истинный $\\varphi$')
plt.plot(sol_full.t, sol_full.y[6], '--', label='Оценка $\\hat{\\varphi}$')
plt.title('Наблюдатель полного порядка (нормальный)')
plt.legend()
plt.grid()
plt.savefig('images/ch3_obs_full.png', dpi=300)
plt.close()

# Пониженный порядок - нормальный наблюдатель
sol_red = sim_output_fb(K_norm, None, x0_obs, obs0_init, is_reduced=True, L_red=L_red_norm)
plt.figure(figsize=(10, 4))
plt.plot(sol_red.t, sol_red.y[3], '-', label='Истинная $\\dot{\\varphi}$')
# Для пониженного оценка скорости x4 (индекс 3) это z_2 + l_2 y_2 = sol_red.y[5] + 16 * sol_red.y[2]
x4_hat = sol_red.y[5] + 16 * sol_red.y[2]
plt.plot(sol_red.t, x4_hat, '--', label='Оценка $\\hat{\\dot{\\varphi}}$')
plt.title('Наблюдатель пониженного порядка (нормальный)')
plt.legend()
plt.grid()
plt.savefig('images/ch3_obs_red.png', dpi=300)
plt.close()

# 3.3.1 Влияние желаемых спектров наблюдателя и регулятора
# Сценарий 1: Сильный регулятор, медленный наблюдатель
sol_strong_slow = sim_output_fb(K_strong, L_slow, x0_obs, obs0_init)
plt.figure(figsize=(10, 4))
plt.plot(sol_strong_slow.t, sol_strong_slow.y[0], label='$a(t)$')
plt.plot(sol_strong_slow.t, sol_strong_slow.y[2], label='$\\varphi(t)$')
plt.title('Сильный регулятор + Медленный наблюдатель')
plt.legend()
plt.grid()
plt.savefig('images/ch3_out_strong_slow.png', dpi=300)
plt.close()

# Сценарий 2: Слабый регулятор, быстрый наблюдатель
sol_weak_fast = sim_output_fb(K_weak, L_fast, x0_obs, obs0_init)
plt.figure(figsize=(10, 4))
plt.plot(sol_weak_fast.t, sol_weak_fast.y[0], label='$a(t)$')
plt.plot(sol_weak_fast.t, sol_weak_fast.y[2], label='$\\varphi(t)$')
plt.title('Слабый регулятор + Быстрый наблюдатель')
plt.legend()
plt.grid()
plt.savefig('images/ch3_out_weak_fast.png', dpi=300)
plt.close()

# Сценарий 3: Равные по силе (нормальный + нормальный)
plt.figure(figsize=(10, 4))
plt.plot(sol_full.t, sol_full.y[0], label='$a(t)$')
plt.plot(sol_full.t, sol_full.y[2], label='$\\varphi(t)$')
plt.title('Нормальный регулятор + Нормальный наблюдатель')
plt.legend()
plt.grid()
plt.savefig('images/ch3_out_norm_norm.png', dpi=300)
plt.close()

print("\n=== ВЫПОЛНЕНО ===")
