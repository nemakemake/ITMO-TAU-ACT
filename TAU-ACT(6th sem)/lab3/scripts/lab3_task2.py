import os
import numpy as np
import scipy.linalg as la
import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cp

np.set_printoptions(precision=6, suppress=True)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def synthesize_K_LMI(A, B, alpha):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    eps = 1e-6
    LMI = P @ A.T + A @ P + 2 * alpha * P + Y.T @ B.T + B @ Y
    constraints = [P >> eps * np.eye(n), LMI << -eps * np.eye(n)]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        prob.solve(solver=cp.SCS, max_iters=50000, eps=1e-9)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"LMI solver failed: {prob.status}")
    return Y.value @ np.linalg.inv(P.value)

def synthesize_L_LMI(A, C, alpha):
    # Дуальная задача: A^T, C^T
    K_dual = synthesize_K_LMI(A.T, C.T, alpha)
    return K_dual.T

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'images')
    ensure_dir(images_dir)

    # Вариант 24, условие №4 (Таблица 4)
    A = np.array([[5, -7, -5, 1],
                  [-7, 5, -1, 5],
                  [-5, -1,  5, 7],
                  [1,  5,  7, 5]], dtype=float)
    B = np.array([[5], [7], [1], [9]], dtype=float)
    C = np.array([[0, 0, 2, 2],
                  [1, 1, -1, 1]], dtype=float)
    n = A.shape[0]

    print("=" * 60)
    print("ЗАДАНИЕ 2: Управление по выходу с заданной степенью устойчивости")
    print("=" * 60)

    # 1. Собственные числа
    eigs_A = np.linalg.eigvals(A)
    print(f"Собственные числа A: {eigs_A}")

    # Управляемость
    U = ctrl.ctrb(A, B)
    rank_U = np.linalg.matrix_rank(U)
    print(f"Ранг матрицы управляемости: {rank_U} (из {n})")

    # Наблюдаемость
    V = ctrl.obsv(A, C)
    rank_V = np.linalg.matrix_rank(V)
    print(f"Ранг матрицы наблюдаемости: {rank_V} (из {n})")

    # Управляемость мод по Хаутусу
    print("\nУправляемость мод (Хаутус):")
    uncontrollable = []
    for lam in eigs_A:
        H = np.hstack([lam * np.eye(n) - A, B])
        r = np.linalg.matrix_rank(H)
        flag = r == n
        print(f"  lambda={lam:.4f}: rank={r}, {'управл.' if flag else 'НЕУПРАВЛ.'}")
        if not flag:
            uncontrollable.append(lam)

    # Наблюдаемость мод по Хаутусу
    print("\nНаблюдаемость мод (Хаутус):")
    unobservable = []
    for lam in eigs_A:
        H = np.vstack([lam * np.eye(n) - A, C])
        r = np.linalg.matrix_rank(H)
        flag = r == n
        print(f"  lambda={lam:.4f}: rank={r}, {'наблюд.' if flag else 'НЕНАБЛЮД.'}")
        if not flag:
            unobservable.append(lam)

    # 2. Достижимые степени
    if len(uncontrollable) == 0:
        print("\nСистема полностью управляема => alpha_K любое.")
        alpha_K_max = None
    else:
        alpha_K_max = min(-np.real(lam) for lam in uncontrollable)
        print(f"\nalpha_K_max = {alpha_K_max:.4f}")

    if len(unobservable) == 0:
        print("Система полностью наблюдаема => alpha_L любое.")
        alpha_L_max = None
    else:
        alpha_L_max = min(-np.real(lam) for lam in unobservable)
        print(f"alpha_L_max = {alpha_L_max:.4f}")

    # 5. Выбор значений alpha
    if alpha_K_max is not None:
        aK_base  = alpha_K_max * 0.5
        aK_high  = alpha_K_max
    else:
        aK_base  = 4.0
        aK_high  = 8.0
    if alpha_L_max is not None:
        aL_low   = alpha_L_max * 0.25   # заведомо < aK_high → обеспечивает aK > aL
        aL_high  = alpha_L_max - 2.0    # достаточно близко к границе
    else:
        aL_low   = 2.0
        aL_high  = 8.0

    # Формируем наборы: a) aK=aL, b) aK>aL, c) aK<aL
    pairs = [
        (aK_base,  aK_base),   # aK = aL
        (aK_high,  aL_low),    # aK > aL  ← исправлено
        (aK_base,  aL_high),   # aK < aL
    ]
    print(f"\nНаборы (alpha_K, alpha_L): {pairs}")

    x0 = np.array([1, 1, 1, 1])
    xh0 = np.array([0, 0, 0, 0])
    t_sim = np.linspace(0, 5, 2000)

    for idx, (aK, aL) in enumerate(pairs):
        print(f"\n{'='*40}")
        print(f"Набор {idx+1}: alpha_K={aK}, alpha_L={aL}")
        print(f"{'='*40}")

        # Синтез K
        K = synthesize_K_LMI(A, B, aK)
        print(f"K = {K}")
        eigs_K = np.linalg.eigvals(A + B @ K)
        print(f"sigma(A+BK) = {eigs_K}")
        print(f"max Re(A+BK) = {np.max(np.real(eigs_K)):.6f} <= {-aK}")

        # Синтез L
        L = synthesize_L_LMI(A, C, aL)
        print(f"L =\n{L}")
        eigs_L = np.linalg.eigvals(A + L @ C)
        print(f"sigma(A+LC) = {eigs_L}")
        print(f"max Re(A+LC) = {np.max(np.real(eigs_L)):.6f} <= {-aL}")

        # Замкнутая система: dx/dt = Ax + BKx_hat
        # dx_hat/dt = (A+BK+LC)x_hat - LCx
        # Переменная состояния: [x; x_hat]
        A_cl = np.block([
            [A, B @ K],
            [-L @ C, A + B @ K + L @ C]
        ])

        X0 = np.concatenate([x0, xh0])
        sys_cl = ctrl.StateSpace(A_cl, np.zeros((2*n, 1)), np.eye(2*n), np.zeros((2*n, 1)))
        t, y_all = ctrl.initial_response(sys_cl, T=t_sim, X0=X0)

        x = y_all[:n, :]
        x_hat = y_all[n:, :]
        e = x - x_hat
        u = (K @ x_hat).flatten()

        print(f"max|u(t)| = {np.max(np.abs(u)):.4f}, u(0) = {u[0]:.4f}")

        # График u(t)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, u)
        ax.set_xlabel('t, с')
        ax.set_ylabel('u(t)')
        ax.set_title(f'Управление u(t), (aK={aK:g}, aL={aL:g})')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(images_dir, f'task2_u_{idx+1}.png'), dpi=200)
        plt.close(fig)

        # Графики x_i и x_hat_i
        for i in range(n):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, x[i], label=f'$x_{i+1}$', linewidth=1.5)
            ax.plot(t, x_hat[i], '--', label=f'$\\hat{{x}}_{i+1}$', linewidth=1.5)
            ax.set_ylabel(f'$x_{i+1}, \\hat{{x}}_{i+1}$')
            ax.set_xlabel('t, с')
            ax.set_title(f'Сравнение x_{i+1}(t) и x_hat_{i+1}(t), (aK={aK:g}, aL={aL:g})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(images_dir, f'task2_x_{idx+1}_{i+1}.png'), dpi=200)
            plt.close(fig)

        # График e(t) (отдельно для каждой компоненты)
        for i in range(n):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, e[i], label=f'$e_{i+1}$')
            ax.set_xlabel('t, с')
            ax.set_ylabel(f'e_{i+1}(t)')
            ax.set_title(f'Ошибка наблюдателя e_{i+1}(t), (aK={aK:g}, aL={aL:g})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(images_dir, f'task2_e_{idx+1}_{i+1}.png'), dpi=200)
            plt.close(fig)

    print("\nГрафики задания 2 сохранены.")

if __name__ == "__main__":
    main()
