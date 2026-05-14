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

def synthesize_LMI(A, B, alpha, minimize_control=False):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    eps = 1e-6
    LMI = P @ A.T + A @ P + 2 * alpha * P + Y.T @ B.T + B @ Y
    constraints = [P >> eps * np.eye(n), LMI << -eps * np.eye(n)]
    if minimize_control:
        Z = cp.Variable((m, m), symmetric=True)
        block = cp.bmat([[Z, Y], [Y.T, P]])
        constraints.append(block >> 0)
        objective = cp.Minimize(cp.trace(Z))
    else:
        objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        prob.solve(solver=cp.SCS, max_iters=50000, eps=1e-9)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"LMI solver failed: {prob.status}")
    return Y.value @ np.linalg.inv(P.value)

def synthesize_ARE(A, B, alpha, Q, R_val, nu=2):
    n = A.shape[0]
    A_hat = A + alpha * np.eye(n)
    B_hat = np.sqrt(nu) * B
    m = B.shape[1]
    R_hat = R_val * np.eye(m) if np.isscalar(R_val) else R_val
    P = la.solve_continuous_are(A_hat, B_hat, Q, R_hat)
    R_inv = (1.0 / R_val) * np.eye(m) if np.isscalar(R_val) else np.linalg.inv(R_val)
    K = -R_inv @ B.T @ P
    return K

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'images')
    ensure_dir(images_dir)

    # Вариант 24, условие №14 (Таблица 3)
    A = np.array([[12, -1, 14],
                  [6,   0,  6],
                  [-6, -2, -8]], dtype=float)
    B = np.array([[11], [7], [-7]], dtype=float)
    n = A.shape[0]

    # 1. Собственные числа
    eigs_A = np.linalg.eigvals(A)
    print("=" * 60)
    print("ЗАДАНИЕ 1: Синтез регулятора с заданной степенью устойчивости")
    print("=" * 60)
    print(f"Собственные числа A: {eigs_A}")

    # Матрица управляемости
    U = ctrl.ctrb(A, B)
    rank_U = np.linalg.matrix_rank(U)
    print(f"Ранг матрицы управляемости: {rank_U} (из {n})")

    # Проверка управляемости каждой моды по Хаутусу
    print("\nПроверка управляемости мод по Хаутусу:")
    uncontrollable_eigs = []
    for lam in eigs_A:
        H = np.hstack([lam * np.eye(n) - A, B])
        r = np.linalg.matrix_rank(H)
        ctrl_flag = r == n
        print(f"  lambda = {lam:.6f}: rank(H) = {r}, {'управляемо' if ctrl_flag else 'НЕУПРАВЛЯЕМО'}")
        if not ctrl_flag:
            uncontrollable_eigs.append(lam)

    # 2. Максимальная степень устойчивости
    if rank_U == n:
        print("\nСистема полностью управляема => любая степень устойчивости достижима.")
        alpha_max = None
    else:
        # Ограничение: Re(lambda_uncontrollable) <= -alpha
        alpha_max = min(-np.real(lam) for lam in uncontrollable_eigs)
        print(f"\nМаксимальная степень устойчивости: alpha_max = {alpha_max:.4f}")
        print(f"Неуправляемые собственные числа: {uncontrollable_eigs}")

    # 4. Выбор alpha
    if alpha_max is not None:
        alpha1 = 1.0
        alpha2 = alpha_max - 1e-4  # вплотную к alpha_max (отобразится как 2.00)
        if alpha1 >= alpha2:
            alpha1 = alpha_max * 0.5
    else:
        alpha1 = 1.0
        alpha2 = 5.0
    alphas = [alpha1, alpha2]
    print(f"\nВыбраны alpha: {alphas}")

    x0 = np.array([1, 1, 1])
    t_sim = np.linspace(0, 6, 2000)

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*40}")
        print(f"alpha = {alpha}")
        print(f"{'='*40}")

        results = {}

        # 5. LMI
        K1 = synthesize_LMI(A, B, alpha, minimize_control=False)
        print(f"K1 (LMI): {K1}")
        eigs1 = np.linalg.eigvals(A + B @ K1)
        print(f"  sigma(A+BK1) = {eigs1}, max Re = {np.max(np.real(eigs1)):.6f}")
        results['K1'] = (K1, eigs1)

        K2 = synthesize_LMI(A, B, alpha, minimize_control=True)
        print(f"K2 (LMI min): {K2}")
        eigs2 = np.linalg.eigvals(A + B @ K2)
        print(f"  sigma(A+BK2) = {eigs2}, max Re = {np.max(np.real(eigs2)):.6f}")
        results['K2'] = (K2, eigs2)

        # 6. Riccati
        K3 = synthesize_ARE(A, B, alpha, Q=np.eye(n), R_val=1.0)
        print(f"K3 (ARE Q=I): {K3}")
        eigs3 = np.linalg.eigvals(A + B @ K3)
        print(f"  sigma(A+BK3) = {eigs3}, max Re = {np.max(np.real(eigs3)):.6f}")
        results['K3'] = (K3, eigs3)

        K4 = synthesize_ARE(A, B, alpha, Q=np.zeros((n, n)), R_val=1.0)
        print(f"K4 (ARE Q=0): {K4}")
        eigs4 = np.linalg.eigvals(A + B @ K4)
        print(f"  sigma(A+BK4) = {eigs4}, max Re = {np.max(np.real(eigs4)):.6f}")
        results['K4'] = (K4, eigs4)

        all_results[alpha] = results

        # Графики u(t) для всех K на одном рисунке
        fig_u, ax_u = plt.subplots(figsize=(10, 5))
        for name in ['K1', 'K2', 'K3', 'K4']:
            K, _ = results[name]
            A_cl = A + B @ K
            sys_cl = ctrl.StateSpace(A_cl, np.zeros((n, 1)), np.eye(n), np.zeros((n, 1)))
            t, y = ctrl.initial_response(sys_cl, T=t_sim, X0=x0)
            u = (K @ y).flatten()
            ax_u.plot(t, u, label=f'${name}$')
        ax_u.set_xlabel('t, с')
        ax_u.set_ylabel('u(t)')
        ax_u.set_title(f'Управление u(t), alpha = {alpha:g}')
        ax_u.grid(True)
        ax_u.legend()
        fig_u.tight_layout()
        fig_u.savefig(os.path.join(images_dir, f'task1_u_a{alpha:.2f}.png'), dpi=200)
        plt.close(fig_u)

        # Графики x(t) для каждого Ki отдельно
        for name in ['K1', 'K2', 'K3', 'K4']:
            K, _ = results[name]
            A_cl = A + B @ K
            sys_cl = ctrl.StateSpace(A_cl, np.zeros((n, 1)), np.eye(n), np.zeros((n, 1)))
            t, y = ctrl.initial_response(sys_cl, T=t_sim, X0=x0)
            fig_xi, ax_xi = plt.subplots(figsize=(10, 5))
            for i in range(n):
                ax_xi.plot(t, y[i], label=f'$x_{i+1}(t)$')
            ax_xi.set_xlabel('t, с')
            ax_xi.set_ylabel('x(t)')
            ax_xi.set_title(f'Состояние x(t), alpha={alpha:g}, {name}')
            ax_xi.grid(True)
            ax_xi.legend()
            fig_xi.tight_layout()
            fig_xi.savefig(os.path.join(images_dir, f'task1_x_a{alpha:.2f}_{name}.png'), dpi=200)
            plt.close(fig_xi)

    # Сводная таблица
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 60)
    for alpha in alphas:
        print(f"\nalpha = {alpha}:")
        print(f"{'Regulator':<12} {'max Re(lam)':<14} {'u(0)':<14} {'max|u(t)|':<14}")
        for name in ['K1', 'K2', 'K3', 'K4']:
            K, eigs = all_results[alpha][name]
            u0 = (K @ x0).item()
            A_cl = A + B @ K
            sys_cl = ctrl.StateSpace(A_cl, np.zeros((n, 1)), np.eye(n), np.zeros((n, 1)))
            t, y = ctrl.initial_response(sys_cl, T=t_sim, X0=x0)
            u_all = (K @ y).flatten()
            max_u = np.max(np.abs(u_all))
            max_re = np.max(np.real(eigs))
            print(f"{name:<12} {max_re:<14.6f} {u0:<14.6f} {max_u:<14.6f}")

    print("\nГрафики сохранены.")

if __name__ == "__main__":
    main()
