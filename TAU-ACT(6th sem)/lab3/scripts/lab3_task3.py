import os
import numpy as np
import scipy.linalg as la
import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

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
    m = B.shape[1]

    print("=" * 60)
    print("ЗАДАНИЕ 3: Регулятор с качественной экспоненциальной устойчивостью")
    print("=" * 60)

    eigs_A = np.linalg.eigvals(A)
    print(f"Собственные числа A: {eigs_A}")

    # Управляемость мод
    U = ctrl.ctrb(A, B)
    rank_U = np.linalg.matrix_rank(U)
    print(f"Ранг матрицы управляемости: {rank_U} (из {n})")

    uncontrollable = []
    for lam in eigs_A:
        H = np.hstack([lam * np.eye(n) - A, B])
        r = np.linalg.matrix_rank(H)
        if r < n:
            uncontrollable.append(lam)

    # 2. Выбор beta и r
    # beta < 0, |beta| — средняя степень устойчивости
    # r > 0, beta + r < 0, r = |beta|/k, 1.5 <= k <= 4
    # Неуправляемые моды должны лежать в круге (beta, r)
    # Для нашей системы нужно учесть неуправляемые моды

    if len(uncontrollable) > 0:
        # Неуправляемые моды ограничивают выбор beta и r:
        # каждая неуправляемая мода lambda должна быть в круге |lambda - beta| < r
        # Выберем beta по модулю немного больше Re неуправляемых мод
        unc_re = [np.real(lam) for lam in uncontrollable]
        unc_im = [np.imag(lam) for lam in uncontrollable]
        # Центр beta по реальной оси должен быть таким, чтобы моды попали в круг
        beta = -3.0
        k = 2.0  # r = |beta|/k
        r = abs(beta) / k
        # Проверим, что неуправляемые моды в круге
        for lam in uncontrollable:
            dist = abs(lam - beta)
            print(f"  Неуправляемая мода {lam:.4f}: расстояние от beta={beta} равно {dist:.4f}, r={r:.4f}, "
                  f"{'OK' if dist < r else 'НЕ попадает!'}")
    else:
        beta = -3.0
        k = 2.0
        r = abs(beta) / k

    print(f"\nВыбрано: beta = {beta}, r = {r}, k = {k}")
    print(f"Проверка: beta + r = {beta + r} < 0: {beta + r < 0}")

    # 3. Четыре набора Q, R
    # Уравнение Риккати:
    # (A+BK-betaI)^T P (A+BK-betaI) - r^2 P = -Q
    # K = -(R + B^T P B)^{-1} B^T P (A - beta I)
    # Сводим к DARE: A_d = (A-betaI)/r, B_d = B/r
    # Стандартная DARE: A_d^T P A_d - P - A_d^T P B_d (R + B_d^T P B_d)^-1 B_d^T P A_d + Q = 0

    A_beta = A - beta * np.eye(n)

    sets = [
        ("Q=I, R=1", np.eye(n), 1.0),
        ("Q=I, R=0", np.eye(n), 0.0),
        ("Q=0, R=1", np.zeros((n, n)), 1.0),
        ("Q=0, R=0", np.zeros((n, n)), 0.0),
    ]

    t_sim = np.linspace(0, 6, 2000)
    x0 = np.array([1, 1, 1])

    fig_u, ax_u = plt.subplots(figsize=(10, 5))
    fig_x, ax_x = plt.subplots(figsize=(10, 5))
    fig_poles, ax_poles = plt.subplots(figsize=(8, 8))

    circle = plt.Circle((beta, 0), r, color='blue', fill=False, linestyle='--', linewidth=2, label='Круг (beta, r)')
    ax_poles.add_patch(circle)
    ax_poles.axhline(0, color='black', lw=0.5)
    ax_poles.axvline(0, color='black', lw=0.5)

    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']
    markers = ['o', 's', '^', 'D']

    results_table = []

    for idx, (name, Q, R) in enumerate(sets):
        print(f"\n--- {name} ---")

        # Особый случай Q=0, R=0: прямой DARE не сходится,
        # синтез через управляемую подсистему с размещением полюсов на границе круга
        if np.allclose(Q, 0) and R == 0:
            U_ctrl = ctrl.ctrb(A, B)
            U_svd, _, _ = np.linalg.svd(U_ctrl)
            T_ctrl = U_svd
            T_ctrl_inv = np.linalg.inv(T_ctrl)
            Ac = (T_ctrl_inv @ A @ T_ctrl)[:2, :2]
            Bc = (T_ctrl_inv @ B)[:2, :]
            poles_c = [beta - r, beta + r]  # на границе круга
            Kc = -ctrl.place(Ac, Bc, poles_c)
            Kt = np.hstack([Kc, np.zeros((1, 1))])
            K = Kt @ T_ctrl_inv
        else:
            eps_reg = 1e-8
            Q_eff = Q.copy()
            R_eff = R

            # Регуляризация для вырожденных случаев
            if np.allclose(Q, 0):
                Q_eff = eps_reg * np.eye(n)
            elif R == 0:
                R_eff = eps_reg

            A_d = A_beta / r
            B_d = B / r
            Q_d = Q_eff
            R_d = np.array([[R_eff]])

            P = la.solve_discrete_are(A_d, B_d, Q_d, R_d)
            M = np.array([[R_eff]]) + B.T @ P @ B
            K = -np.linalg.inv(M) @ B.T @ P @ A_beta

        print(f"K = {K}")
        A_cl = A + B @ K
        eigs_cl = np.linalg.eigvals(A_cl)
        print(f"sigma(A+BK) = {eigs_cl}")

        # Проверка: все в круге?
        for lam in eigs_cl:
            dist = abs(lam - beta)
            ok = dist <= r + 1e-6
            print(f"  lambda={lam:.6f}: |lambda-beta|={dist:.6f}, r={r:.4f}, {'OK' if ok else 'OUTSIDE'}")

        ax_poles.scatter(np.real(eigs_cl), np.imag(eigs_cl),
                       color=colors[idx], marker=markers[idx], s=80, zorder=5, label=name)

        # Моделирование
        sys_cl = ctrl.StateSpace(A_cl, np.zeros((n, 1)), np.eye(n), np.zeros((n, 1)))
        t, y = ctrl.initial_response(sys_cl, T=t_sim, X0=x0)
        u = (K @ y).flatten()
        norm_x = np.linalg.norm(y, axis=0)

        ax_x.plot(t, norm_x, label=name, color=colors[idx])
        ax_u.plot(t, u, label=name, color=colors[idx])

        u0 = (K @ x0).item()
        max_u = np.max(np.abs(u))
        max_re = np.max(np.real(eigs_cl))
        results_table.append((name, K, eigs_cl, max_re, u0, max_u))

        # Отдельный график x(t) по компонентам
        fig_xi, ax_xi = plt.subplots(figsize=(10, 5))
        for i in range(n):
            ax_xi.plot(t, y[i], label=f'$x_{i+1}(t)$')
        ax_xi.set_xlabel('t, с')
        ax_xi.set_ylabel('x(t)')
        ax_xi.set_title(f'Состояние x(t), {name}')
        ax_xi.grid(True)
        ax_xi.legend()
        fig_xi.tight_layout()
        fig_xi.savefig(os.path.join(images_dir, f'task3_x_{idx+1}.png'), dpi=200)
        plt.close(fig_xi)

    ax_u.set_xlabel('t, с')
    ax_u.set_ylabel('u(t)')
    ax_u.set_title('Управление u(t) для разных наборов (Q, R)')
    ax_u.grid(True)
    ax_u.legend()
    fig_u.tight_layout()
    fig_u.savefig(os.path.join(images_dir, 'task3_u.png'), dpi=200)
    plt.close(fig_u)

    ax_x.set_xlabel('t, с')
    ax_x.set_ylabel('||x(t)||')
    ax_x.set_title('Норма вектора состояния ||x(t)||')
    ax_x.grid(True)
    ax_x.legend()
    fig_x.tight_layout()
    fig_x.savefig(os.path.join(images_dir, 'task3_xnorm.png'), dpi=200)
    plt.close(fig_x)

    ax_poles.set_xlabel('Re')
    ax_poles.set_ylabel('Im')
    ax_poles.set_title('Собственные числа на комплексной плоскости')
    ax_poles.grid(True)
    ax_poles.legend()
    ax_poles.set_aspect('equal', adjustable='datalim')
    margin = r + 1.5
    ax_poles.set_xlim(beta - margin, beta + margin)
    ax_poles.set_ylim(-margin, margin)
    fig_poles.tight_layout()
    fig_poles.savefig(os.path.join(images_dir, 'task3_poles.png'), dpi=200)
    plt.close(fig_poles)

    # Сводная таблица
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА (Задание 3)")
    print("=" * 60)
    print(f"{'Nabor':<16} {'max Re(lam)':<14} {'u(0)':<14} {'max|u|':<14}")
    for name, K, eigs, max_re, u0, max_u in results_table:
        print(f"{name:<16} {max_re:<14.6f} {u0:<14.6f} {max_u:<14.6f}")

    print("\nГрафики задания 3 сохранены.")

if __name__ == "__main__":
    main()
