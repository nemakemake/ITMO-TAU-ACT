import os
import json
import numpy as np
import scipy.linalg as la
import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.set_printoptions(precision=6, suppress=True)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def fmt_mat(M, prec=6):
    if np.isscalar(M):
        return f"{M:.{prec}f}"
    return np.array2string(np.atleast_2d(M), precision=prec, suppress_small=True)


def solve_francis_g(A, B, C, D, K, Gam_g, Y_g):
    """Решает систему ур-ний Франкиса-Дэвисона:
       P Γ_g - (A+BK) P = B K_g
       (C+DK) P + D K_g = Y_g
    """
    n = A.shape[0]
    m = B.shape[1]
    s = Gam_g.shape[0]
    Acl = A + B @ K
    Ccl = C + D @ K
    I_n = np.eye(n)
    I_s = np.eye(s)
    M11 = np.kron(Gam_g.T, I_n) - np.kron(I_s, Acl)
    M12 = -np.kron(I_s, B)
    M21 = np.kron(I_s, Ccl)
    M22 = np.kron(I_s, D)
    M = np.block([[M11, M12], [M21, M22]])
    rhs = np.concatenate([np.zeros(n * s), Y_g.flatten('F')])
    sol = np.linalg.solve(M, rhs)
    P = sol[:n * s].reshape((n, s), order='F')
    Kg = sol[n * s:].reshape((m, s), order='F')
    return P, Kg


def solve_francis_f(A, B, C, D, B_f, D_f, K, Gam_f, Y_f):
    """Решает систему ур-ний Франкиса-Дэвисона для возмущения:
       P Γ_f - (A+BK) P - B_f Y_f = B K_f
       (C+DK) P + D K_f + D_f Y_f = 0
    """
    n = A.shape[0]
    m = B.shape[1]
    s = Gam_f.shape[0]
    Acl = A + B @ K
    Ccl = C + D @ K
    I_n = np.eye(n)
    I_s = np.eye(s)
    M11 = np.kron(Gam_f.T, I_n) - np.kron(I_s, Acl)
    M12 = -np.kron(I_s, B)
    M21 = np.kron(I_s, Ccl)
    M22 = np.kron(I_s, D)
    M = np.block([[M11, M12], [M21, M22]])
    rhs1 = (B_f @ Y_f).flatten('F')
    rhs2 = -(D_f @ Y_f).flatten('F')
    rhs = np.concatenate([rhs1, rhs2])
    sol = np.linalg.solve(M, rhs)
    P = sol[:n * s].reshape((n, s), order='F')
    Kf = sol[n * s:].reshape((m, s), order='F')
    return P, Kf


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'images')
    ensure_dir(images_dir)

    # =============================================
    # Вариант 24: ОУ № 14, Генераторы № 4
    # =============================================
    A = np.array([[12.0, -1.0, 14.0],
                  [6.0,   0.0,  6.0],
                  [-6.0, -2.0, -8.0]])
    B = np.array([[11.0], [7.0], [-7.0]])
    B_f = np.array([[2.0, -1.0],
                    [0.0,  0.0],
                    [-2.0, 0.0]])
    C = np.array([[-2.0, 2.0, -1.0]])
    D = np.array([[14.0]])
    D_f = np.array([[2.0, 3.0]])

    Gam_f = np.array([[-40.0, 16.0,   9.0,  -7.0],
                      [-64.0, 25.0,  14.0, -12.0],
                      [-26.0, 11.0,   7.0,  -3.0],
                      [ 48.0, -18.0, -14.0,  8.0]])
    Y_f = np.array([[6.0, -6.0, -4.0, -5.0],
                    [-8.0, -2.0, 8.0, -5.0]])
    w_f0 = np.array([1.0, 1.0, 1.0, 1.0])

    # g(t) = 11 cos(t) + 10
    Gam_g = np.array([[0.0,  1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0,  0.0, 0.0]])
    Y_g = np.array([[11.0, 0.0, 10.0]])
    w_g0 = np.array([1.0, 0.0, 1.0])

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    print("=" * 60)
    print("ЗАДАНИЕ 1: Синтез регулятора по Франкису-Дэвисону")
    print("=" * 60)

    # 1) Собственные числа Γ_f
    eigs_Gf = np.linalg.eigvals(Gam_f)
    print(f"\n1) σ(Γ_f) = {eigs_Gf}")

    # 2) Γ_g, Y_g, w_g(0) — построены вручную, σ(Γ_g)
    eigs_Gg = np.linalg.eigvals(Gam_g)
    print(f"\n2) σ(Γ_g) = {eigs_Gg}")
    print(f"   Γ_g = \n{Gam_g}")
    print(f"   Y_g = {Y_g}")
    print(f"   w_g(0) = {w_g0}")

    # Управляемость (А, В) — Калман и Хаутус
    AB = A @ B
    A2B = A @ AB
    U = np.hstack([B, AB, A2B])
    print(f"\n3) AB = {AB.flatten()}, A^2 B = {A2B.flatten()}")
    print(f"   U = \n{U}")
    print(f"   det(U) = {np.linalg.det(U)}, rank(U) = {np.linalg.matrix_rank(U)}")

    eigs_A = np.linalg.eigvals(A)
    print(f"   σ(A) = {eigs_A}")
    for lam in eigs_A:
        H = np.hstack([lam * np.eye(n) - A, B])
        r = np.linalg.matrix_rank(H)
        print(f"   λ = {lam:.4f}: rank(H) = {r}, "
              f"{'управ.' if r == n else 'НЕУПРАВ.'}")

    # Синтез K через LQR (Q=I, R=1)
    Q_lqr = np.eye(n)
    R_lqr = np.array([[1.0]])
    P_lqr = la.solve_continuous_are(A, B, Q_lqr, R_lqr)
    K = -np.linalg.inv(R_lqr) @ B.T @ P_lqr
    Acl = A + B @ K
    print(f"\n   K = {K}")
    print(f"   A + BK = \n{Acl}")
    print(f"   σ(A + BK) = {np.linalg.eigvals(Acl)}")

    # 5) K_g
    print("\n5) Синтез K_g")
    # Проверка существования: det([A-λI, B; C, D]) ≠ 0 для λ ∈ σ(Γ_g)
    print("   Условие существования (Франкис-Дэвисона):")
    for lam in eigs_Gg:
        R = np.block([[A - lam * np.eye(n), B], [C, D]])
        det_R = np.linalg.det(R)
        print(f"   λ = {lam:.4f}: det R(λ) = {det_R}")
    Pi_g, K_g = solve_francis_g(A, B, C, D, K, Gam_g, Y_g)
    print(f"   Π_g = \n{Pi_g}")
    print(f"   K_g = {K_g}")
    res1 = Pi_g @ Gam_g - Acl @ Pi_g - B @ K_g
    res2 = (C + D @ K) @ Pi_g + D @ K_g - Y_g
    print(f"   max|невязка1| = {np.max(np.abs(res1))}")
    print(f"   max|невязка2| = {np.max(np.abs(res2))}")

    # 6) K_f
    print("\n6) Синтез K_f")
    eigs_Gf_unique = np.unique(np.round(eigs_Gf, 6))
    print("   Условие существования:")
    for lam in eigs_Gf:
        R = np.block([[A - lam * np.eye(n), B], [C, D]])
        det_R = np.linalg.det(R)
        print(f"   λ = {lam:.4f}: det R(λ) = {det_R}")
    Pi_f, K_f = solve_francis_f(A, B, C, D, B_f, D_f, K, Gam_f, Y_f)
    print(f"   Π_f = \n{Pi_f}")
    print(f"   K_f = {K_f}")
    res1f = Pi_f @ Gam_f - Acl @ Pi_f - B_f @ Y_f - B @ K_f
    res2f = (C + D @ K) @ Pi_f + D @ K_f + D_f @ Y_f
    print(f"   max|невязка1| = {np.max(np.abs(res1f))}")
    print(f"   max|невязка2| = {np.max(np.abs(res2f))}")

    # ====================================================
    # 7) Моделирование 5 режимов
    # ====================================================
    T_END = 12.0
    t_eval = np.linspace(0, T_END, 4000)
    x0 = np.zeros(n)

    def gen_f(t, w_full):
        return Gam_f @ w_full[:4]

    def gen_g(t, w_full):
        return Gam_g @ w_full[:3]

    # Полный ОДУ-вектор: [x (3), w_f (4), w_g (3)] = 10
    def make_rhs(u_law):
        def rhs(t, z):
            x = z[0:3]
            wf = z[3:7]
            wg = z[7:10]
            f = Y_f @ wf
            u = u_law(x, wf, wg)
            dx = A @ x + (B @ np.atleast_1d(u)).flatten() + B_f @ f
            dwf = Gam_f @ wf
            dwg = Gam_g @ wg
            return np.concatenate([dx, dwf, dwg])
        return rhs

    def y_of(x, wf, u):
        f = Y_f @ wf
        return (C @ x + D @ np.atleast_1d(u) + D_f @ f).flatten()

    z0 = np.concatenate([x0, w_f0, w_g0])

    laws = {
        'open':       lambda x, wf, wg: 0.0,
        'feedback':   lambda x, wf, wg: float((K @ x).item()),
        'no_track':   lambda x, wf, wg: float((K @ x + K_f @ wf).item()),
        'no_comp':    lambda x, wf, wg: float((K @ x + K_g @ wg).item()),
        'full':       lambda x, wf, wg: float((K @ x + K_g @ wg + K_f @ wf).item()),
    }

    sims = {}
    for name, ul in laws.items():
        sol = solve_ivp(make_rhs(ul), [0, T_END], z0, t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        x_t = sol.y[0:3]
        wf_t = sol.y[3:7]
        wg_t = sol.y[7:10]
        u_t = np.array([ul(x_t[:, i], wf_t[:, i], wg_t[:, i])
                        for i in range(len(sol.t))])
        f_t = Y_f @ wf_t
        g_t = (Y_g @ wg_t).flatten()
        y_t = np.array([y_of(x_t[:, i], wf_t[:, i], u_t[i])
                        for i in range(len(sol.t))]).flatten()
        e_t = g_t - y_t
        sims[name] = dict(t=sol.t, x=x_t, wf=wf_t, wg=wg_t, u=u_t,
                          f=f_t, g=g_t, y=y_t, e=e_t)

    # ===== Графики =====
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 110})
    W, H = 8, 4

    def save(fig, name):
        path = os.path.join(images_dir, name)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    # а) Разомкнутая
    s = sims['open']
    fig = plt.figure(figsize=(W, H))
    plt.plot(s['t'], s['f'][0], label='$f_1(t)$')
    plt.plot(s['t'], s['f'][1], label='$f_2(t)$')
    plt.plot(s['t'], s['g'], label='$g(t)$', linestyle='--')
    plt.xlabel('t, c'); plt.ylabel('Сигналы'); plt.grid(True); plt.legend()
    save(fig, 'task1_open_fg.png')

    fig = plt.figure(figsize=(W, H))
    for i in range(3):
        plt.plot(s['t'], s['x'][i], label=f'$x_{i+1}(t)$')
    plt.xlabel('t, c'); plt.ylabel('x(t)'); plt.grid(True); plt.legend()
    save(fig, 'task1_open_x.png')

    fig = plt.figure(figsize=(W, H))
    plt.plot(s['t'], s['g'], label='$g(t)$')
    plt.plot(s['t'], s['y'], label='$y(t)$', linestyle='--')
    plt.xlabel('t, c'); plt.ylabel('Выход'); plt.grid(True); plt.legend()
    save(fig, 'task1_open_y.png')

    # для замкнутых режимов
    closed_modes = ['feedback', 'no_track', 'no_comp', 'full']
    mode_labels = {
        'feedback': 'u = Kx',
        'no_track': 'u = Kx + K_f w_f',
        'no_comp':  'u = Kx + K_g w_g',
        'full':     'u = Kx + K_g w_g + K_f w_f',
    }

    # u(t) на одной плоскости для замкнутых режимов
    fig = plt.figure(figsize=(W, H))
    for nm in closed_modes:
        plt.plot(sims[nm]['t'], sims[nm]['u'], label=mode_labels[nm])
    plt.xlabel('t, c'); plt.ylabel('u(t)'); plt.grid(True); plt.legend()
    save(fig, 'task1_closed_u.png')

    # y(t) на одной плоскости + g
    fig = plt.figure(figsize=(W, H))
    plt.plot(sims['feedback']['t'], sims['feedback']['g'],
             label='g(t)', color='black', linewidth=1.5)
    for nm in closed_modes:
        plt.plot(sims[nm]['t'], sims[nm]['y'], label=mode_labels[nm], linestyle='--')
    plt.xlabel('t, c'); plt.ylabel('y(t)'); plt.grid(True); plt.legend()
    save(fig, 'task1_closed_y.png')

    # e(t) на одной плоскости
    fig = plt.figure(figsize=(W, H))
    for nm in closed_modes:
        plt.plot(sims[nm]['t'], sims[nm]['e'], label=mode_labels[nm])
    plt.xlabel('t, c'); plt.ylabel('e(t) = g(t) - y(t)'); plt.grid(True); plt.legend()
    save(fig, 'task1_closed_e.png')

    # x(t) — отдельный рисунок для каждого режима
    for nm in closed_modes:
        s = sims[nm]
        fig = plt.figure(figsize=(W, H))
        for i in range(3):
            plt.plot(s['t'], s['x'][i], label=f'$x_{i+1}(t)$')
        plt.xlabel('t, c'); plt.ylabel('x(t)'); plt.grid(True); plt.legend()
        save(fig, f'task1_x_{nm}.png')

    # Сводные характеристики
    summary = {}
    for nm, s in sims.items():
        summary[nm] = dict(
            max_u=float(np.max(np.abs(s['u']))),
            max_x=float(np.max(np.abs(s['x']))),
            max_y=float(np.max(np.abs(s['y']))),
            max_e=float(np.max(np.abs(s['e']))),
            e_end=float(s['e'][-1]),
        )
    print("\n7) Сводная таблица режимов:")
    for nm, d in summary.items():
        print(f"   {nm:10s}: {d}")

    # Сохраним результаты в JSON для других скриптов
    out = dict(
        A=A.tolist(), B=B.tolist(), B_f=B_f.tolist(),
        C=C.tolist(), D=D.tolist(), D_f=D_f.tolist(),
        Gam_f=Gam_f.tolist(), Y_f=Y_f.tolist(), w_f0=w_f0.tolist(),
        Gam_g=Gam_g.tolist(), Y_g=Y_g.tolist(), w_g0=w_g0.tolist(),
        K=K.tolist(), Pi_g=Pi_g.tolist(), K_g=K_g.tolist(),
        Pi_f=Pi_f.tolist(), K_f=K_f.tolist(),
        eigs_A=[[float(np.real(z)), float(np.imag(z))] for z in eigs_A],
        eigs_Gf=[[float(np.real(z)), float(np.imag(z))] for z in eigs_Gf],
        eigs_Gg=[[float(np.real(z)), float(np.imag(z))] for z in eigs_Gg],
        eigs_Acl=[[float(np.real(z)), float(np.imag(z))]
                  for z in np.linalg.eigvals(Acl)],
        summary=summary,
    )
    with open(os.path.join(script_dir, 'lab4_task1_results.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
