import os
import json
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.set_printoptions(precision=6, suppress=True)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'images')
    ensure_dir(images_dir)

    with open(os.path.join(script_dir, 'lab4_task1_results.json'),
              encoding='utf-8') as fp:
        d = json.load(fp)

    A = np.array(d['A']); B = np.array(d['B']); B_f = np.array(d['B_f'])
    C = np.array(d['C']); D = np.array(d['D']); D_f = np.array(d['D_f'])
    Gam_f = np.array(d['Gam_f']); Y_f = np.array(d['Y_f'])
    Gam_g = np.array(d['Gam_g']); Y_g = np.array(d['Y_g'])
    w_f0 = np.array(d['w_f0']); w_g0 = np.array(d['w_g0'])
    K = np.array(d['K']); K_g = np.array(d['K_g']); K_f = np.array(d['K_f'])

    n = A.shape[0]
    s_f = Gam_f.shape[0]
    s_g = Gam_g.shape[0]

    print("=" * 60)
    print("ЗАДАНИЕ 2: Слежение и компенсация по выходу")
    print("=" * 60)
    print(f"K   = {K}")
    print(f"K_g = {K_g}")
    print(f"K_f = {K_f}")

    # =====================================================
    # 1) Наблюдатель задающего воздействия
    #    Желаемая динамика: σ(Γ_q) = {-2, -3, -4}
    #    Уравнение Сильвестра: Q Γ_g - Γ_q Q = Y_q Y_g
    # =====================================================
    Gam_q = np.diag([-2.0, -3.0, -4.0])
    Y_q = np.array([[1.0], [1.0], [1.0]])  # столбец из единиц

    # Q Γ_g - Γ_q Q = Y_q Y_g
    # Решим: -Γ_q Q + Q Γ_g = Y_q Y_g  →  scipy.linalg.solve_sylvester(-Γ_q, Γ_g, Y_q Y_g)
    # solve_sylvester решает A X + X B = C → A=-Γ_q, B=Γ_g, C=Y_q Y_g
    rhs = Y_q @ Y_g
    Q_obs = la.solve_sylvester(-Gam_q, Gam_g, rhs)
    res_Q = Q_obs @ Gam_g - Gam_q @ Q_obs - Y_q @ Y_g
    print(f"\n1) Наблюдатель задающего воздействия:")
    print(f"   Γ_q = \n{Gam_q}")
    print(f"   Y_q = {Y_q.T}")
    print(f"   Q = \n{Q_obs}")
    print(f"   max|невязка| = {np.max(np.abs(res_Q))}")
    print(f"   det(Q) = {np.linalg.det(Q_obs)}")

    # =====================================================
    # 2) Расширенная система ξ = [w_f; x]
    # =====================================================
    A_ext = np.block([
        [Gam_f,                     np.zeros((s_f, n))],
        [B_f @ Y_f,                 A]
    ])
    B_ext = np.vstack([np.zeros((s_f, B.shape[1])), B])
    C_ext = np.hstack([D_f @ Y_f,    C])

    print(f"\n2) Расширенная система:")
    print(f"   A_ext = \n{A_ext}")
    print(f"   B_ext = {B_ext.flatten()}")
    print(f"   C_ext = {C_ext}")
    BfYf = B_f @ Y_f
    DfYf = D_f @ Y_f
    print(f"   B_f Y_f = \n{BfYf}")
    print(f"   D_f Y_f = {DfYf}")

    # Наблюдаемость расширенной системы
    n_ext = A_ext.shape[0]
    O = C_ext.copy()
    for i in range(1, n_ext):
        O = np.vstack([O, C_ext @ np.linalg.matrix_power(A_ext, i)])
    rank_O = np.linalg.matrix_rank(O)
    print(f"   rank(O_ext) = {rank_O} (n_ext = {n_ext})")

    # =====================================================
    # 3) Наблюдатель расширенной размерности.
    #    Задаём σ(A_ext + L C_ext) = {-2, -3, -4, -5, -6, -7, -8}
    # =====================================================
    desired = np.array([-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
    # Через двойственную задачу: разместить полюса для (A_ext^T, C_ext^T)
    from scipy.signal import place_poles
    pp = place_poles(A_ext.T, C_ext.T, desired)
    L = -pp.gain_matrix.T          # L размером n_ext × p
    eigs_obs = np.linalg.eigvals(A_ext + L @ C_ext)
    print(f"\n3) Наблюдатель расширенной размерности:")
    print(f"   Желаемый спектр: {desired}")
    print(f"   L = \n{L}")
    print(f"   σ(A_ext + L C_ext) = {np.sort(eigs_obs.real)}")

    # =====================================================
    # 4) Моделирование замкнутой системы
    # =====================================================
    T_END = 6.0
    t_eval = np.linspace(0, T_END, 4000)
    x0 = np.zeros(n)
    xi0 = np.zeros(n_ext)               # ξ̂(0) = 0
    wbar0 = np.zeros(s_g)               # w̄_g(0) = 0

    # Состояние ОДУ:
    # z = [x (3), w_f (4), w_g (3), ξ̂ (7), w̄_g (3)]   = 20
    Q_inv = np.linalg.inv(Q_obs)

    def rhs(t, z):
        x = z[0:3]
        wf = z[3:7]
        wg = z[7:10]
        xi_hat = z[10:17]               # [ŵ_f (4); x̂ (3)]
        wbar = z[17:20]
        wf_hat = xi_hat[:s_f]
        x_hat = xi_hat[s_f:]
        # Оценка задания
        wg_hat = Q_inv @ wbar
        # Управление по оценкам
        u = (K @ x_hat + K_g @ wg_hat + K_f @ wf_hat).item()
        # Объект
        f = Y_f @ wf
        dx = A @ x + (B.flatten() * u) + B_f @ f
        dwf = Gam_f @ wf
        dwg = Gam_g @ wg
        # Выходы
        y_meas = (C @ x).flatten() + D.flatten() * u + D_f @ f
        # Расширенный наблюдатель: ξ̂̇ = (A_ext + L C_ext) ξ̂ + (B_ext + L D) u - L y
        dxi = (A_ext + L @ C_ext) @ xi_hat + (B_ext + L @ D).flatten() * u \
              - (L @ y_meas).flatten()
        # Наблюдатель задания: w̄̇_g = Γ_q w̄_g + Y_q g  (с g = Y_g w_g)
        g = (Y_g @ wg).item()
        dwbar = Gam_q @ wbar + (Y_q.flatten() * g)
        return np.concatenate([dx, dwf, dwg, dxi, dwbar])

    z0 = np.concatenate([x0, w_f0, w_g0, xi0, wbar0])
    sol = solve_ivp(rhs, [0, T_END], z0, t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)
    t = sol.t
    x_t = sol.y[0:3]
    wf_t = sol.y[3:7]
    wg_t = sol.y[7:10]
    xi_t = sol.y[10:17]
    wbar_t = sol.y[17:20]
    wf_hat_t = xi_t[0:4]
    x_hat_t = xi_t[4:7]
    wg_hat_t = Q_inv @ wbar_t

    # Сигналы
    g_t = (Y_g @ wg_t).flatten()
    f_t = Y_f @ wf_t
    u_t = np.array([(K @ x_hat_t[:, i] + K_g @ wg_hat_t[:, i]
                     + K_f @ wf_hat_t[:, i]).item()
                    for i in range(len(t))])
    y_t = np.array([(C @ x_t[:, i] + D.flatten() * u_t[i]
                     + D_f @ f_t[:, i]).item() for i in range(len(t))])
    e_t = g_t - y_t
    e_f = np.vstack([wf_t, x_t]) - xi_t          # 7 компонент
    e_g = wg_t - wg_hat_t                         # 3 компоненты

    # =====================================================
    # Графики
    # =====================================================
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 110})
    W, H = 8, 4

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(images_dir, name), bbox_inches='tight')
        plt.close(fig)

    # u(t)
    fig = plt.figure(figsize=(W, H))
    plt.plot(t, u_t, label='$u(t)$', color='C3')
    plt.xlabel('t, c'); plt.ylabel('u(t)'); plt.grid(True); plt.legend()
    save(fig, 'task2_u.png')

    # Сравнение [w_f; x] и [ŵ_f; x̂]
    fig = plt.figure(figsize=(W, H + 1))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    labels = ['$w_{f,1}$', '$w_{f,2}$', '$w_{f,3}$', '$w_{f,4}$',
              '$x_1$', '$x_2$', '$x_3$']
    real = np.vstack([wf_t, x_t])
    for i in range(7):
        plt.plot(t, real[i], color=colors[i], label=labels[i])
        plt.plot(t, xi_t[i], color=colors[i], linestyle='--', alpha=0.7)
    plt.xlabel('t, c'); plt.ylabel('Компоненты'); plt.grid(True)
    plt.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.35))
    save(fig, 'task2_xi.png')

    # Ошибка наблюдателя расширенной размерности
    fig = plt.figure(figsize=(W, H))
    for i in range(7):
        plt.plot(t, e_f[i], label=labels[i].replace('$', '$e_{')[:-1] + '}$')
    plt.xlabel('t, c'); plt.ylabel('$e_f(t)$'); plt.grid(True)
    plt.legend(ncol=4, loc='upper right', fontsize=9)
    save(fig, 'task2_ef.png')

    # w_g vs ŵ_g
    fig = plt.figure(figsize=(W, H))
    for i in range(3):
        plt.plot(t, wg_t[i], color=f'C{i}', label=f'$w_{{g,{i+1}}}$')
        plt.plot(t, wg_hat_t[i], color=f'C{i}', linestyle='--', alpha=0.7)
    plt.xlabel('t, c'); plt.ylabel('Компоненты'); plt.grid(True); plt.legend()
    save(fig, 'task2_wg.png')

    # Ошибка наблюдателя задания
    fig = plt.figure(figsize=(W, H))
    for i in range(3):
        plt.plot(t, e_g[i], label=f'$e_{{g,{i+1}}}(t)$')
    plt.xlabel('t, c'); plt.ylabel('$e_g(t)$'); plt.grid(True); plt.legend()
    save(fig, 'task2_eg.png')

    # y(t) vs g(t)
    fig = plt.figure(figsize=(W, H))
    plt.plot(t, g_t, label='$g(t)$')
    plt.plot(t, y_t, label='$y(t)$', linestyle='--')
    plt.xlabel('t, c'); plt.ylabel('Выход'); plt.grid(True); plt.legend()
    save(fig, 'task2_y.png')

    # e(t)
    fig = plt.figure(figsize=(W, H))
    plt.plot(t, e_t, label='$e(t)=g(t)-y(t)$', color='C3')
    plt.xlabel('t, c'); plt.ylabel('e(t)'); plt.grid(True); plt.legend()
    save(fig, 'task2_e.png')

    # Сводка
    summary = dict(
        max_u=float(np.max(np.abs(u_t))),
        max_x=float(np.max(np.abs(x_t))),
        max_ef=float(np.max(np.abs(e_f))),
        max_eg=float(np.max(np.abs(e_g))),
        e_end=float(e_t[-1]),
    )
    print(f"\n4) Моделирование (T = {T_END}):")
    for k, v in summary.items():
        print(f"   {k} = {v}")

    out = dict(
        Gam_q=Gam_q.tolist(), Y_q=Y_q.tolist(), Q=Q_obs.tolist(),
        A_ext=A_ext.tolist(), B_ext=B_ext.tolist(), C_ext=C_ext.tolist(),
        BfYf=BfYf.tolist(), DfYf=DfYf.tolist(),
        L=L.tolist(),
        eigs_obs_ext=[[float(np.real(z)), float(np.imag(z))] for z in eigs_obs],
        rank_O=int(rank_O),
        summary=summary,
    )
    with open(os.path.join(script_dir, 'lab4_task2_results.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
