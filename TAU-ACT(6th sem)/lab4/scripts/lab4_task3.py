import os
import json
import numpy as np
import scipy.linalg as la
from scipy.signal import place_poles
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

    with open(os.path.join(script_dir, 'lab4_task2_results.json'),
              encoding='utf-8') as fp:
        d2 = json.load(fp)
    Gam_q = np.array(d2['Gam_q']); Y_q = np.array(d2['Y_q'])
    Q_obs = np.array(d2['Q'])

    n = A.shape[0]
    s_f = Gam_f.shape[0]
    s_g = Gam_g.shape[0]
    Q_inv = np.linalg.inv(Q_obs)

    print("=" * 60)
    print("ЗАДАНИЕ 3: Наблюдатели возмущения")
    print("=" * 60)

    # =====================================================
    # 1) Наблюдатель возмущения по состоянию
    #    ŵ_f = z + L̃ C̃ x,  ż = F z + ...
    #    Условие: C̃ B_f = I_2  (т.е. C̃ — левое обратное к B_f)
    #    F = Γ_f − L Y_f. Желаемый спектр: {−2,−3,−4,−5}.
    # =====================================================
    print("\n1) Наблюдатель возмущения по состоянию:")
    BtB = B_f.T @ B_f
    BtB_inv = np.linalg.inv(BtB)
    C_tilde = BtB_inv @ B_f.T              # 2×3
    print(f"   B_f^T B_f = \n{BtB}")
    print(f"   (B_f^T B_f)^(-1) = \n{BtB_inv}")
    print(f"   C̃ = (B_f^T B_f)^(-1) B_f^T = \n{C_tilde}")
    print(f"   C̃ B_f = \n{C_tilde @ B_f}")

    desired_F = np.array([-2.0, -3.0, -4.0, -5.0])
    pp = place_poles(Gam_f.T, Y_f.T, desired_F)
    L_st = pp.gain_matrix.T                 # 4×2
    F = Gam_f - L_st @ Y_f
    eigs_F = np.sort(np.linalg.eigvals(F).real)
    print(f"   Желаемый спектр F: {desired_F}")
    print(f"   L = \n{L_st}")
    print(f"   F = Γ_f − L Y_f = \n{F}")
    print(f"   σ(F) = {eigs_F}")

    # =====================================================
    # 2) Наблюдатель возмущения по выходу
    #    f̂ = y − C x − D u = D_f Y_f w_f
    #    Сильвестр: Q Γ_f − Γ̄ Q = Y · (D_f Y_f)
    #    Желаемый спектр Γ̄: {−2,−3,−4,−5}.
    # =====================================================
    print("\n2) Наблюдатель возмущения по выходу:")
    Gam_bar = np.diag([-2.0, -3.0, -4.0, -5.0])
    Y_bar = np.array([[1.0], [1.0], [1.0], [1.0]])
    DfYf = (D_f @ Y_f)
    print(f"   D_f Y_f = {DfYf}")
    rhs = Y_bar @ DfYf
    Q_d = la.solve_sylvester(-Gam_bar, Gam_f, rhs)   # -Γ̄ Q + Q Γ_f = Y · DfYf
    res_Qd = Q_d @ Gam_f - Gam_bar @ Q_d - rhs
    print(f"   Γ̄ = \n{Gam_bar}")
    print(f"   Y = {Y_bar.T}")
    print(f"   Q = \n{Q_d}")
    print(f"   max|невязка| = {np.max(np.abs(res_Qd))}")
    print(f"   det(Q) = {np.linalg.det(Q_d)}")
    Q_d_inv = np.linalg.inv(Q_d)

    # =====================================================
    # Моделирование. Для каждого наблюдателя возмущения:
    #   x — измеряемо, регулятор u = K x + K_g ŵ_g + K_f ŵ_f.
    #   Наблюдатель задания: w̄̇_g = Γ_q w̄_g + Y_q g, ŵ_g = Q_obs^(-1) w̄_g.
    # =====================================================
    T_END = 8.0
    t_eval = np.linspace(0, T_END, 4000)
    x0 = np.zeros(n)
    wbar0 = np.zeros(s_g)

    # ========= вариант А: наблюдатель по состоянию =========
    # z = z(t), ŵ_f = z + L̃ x
    # ż = F z + F L̃ x − L̃ A x − L̃ B u
    # где L̃ = L_st @ C̃        (размер 4×3, действует на x напрямую)
    L_tilde = L_st @ C_tilde

    def rhs_A(t, ζ):
        x = ζ[0:3]
        wf = ζ[3:7]
        wg = ζ[7:10]
        z = ζ[10:14]
        wbar = ζ[14:17]
        wf_hat = z + L_tilde @ x
        wg_hat = Q_inv @ wbar
        u = (K @ x + K_g @ wg_hat + K_f @ wf_hat).item()
        f = Y_f @ wf
        dx = A @ x + B.flatten() * u + B_f @ f
        dwf = Gam_f @ wf
        dwg = Gam_g @ wg
        dz = F @ z + F @ L_tilde @ x - L_tilde @ A @ x - L_tilde @ B.flatten() * u
        g = (Y_g @ wg).item()
        dwbar = Gam_q @ wbar + Y_q.flatten() * g
        return np.concatenate([dx, dwf, dwg, dz, dwbar])

    z0_A = np.concatenate([x0, w_f0, w_g0, np.zeros(4), wbar0])
    sol_A = solve_ivp(rhs_A, [0, T_END], z0_A, t_eval=t_eval,
                      method='RK45', rtol=1e-8, atol=1e-10)
    t = sol_A.t
    x_A = sol_A.y[0:3]; wf_A = sol_A.y[3:7]; wg_A = sol_A.y[7:10]
    z_A = sol_A.y[10:14]; wbar_A = sol_A.y[14:17]
    wf_hat_A = z_A + L_tilde @ x_A
    wg_hat_A = Q_inv @ wbar_A
    f_A = Y_f @ wf_A
    g_A = (Y_g @ wg_A).flatten()
    u_A = np.array([(K @ x_A[:, i] + K_g @ wg_hat_A[:, i]
                     + K_f @ wf_hat_A[:, i]).item() for i in range(len(t))])
    y_A = np.array([(C @ x_A[:, i] + D.flatten() * u_A[i]
                     + D_f @ f_A[:, i]).item() for i in range(len(t))])
    e_A = g_A - y_A
    ef_A = wf_A - wf_hat_A

    # ========= вариант B: наблюдатель по выходу =========
    def rhs_B(t, ζ):
        x = ζ[0:3]
        wf = ζ[3:7]
        wg = ζ[7:10]
        wbar_f = ζ[10:14]
        wbar_g = ζ[14:17]
        wf_hat = Q_d_inv @ wbar_f
        wg_hat = Q_inv @ wbar_g
        u = (K @ x + K_g @ wg_hat + K_f @ wf_hat).item()
        f = Y_f @ wf
        dx = A @ x + B.flatten() * u + B_f @ f
        dwf = Gam_f @ wf
        dwg = Gam_g @ wg
        # f̂ = y − C x − D u  (скаляр)
        y_meas = (C @ x).item() + D.item() * u + (D_f @ f).item()
        f_hat = y_meas - (C @ x).item() - D.item() * u
        dwbar_f = (Gam_bar @ wbar_f) + Y_bar.flatten() * f_hat
        g = (Y_g @ wg).item()
        dwbar_g = Gam_q @ wbar_g + Y_q.flatten() * g
        return np.concatenate([dx, dwf, dwg, dwbar_f, dwbar_g])

    z0_B = np.concatenate([x0, w_f0, w_g0, np.zeros(4), wbar0])
    sol_B = solve_ivp(rhs_B, [0, T_END], z0_B, t_eval=t_eval,
                      method='RK45', rtol=1e-8, atol=1e-10)
    x_B = sol_B.y[0:3]; wf_B = sol_B.y[3:7]; wg_B = sol_B.y[7:10]
    wbar_f_B = sol_B.y[10:14]; wbar_g_B = sol_B.y[14:17]
    wf_hat_B = Q_d_inv @ wbar_f_B
    wg_hat_B = Q_inv @ wbar_g_B
    f_B = Y_f @ wf_B
    g_B = (Y_g @ wg_B).flatten()
    u_B = np.array([(K @ x_B[:, i] + K_g @ wg_hat_B[:, i]
                     + K_f @ wf_hat_B[:, i]).item() for i in range(len(t))])
    y_B = np.array([(C @ x_B[:, i] + D.flatten() * u_B[i]
                     + D_f @ f_B[:, i]).item() for i in range(len(t))])
    e_B = g_B - y_B
    ef_B = wf_B - wf_hat_B

    # =====================================================
    # Графики
    # =====================================================
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 110})
    W, H = 8, 4

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(images_dir, name), bbox_inches='tight')
        plt.close(fig)

    def plot_set(prefix, x, wf, wf_hat, ef, u, y, g, e):
        fig = plt.figure(figsize=(W, H))
        plt.plot(t, u, color='C3', label='$u(t)$')
        plt.xlabel('t, c'); plt.ylabel('u(t)'); plt.grid(True); plt.legend()
        save(fig, f'{prefix}_u.png')

        fig = plt.figure(figsize=(W, H))
        for i in range(3):
            plt.plot(t, x[i], label=f'$x_{i+1}(t)$')
        plt.xlabel('t, c'); plt.ylabel('x(t)'); plt.grid(True); plt.legend()
        save(fig, f'{prefix}_x.png')

        fig = plt.figure(figsize=(W, H + 0.5))
        for i in range(4):
            plt.plot(t, wf[i], color=f'C{i}', label=f'$w_{{f,{i+1}}}$')
            plt.plot(t, wf_hat[i], color=f'C{i}', linestyle='--', alpha=0.7)
        plt.xlabel('t, c'); plt.ylabel('Компоненты'); plt.grid(True)
        plt.legend(ncol=4, loc='upper right', fontsize=9)
        save(fig, f'{prefix}_wf.png')

        fig = plt.figure(figsize=(W, H))
        for i in range(4):
            plt.plot(t, ef[i], label=f'$e_{{f,{i+1}}}$')
        plt.xlabel('t, c'); plt.ylabel('$e_f(t)$'); plt.grid(True); plt.legend()
        save(fig, f'{prefix}_ef.png')

        fig = plt.figure(figsize=(W, H))
        plt.plot(t, g, label='$g(t)$')
        plt.plot(t, y, linestyle='--', label='$y(t)$')
        plt.xlabel('t, c'); plt.ylabel('Выход'); plt.grid(True); plt.legend()
        save(fig, f'{prefix}_y.png')

        fig = plt.figure(figsize=(W, H))
        plt.plot(t, e, color='C3', label='$e(t)=g(t)-y(t)$')
        plt.xlabel('t, c'); plt.ylabel('e(t)'); plt.grid(True); plt.legend()
        save(fig, f'{prefix}_e.png')

    plot_set('task3_state', x_A, wf_A, wf_hat_A, ef_A, u_A, y_A, g_A, e_A)
    plot_set('task3_output', x_B, wf_B, wf_hat_B, ef_B, u_B, y_B, g_B, e_B)

    sumA = dict(max_u=float(np.max(np.abs(u_A))),
                max_x=float(np.max(np.abs(x_A))),
                max_ef=float(np.max(np.abs(ef_A))),
                e_end=float(e_A[-1]))
    sumB = dict(max_u=float(np.max(np.abs(u_B))),
                max_x=float(np.max(np.abs(x_B))),
                max_ef=float(np.max(np.abs(ef_B))),
                e_end=float(e_B[-1]))
    print(f"\n3) Сводка:\n   По состоянию: {sumA}\n   По выходу:    {sumB}")

    out = dict(
        C_tilde=C_tilde.tolist(),
        L_st=L_st.tolist(),
        L_tilde=L_tilde.tolist(),
        F=F.tolist(),
        eigs_F=eigs_F.tolist(),
        Gam_bar=Gam_bar.tolist(), Y_bar=Y_bar.tolist(),
        DfYf=DfYf.tolist(),
        Q_d=Q_d.tolist(),
        det_Qd=float(np.linalg.det(Q_d)),
        sumA=sumA, sumB=sumB,
    )
    with open(os.path.join(script_dir, 'lab4_task3_results.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
