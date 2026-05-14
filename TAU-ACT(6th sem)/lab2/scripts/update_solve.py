import sys

path = r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab2\scripts\solve_lab2.py"
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Make task1_x plot generation separate
text = text.replace("fig_x.tight_layout()\nfig_x.savefig(os.path.join(img_dir, 'task1_x.png'), bbox_inches='tight')\nplt.close(fig_x)", "")

# Inside Task 1 simulation loop, save separate fig
plot_loop_orig = "    ax.legend(loc='best')"
plot_loop_new = """    ax.legend(loc='best')
    
    # Save isolated plot
    fig_single, ax_single = plt.subplots(n1, 1, figsize=(10, 6), sharex=True)
    if n1 == 1:
        ax_single = [ax_single]
    for j in range(n1):
        ax_single[j].plot(sol.t, x_t[j], label=f'$x_{j+1}(t)$')
        ax_single[j].legend(loc='best')
        ax_single[j].grid(True, alpha=0.3)
    ax_single[-1].set_xlabel('t')
    fig_single.suptitle(f"Спектр {sp['name']}: {sp['label']}")
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(img_dir, f'task1_x_{sp["name"]}.png'), bbox_inches='tight')
    plt.close(fig_single)"""
text = text.replace(plot_loop_orig, plot_loop_new)

# Add Task 1 Sylvester intermediates
syl_orig = """        try:
            K_syl = sylvester_method(A1, B1, demo_poles)"""
syl_new = """        try:
            coeffs_syl = np.real(np.poly(demo_poles))
            Lambda_syl = np.zeros((n1, n1))
            Lambda_syl[:-1, 1:] = np.eye(n1 - 1)
            Lambda_syl[-1, :] = -coeffs_syl[1:][::-1]
            Psi_syl = np.zeros((1, n1))
            Psi_syl[0, 0] = 1.0
            X_syl = la.solve_sylvester(A1, -Lambda_syl, -B1 @ Psi_syl)
            if abs(np.linalg.det(X_syl)) < 1e-10:
                Psi_syl = np.ones((1, n1))
                X_syl = la.solve_sylvester(A1, -Lambda_syl, -B1 @ Psi_syl)
            V("T1_Syl_Lambda", mat2typ(Lambda_syl))
            V("T1_Syl_Y", mat2typ(Psi_syl))
            V("T1_Syl_X", mat2typ(X_syl))
            K_syl = sylvester_method(A1, B1, demo_poles)"""
text = text.replace(syl_orig, syl_new)

# Task 2 Observer Sylvester explicitly for Spectrum 1 to show variables
t2_orig = """    eigs_obs = np.linalg.eigvals(A2 + L @ C2)"""
t2_new = """    eigs_obs = np.linalg.eigvals(A2 + L @ C2)
    # Output matrices for Sylvester method (for observer dual problem)
    # Observer: eig(A+LC), dual is eig(A^T + C^T L^T).
    # Sylvester: A^T P - P Gamma = -C^T Y -> L^T = Y P^-1 => L = P^-T Y^T
    coeffs_obs = np.real(np.poly(poles))
    Gamma_obs = np.zeros((n2, n2))
    Gamma_obs[:-1, 1:] = np.eye(n2 - 1)
    Gamma_obs[-1, :] = -coeffs_obs[1:][::-1]
    Y_obs = np.ones((1, n2))  # Typically all 1s for observer
    P_obs = la.solve_sylvester(A2.T, -Gamma_obs, -C2.T @ Y_obs)
    V(f"T2_Syl_Gamma_{sp['name']}", mat2typ(Gamma_obs))
    V(f"T2_Syl_Y_{sp['name']}", mat2typ(Y_obs))
    V(f"T2_Syl_P_{sp['name']}", mat2typ(P_obs))"""
text = text.replace(t2_orig, t2_new)


# Task 4 matrices
t4_orig = """print(f"A11 = {A11}")"""
t4_new = """print(f"A11 = {A11}")
V("T4_A11", mat2typ(A11))
V("T4_A12", mat2typ(A12))
V("T4_A21", mat2typ(A21))
V("T4_A22", mat2typ(A22))
V("T4_T", mat2typ(T_perm))"""
text = text.replace(t4_orig, t4_new)


with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated script.")
