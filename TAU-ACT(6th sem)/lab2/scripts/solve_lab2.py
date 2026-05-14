"""
Лабораторная работа №2 — Модальные регуляторы и наблюдатели
Вариант 24: Задания 1,2 → условие №14; Задания 3,4 → условие №4
"""
import os, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_visual_end(t, signals, tol=None):
    sig = np.atleast_2d(signals)
    env = np.max(np.abs(sig), axis=0)
    if tol is None:
        tol = max(1e-3, 0.01 * np.max(env))
    above = np.where(env > tol)[0]
    if len(above) == 0: return t[0] + 0.5
    idx = above[-1]
    tend = t[idx] * 1.15
    return min(t[-1], max(tend, t[0]+0.1))

plt.rcParams.update({
    'font.size': 11, 'figure.figsize': (10, 5),
    'lines.linewidth': 1.5, 'axes.grid': True,
    'grid.alpha': 0.3, 'figure.dpi': 150,
})

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_dir = os.path.join(base_dir, 'images')
os.makedirs(img_dir, exist_ok=True)
vars_file = os.path.join(base_dir, 'variables.typ')

# ============================================================
# Утилиты
# ============================================================
def fmt(val, tol=1e-4):
    if isinstance(val, (complex, np.complex128, np.complex64)):
        if abs(val.imag) > tol:
            r = fmt(val.real)
            i_abs = abs(val.imag)
            sign = "+" if val.imag > 0 else "-"
            i_s = fmt(i_abs)
            if i_s == "1": i_s = ""
            if r == "0":
                return f"{'-' if sign=='-' else ''}{i_s}i"
            return f"{r} {sign} {i_s}i"
        return fmt(val.real)
    if abs(val) < tol:
        return "0"
    if abs(val - round(val)) < tol:
        return str(int(round(val)))
    s = f"{val:.4f}".rstrip('0').rstrip('.')
    if s == "-0": return "0"
    return s

def mat2typ(M):
    M = np.atleast_2d(np.array(M))
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows = []
    for r in M:
        rows.append(", ".join(fmt(v) for v in r))
    return "$mat(" + "; ".join(rows) + ")$"

def vec2typ(v):
    v = np.array(v).flatten()
    return "$mat(" + "; ".join(fmt(x) for x in v) + ")$"

def eigs_str(ev):
    parts = []
    for e in sorted(ev, key=lambda x: (x.real, x.imag)):
        parts.append(fmt(e))
    return ", ".join(parts)

def fmt_sci(val):
    import math
    if abs(val) < 1e-15: return "$0$"
    exp = int(math.floor(math.log10(abs(val))))
    m = val / 10**exp
    ms = f"{m:.2f}".rstrip('0').rstrip('.')
    return f"${ms} times 10^({exp})$"

variables = {}
def V(name, val):
    variables[name] = val

# ============================================================
# Ackermann formula for SISO: K such that eig(A+BK) = desired
# ============================================================
def ackermann(A, B, desired_poles):
    n = A.shape[0]
    # Controllability matrix
    U = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    if abs(np.linalg.det(U)) < 1e-10:
        raise ValueError("System not fully controllable, Ackermann not applicable")
    # Desired char poly: p(s) = prod(s - pi)
    coeffs = np.poly(desired_poles)  # [1, c1, c2, ..., cn]
    # p(A) = A^n + c1*A^(n-1) + ... + cn*I
    pA = np.zeros_like(A, dtype=float)
    for i, c in enumerate(coeffs):
        pA += c * np.linalg.matrix_power(A, n - i)
    # K = -e_n^T @ U^{-1} @ p(A)
    e_n = np.zeros(n)
    e_n[-1] = 1.0
    K = -(e_n @ np.linalg.inv(U) @ pA).reshape(1, -1)
    return K

# ============================================================
# Bass-Gura formula for SISO
# ============================================================
def bass_gura(A, B, desired_poles):
    n = A.shape[0]
    U = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    if abs(np.linalg.det(U)) < 1e-10:
        raise ValueError("System not fully controllable")
    # Current char poly coefficients
    a = np.poly(np.linalg.eigvals(A))  # [1, a1, ..., an]
    # Desired char poly coefficients
    a_star = np.poly(desired_poles)  # [1, a1*, ..., an*]
    # Difference
    delta = a_star[1:] - a[1:]
    # T matrix (transformation to controllable canonical form)
    # T = U @ P, where P is built from a coefficients
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n - i):
            P[i, i + j] = a[j]  # upper triangular from char poly
    T = U @ P
    K = -(delta @ np.linalg.inv(T)).reshape(1, -1)
    return K

# ============================================================
# Sylvester equation method: A X - X Lambda = -B Psi
# K = Psi @ X^{-1}
# ============================================================
def sylvester_method(A, B, desired_poles):
    n = A.shape[0]
    desired = np.array(desired_poles, dtype=complex)
    # Use companion matrix for Lambda to handle repeated poles
    coeffs = np.real(np.poly(desired))  # [1, c1, ..., cn]
    # Companion matrix (Frobenius form)
    Lambda = np.zeros((n, n))
    Lambda[:-1, 1:] = np.eye(n - 1)
    Lambda[-1, :] = -coeffs[1:][::-1]
    # Choose Psi = e_1^T = [1, 0, ..., 0]
    Psi = np.zeros((1, n))
    Psi[0, 0] = 1.0
    # Solve Sylvester: A X - X Lambda = -B Psi
    X = la.solve_sylvester(A, -Lambda, -B @ Psi)
    if abs(np.linalg.det(X)) < 1e-10:
        # Try Psi = [1, 1, ..., 1]
        Psi = np.ones((1, n))
        X = la.solve_sylvester(A, -Lambda, -B @ Psi)
    K = np.real(Psi @ np.linalg.inv(X))
    return K

# ============================================================
# Place poles — robust version for controllable and partially
# controllable systems. Returns K such that eig(A+BK) = desired.
# ============================================================
def place_K(A, B, desired):
    """Place poles: returns K such that eig(A+BK) = desired."""
    n = A.shape[0]
    desired = np.array(desired, dtype=complex)

    # Check controllability
    U = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(U, tol=1e-8)

    if rank == n:
        # Fully controllable — use characteristic polynomial matching
        # For SISO: solve via Ackermann or direct polynomial matching
        return _place_full_ctrl(A, B, desired)
    else:
        # Partially controllable — use controllable decomposition
        return _place_partial_ctrl(A, B, desired)

def _place_full_ctrl(A, B, desired):
    """Place poles for fully controllable system (SISO or MIMO)."""
    n = A.shape[0]
    B = np.atleast_2d(B)
    if B.shape[0] != n:
        B = B.T
    m = B.shape[1]  # number of inputs
    desired = np.array(desired, dtype=complex)

    if m == 1:
        # SISO: Bass-Gura type formula
        a_cur = np.real(np.poly(np.linalg.eigvals(A)))
        a_des = np.real(np.poly(desired))
        U = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                P[i, j] = a_cur[j - i]
        T = U @ P
        delta = a_des[1:] - a_cur[1:]
        K = -(delta @ np.linalg.inv(T)).reshape(1, -1)
        return K
    else:
        # MIMO: try scipy place_poles, then Sylvester fallback
        desired_real = np.array(desired)
        try:
            res = place_poles(A, B, desired_real, method='YT')
            return -res.gain_matrix
        except:
            pass
        # Sylvester fallback
        coeffs = np.real(np.poly(desired))
        Lambda = np.zeros((n, n))
        if n > 1:
            Lambda[:-1, 1:] = np.eye(n - 1)
        Lambda[-1, :] = -coeffs[1:][::-1]
        Psi = np.zeros((m, n))
        for i in range(min(m, n)):
            Psi[i, i] = 1.0
        X = la.solve_sylvester(A, -Lambda, -B @ Psi)
        K = np.real(Psi @ np.linalg.inv(X))
        return K

def _place_partial_ctrl(A, B, desired):
    """Place poles for partially controllable system via decomposition."""
    n = A.shape[0]
    B = np.atleast_2d(B)
    if B.shape[0] != n:
        B = B.T
    m = B.shape[1]
    desired = np.array(desired, dtype=complex)

    # Controllable decomposition
    Umat = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(Umat, tol=1e-8)

    # Find basis for controllable subspace
    Q_full, R, piv = la.qr(Umat, pivoting=True)
    T_ctrl = Q_full[:, :rank]
    T_rest = Q_full[:, rank:]
    T_transform = np.hstack([T_ctrl, T_rest])

    T_inv = np.linalg.inv(T_transform)
    A_new = T_inv @ A @ T_transform
    B_new = T_inv @ B

    Ac = A_new[:rank, :rank]
    Auc = A_new[rank:, rank:]
    Bc = B_new[:rank, :]

    eigs_uc = np.linalg.eigvals(Auc)

    remaining_desired = list(desired)
    for euc in eigs_uc:
        dists = [abs(complex(d) - euc) for d in remaining_desired]
        idx = np.argmin(dists)
        if dists[idx] > 0.5:
            print(f"  Warning: uncontrollable eig {euc} not found in desired spectrum")
        remaining_desired.pop(idx)

    ctrl_desired = np.array(remaining_desired, dtype=complex)

    Kc = _place_full_ctrl(Ac, Bc, ctrl_desired)
    Kc = np.atleast_2d(Kc)

    # Full gain: K_new = [Kc, 0]
    K_new = np.zeros((m, n))
    K_new[:, :rank] = Kc

    K = K_new @ T_inv
    return np.real(K)

# ============================================================
# Hautus controllability test for single eigenvalue
# ============================================================
def hautus_ctrl_rank(A, B, lam):
    n = A.shape[0]
    H = np.hstack([lam * np.eye(n) - A, B])
    return np.linalg.matrix_rank(H, tol=1e-8)

def hautus_obs_rank(A, C, lam):
    n = A.shape[0]
    H = np.vstack([lam * np.eye(n) - A, C])
    return np.linalg.matrix_rank(H, tol=1e-8)

# ============================================================
#   ЗАДАНИЕ 1: МОДАЛЬНЫЙ РЕГУЛЯТОР
# ============================================================
print("=" * 60)
print("ЗАДАНИЕ 1: МОДАЛЬНЫЙ РЕГУЛЯТОР")
print("=" * 60)

A1 = np.array([[12., -1, 14],
               [6., 0, 6],
               [-6., -2, -8]])
B1 = np.array([[11.], [7.], [-7.]])
n1 = 3
x0_1 = np.array([1., 1., 1.])

V("T1_A", mat2typ(A1))
V("T1_B", mat2typ(B1))

# Eigenvalues
eigs1 = np.linalg.eigvals(A1)
eigs1_sorted = sorted(eigs1, key=lambda x: (x.real, x.imag))
print(f"Eigenvalues of A: {eigs1_sorted}")
V("T1_eigs", eigs_str(eigs1))

# Controllability matrix
U1 = np.hstack([np.linalg.matrix_power(A1, i) @ B1 for i in range(n1)])
rank_U1 = np.linalg.matrix_rank(U1)
det_U1 = np.linalg.det(U1)
print(f"Controllability matrix rank: {rank_U1}, det: {det_U1:.4f}")
V("T1_U", mat2typ(U1))
V("T1_rank_U", str(rank_U1))
V("T1_det_U", fmt(det_U1))

# Hautus test for each eigenvalue
unique_eigs1 = []
seen = set()
for e in eigs1_sorted:
    key = (round(e.real, 6), round(e.imag, 6))
    if key not in seen:
        unique_eigs1.append(e)
        seen.add(key)
        # also add conjugate pair key
        if abs(e.imag) > 1e-8:
            seen.add((round(e.real, 6), -round(e.imag, 6)))

hautus_ranks_1 = []
for e in unique_eigs1:
    r = hautus_ctrl_rank(A1, B1, e)
    hautus_ranks_1.append(r)
    ctrl = "управляемо" if r == n1 else "НЕУПРАВЛЯЕМО"
    print(f"  λ={fmt(e)}: rank(H)={r} → {ctrl}")

V("T1_hautus_ranks", ", ".join(str(r) for r in hautus_ranks_1))

# Controllability / stabilizability conclusion
fully_ctrl = rank_U1 == n1
stabilizable = all(
    hautus_ctrl_rank(A1, B1, e) == n1
    for e in eigs1 if e.real >= -1e-10
)
V("T1_fully_ctrl", "да" if fully_ctrl else "нет")
V("T1_stabilizable", "да" if stabilizable else "нет")

# Desired spectra
spectra_1 = [
    {"name": "I",   "poles": [-1., -1., -1.],          "label": r"$\{-1,-1,-1\}$"},
    {"name": "II",  "poles": [-2., -2., -2.],          "label": r"$\{-2,-2,-2\}$"},
    {"name": "III", "poles": [-1., -10., -100.],       "label": r"$\{-1,-10,-100\}$"},
    {"name": "IV",  "poles": [-2., -20., -200.],       "label": r"$\{-2,-20,-200\}$"},
    {"name": "V",   "poles": [-1., -1.+3j, -1.-3j],   "label": r"$\{-1,-1\pm 3i\}$"},
    {"name": "VI",  "poles": [-2., -2.+6j, -2.-6j],   "label": r"$\{-2,-2\pm 6i\}$"},
]

# Determine achievability
# If fully controllable, all are achievable
# If not, only spectra containing the uncontrollable eigenvalue(s) are achievable
uncontrollable_eigs = []
for e in eigs1:
    if hautus_ctrl_rank(A1, B1, e) < n1:
        uncontrollable_eigs.append(e)

for sp in spectra_1:
    if fully_ctrl:
        sp["achievable"] = True
    else:
        # Check: each uncontrollable eigenvalue must appear in desired spectrum
        achievable = True
        for ue in uncontrollable_eigs:
            found = False
            for dp in sp["poles"]:
                if abs(complex(dp) - ue) < 1e-6:
                    found = True
                    break
            if not found:
                achievable = False
                break
        sp["achievable"] = achievable
    print(f"  Spectrum {sp['name']} {sp['label']}: {'ACHIEVABLE' if sp['achievable'] else 'NOT achievable'}")

achievable_spectra = [sp for sp in spectra_1 if sp["achievable"]]
not_achievable_spectra = [sp for sp in spectra_1 if not sp["achievable"]]

V("T1_achievable_list", ", ".join(sp["name"] for sp in achievable_spectra))
V("T1_not_achievable_list", ", ".join(sp["name"] for sp in not_achievable_spectra))

# Synthesize K for each achievable spectrum
results_1 = []
for idx, sp in enumerate(achievable_spectra):
    poles = np.array(sp["poles"])
    print(f"\n--- Spectrum {sp['name']}: {sp['poles']} ---")

    # Method 1: scipy place_poles
    K = place_K(A1, B1, poles)
    eigs_cl = np.linalg.eigvals(A1 + B1 @ K)
    print(f"  K (place) = {K}")
    print(f"  eig(A+BK) = {sorted(eigs_cl, key=lambda x: (x.real, x.imag))}")

    sp["K"] = K
    sp["eigs_cl"] = sorted(eigs_cl, key=lambda x: (x.real, x.imag))

    V(f"T1_K_{sp['name']}", mat2typ(K))
    V(f"T1_eigs_cl_{sp['name']}", eigs_str(eigs_cl))

    results_1.append(sp)

# For one spectrum, demonstrate all 3 methods (Sylvester, Ackermann, Bass-Gura)
# Since system is not fully controllable, work with controllable subsystem
if len(achievable_spectra) > 0:
    demo_sp = achievable_spectra[0]
    demo_poles = np.array(demo_sp["poles"])
    print(f"\n=== Демонстрация 3 методов для спектра {demo_sp['name']} ===")

    if fully_ctrl:
        # Direct application on full system
        try:
            K_ack = ackermann(A1, B1, demo_poles)
            eigs_ack = np.linalg.eigvals(A1 + B1 @ K_ack)
            print(f"  Ackermann K = {K_ack}, eig = {eigs_str(eigs_ack)}")
            V("T1_K_ackermann", mat2typ(K_ack))
            V("T1_eigs_ackermann", eigs_str(eigs_ack))
        except Exception as e:
            print(f"  Ackermann failed: {e}")

        try:
            K_bg = bass_gura(A1, B1, demo_poles)
            eigs_bg = np.linalg.eigvals(A1 + B1 @ K_bg)
            print(f"  Bass-Gura K = {K_bg}, eig = {eigs_str(eigs_bg)}")
            V("T1_K_bass_gura", mat2typ(K_bg))
            V("T1_eigs_bass_gura", eigs_str(eigs_bg))
        except Exception as e:
            print(f"  Bass-Gura failed: {e}")

        try:
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
            K_syl = sylvester_method(A1, B1, demo_poles)
            eigs_syl = np.linalg.eigvals(A1 + B1 @ K_syl)
            print(f"  Sylvester K = {K_syl}, eig = {eigs_str(eigs_syl)}")
            V("T1_K_sylvester", mat2typ(K_syl))
            V("T1_eigs_sylvester", eigs_str(eigs_syl))
        except Exception as e:
            print(f"  Sylvester failed: {e}")
    else:
        # Apply methods to controllable subsystem, then transform back
        # Controllable decomposition
        Uctrl = np.hstack([np.linalg.matrix_power(A1, i) @ B1 for i in range(n1)])
        rank_c = np.linalg.matrix_rank(Uctrl)
        Q_f, R_f, piv = la.qr(Uctrl, pivoting=True)
        T_c = Q_f[:, :rank_c]
        T_r = Q_f[:, rank_c:]
        T_tr = np.hstack([T_c, T_r])
        T_inv = np.linalg.inv(T_tr)

        Ac = (T_inv @ A1 @ T_tr)[:rank_c, :rank_c]
        Bc = (T_inv @ B1)[:rank_c]

        # Determine controllable desired poles
        Auc = (T_inv @ A1 @ T_tr)[rank_c:, rank_c:]
        eigs_uc = np.linalg.eigvals(Auc)
        ctrl_desired = list(demo_poles.copy())
        for euc in eigs_uc:
            dists = [abs(complex(d) - euc) for d in ctrl_desired]
            idx_rm = np.argmin(dists)
            ctrl_desired.pop(idx_rm)
        ctrl_desired = np.array(ctrl_desired, dtype=complex)

        print(f"  Controllable subsystem: Ac={Ac.shape}, Bc={Bc.shape}")
        print(f"  Controllable desired poles: {ctrl_desired}")

        # Ackermann on controllable subsystem
        try:
            Kc_ack = ackermann(Ac, Bc, ctrl_desired)
            K_full = np.zeros((1, n1))
            K_full[0, :rank_c] = Kc_ack.flatten()
            K_ack = K_full @ T_inv
            eigs_ack = np.linalg.eigvals(A1 + B1 @ K_ack)
            print(f"  Ackermann K = {K_ack}, eig = {eigs_str(eigs_ack)}")
            V("T1_K_ackermann", mat2typ(K_ack))
            V("T1_eigs_ackermann", eigs_str(eigs_ack))
        except Exception as e:
            print(f"  Ackermann failed: {e}")

        # Bass-Gura on controllable subsystem
        try:
            Kc_bg = bass_gura(Ac, Bc, ctrl_desired)
            K_full = np.zeros((1, n1))
            K_full[0, :rank_c] = Kc_bg.flatten()
            K_bg = K_full @ T_inv
            eigs_bg = np.linalg.eigvals(A1 + B1 @ K_bg)
            print(f"  Bass-Gura K = {K_bg}, eig = {eigs_str(eigs_bg)}")
            V("T1_K_bass_gura", mat2typ(K_bg))
            V("T1_eigs_bass_gura", eigs_str(eigs_bg))
        except Exception as e:
            print(f"  Bass-Gura failed: {e}")

        # Sylvester on full system (works even for non-fully-ctrl if spectrum is achievable)
        try:
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
            K_syl = sylvester_method(A1, B1, demo_poles)
            eigs_syl = np.linalg.eigvals(A1 + B1 @ K_syl)
            print(f"  Sylvester K = {K_syl}, eig = {eigs_str(eigs_syl)}")
            V("T1_K_sylvester", mat2typ(K_syl))
            V("T1_eigs_sylvester", eigs_str(eigs_syl))
        except Exception as e:
            print(f"  Sylvester failed: {e}")

# Simulation for each achievable spectrum
t_sim = np.linspace(0, 10, 2000)

# Plot all u(t) on one figure
fig_u, ax_u = plt.subplots(1, 1, figsize=(10, 5))
u_ts = []
fig_x, axes_x = plt.subplots(len(achievable_spectra), 1,
                              figsize=(10, 4 * len(achievable_spectra)),
                              squeeze=False)

for idx, sp in enumerate(achievable_spectra):
    K = sp["K"]

    def dynamics(t, x, A=A1, B=B1, K=K):
        u = (K @ x.reshape(-1, 1)).item()
        dx = (A @ x.reshape(-1, 1) + B * u).flatten()
        return dx

    sol = solve_ivp(dynamics, [0, t_sim[-1]], x0_1, t_eval=t_sim,
                    method='RK45', rtol=1e-10, atol=1e-12)

    x_t = sol.y  # (3, N)
    u_t = np.array([(K @ x_t[:, i:i+1]).item() for i in range(x_t.shape[1])])

    # u(t) plot
    ax_u.plot(sol.t, u_t, label=f"{sp['name']}: {sp['label']}")
    u_ts.append(u_t)

    # x(t) plot
    ax = axes_x[idx, 0]
    for j in range(n1):
        ax.plot(sol.t, x_t[j], label=f'$x_{j+1}(t)$')
    ax.set_title(f"Спектр {sp['name']}: {sp['label']}")
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend(loc='best')
    
    # Save isolated plot
    fig_single, ax_single = plt.subplots(n1, 1, figsize=(10, 6), sharex=True)
    if n1 == 1:
        ax_single = [ax_single]
    for j in range(n1):
        ax_single[j].plot(sol.t, x_t[j], label=f'$x_{j+1}(t)$')
        ax_single[j].legend(loc='best')
        ax_single[j].grid(True, alpha=0.3)
    ax_single[-1].set_xlabel('t')
    t_max_x = find_visual_end(sol.t, x_t)
    ax_single[-1].set_xlim(0, t_max_x)
    fig_single.suptitle(f"Спектр {sp['name']}: {sp['label']}")
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(img_dir, f'task1_x_{sp["name"]}.png'), bbox_inches='tight')
    plt.close(fig_single)

ax_u.set_xlabel('t')
ax_u.set_ylabel('u(t)')
ax_u.set_title('Управление u(t) для различных спектров')
ax_u.legend(loc='best')
if u_ts:
    t_max_u = find_visual_end(sol.t, u_ts)
    ax_u.set_xlim(0, t_max_u)
fig_u.tight_layout()
fig_u.savefig(os.path.join(img_dir, 'task1_u.png'), bbox_inches='tight')
plt.close(fig_u)



print("\nЗадание 1: графики сохранены")

# ============================================================
#   ЗАДАНИЕ 2: НАБЛЮДАТЕЛЬ ПОЛНОГО ПОРЯДКА
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: НАБЛЮДАТЕЛЬ ПОЛНОГО ПОРЯДКА")
print("=" * 60)

A2 = np.array([[-35., 11, 6, 11],
               [-56., 17, 10, 18],
               [-22., 7, 5, 6],
               [-42., 12, 10, 13]])
CT2 = np.array([[-1.], [0.], [0.], [1.]])
C2 = CT2.T  # (1, 4)
n2 = 4

V("T2_A", mat2typ(A2))
V("T2_C", mat2typ(C2))

# Eigenvalues
eigs2 = np.linalg.eigvals(A2)
print(f"Eigenvalues of A: {sorted(eigs2, key=lambda x: (x.real, x.imag))}")
V("T2_eigs", eigs_str(eigs2))

# Observability matrix
O2 = np.vstack([C2 @ np.linalg.matrix_power(A2, i) for i in range(n2)])
rank_O2 = np.linalg.matrix_rank(O2)
print(f"Observability rank: {rank_O2}")
V("T2_O", mat2typ(O2))
V("T2_rank_O", str(rank_O2))

# Hautus observability test
unique_eigs2 = []
seen2 = set()
for e in sorted(eigs2, key=lambda x: (x.real, x.imag)):
    key = (round(e.real, 6), round(e.imag, 6))
    if key not in seen2:
        unique_eigs2.append(e)
        seen2.add(key)
        if abs(e.imag) > 1e-8:
            seen2.add((round(e.real, 6), -round(e.imag, 6)))

hautus_obs_ranks_2 = []
for e in unique_eigs2:
    r = hautus_obs_rank(A2, C2, e)
    hautus_obs_ranks_2.append(r)
    obs = "наблюдаемо" if r == n2 else "НЕНАБЛЮДАЕМО"
    print(f"  λ={fmt(e)}: rank(H_obs)={r} → {obs}")

V("T2_hautus_ranks", ", ".join(str(r) for r in hautus_obs_ranks_2))
V("T2_fully_obs", "да" if rank_O2 == n2 else "нет")

# Desired spectra for observer
spectra_2 = [
    {"name": "I",   "poles": [-7., -7., -7., -7.],      "label": r"$\{-7,-7,-7,-7\}$"},
    {"name": "II",  "poles": [-7., -70., -700., -7000.], "label": r"$\{-7,-70,-700,-7000\}$"},
    {"name": "III", "poles": [-7.+8j, -7.-8j, -7.+9j, -7.-9j], "label": r"$\{-7\pm 8i, -7\pm 9i\}$"},
]

# Synthesize L for each spectrum
# Observer: x_hat_dot = A x_hat + L(C x_hat - y), error: e_dot = (A + LC)e
# Dual: eig(A + LC) = eig(A^T + C^T L^T)
# Use place_poles on (A^T, C^T) to find L^T
x0_2 = np.array([1., 1., 1., 1.])
xhat0_2 = np.array([0., 0., 0., 0.])
t_sim2 = np.linspace(0, 5, 2000)

for sp in spectra_2:
    poles = np.array(sp["poles"])
    print(f"\n--- Observer Spectrum {sp['name']}: {sp['poles']} ---")

    # Place poles of (A + LC): dual problem = place poles of (A^T + C^T L^T)
    # Use our robust place_K: K_dual such that eig(A^T + C^T K_dual) = desired
    K_dual = place_K(A2.T, CT2, poles)
    L = K_dual.T  # L is (4, 1)

    eigs_obs = np.linalg.eigvals(A2 + L @ C2)
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
    V(f"T2_Syl_P_{sp['name']}", mat2typ(P_obs))
    print(f"  L = {L.T}")
    print(f"  eig(A+LC) = {sorted(eigs_obs, key=lambda x: (x.real, x.imag))}")

    sp["L"] = L
    sp["eigs_obs"] = eigs_obs

    V(f"T2_L_{sp['name']}", mat2typ(L))
    V(f"T2_eigs_obs_{sp['name']}", eigs_str(eigs_obs))

# Simulate observer for each spectrum
for sp in spectra_2:
    L = sp["L"]

    def obs_dynamics(t, state, A=A2, C=C2, L=L):
        x = state[:n2]
        xhat = state[n2:]
        dx = A @ x
        y = C @ x
        dxhat = A @ xhat + (L @ (C @ xhat - y)).flatten()
        return np.concatenate([dx, dxhat])

    sol = solve_ivp(obs_dynamics, [0, t_sim2[-1]],
                    np.concatenate([x0_2, xhat0_2]),
                    t_eval=t_sim2, method='RK45', rtol=1e-10, atol=1e-12)

    x_t = sol.y[:n2]
    xhat_t = sol.y[n2:]
    e_t = x_t - xhat_t

    t_max2 = find_visual_end(sol.t, np.vstack([x_t, e_t]))

    # x(t) and x_hat(t)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    for j in range(n2):
        ax1.plot(sol.t, x_t[j], label=f'$x_{j+1}$')
        ax1.plot(sol.t, xhat_t[j], '--', label=f'$\\hat{{x}}_{j+1}$')
    ax1.set_title(f'Наблюдатель, спектр {sp["name"]}: $x(t)$ и $\\hat{{x}}(t)$')
    ax1.set_xlabel('t')
    ax1.set_xlim(0, t_max2)
    ax1.legend(loc='best', fontsize=7)
    fig1.tight_layout()
    fig1.savefig(os.path.join(img_dir, f'task2_obs_{sp["name"]}_x.png'), bbox_inches='tight')
    plt.close(fig1)

    # Error
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for j in range(n2):
        ax2.plot(sol.t, e_t[j], label=f'$e_{j+1}$')
    ax2.set_title(f'Наблюдатель, спектр {sp["name"]}: Ошибка $e(t) = x(t) - \\hat{{x}}(t)$')
    ax2.set_xlabel('t')
    ax2.set_xlim(0, t_max2)
    ax2.legend(loc='best')
    fig2.tight_layout()
    fig2.savefig(os.path.join(img_dir, f'task2_obs_{sp["name"]}_e.png'), bbox_inches='tight')
    plt.close(fig2)

    # Error norm
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    e_norm = np.linalg.norm(e_t, axis=0)
    ax3.plot(sol.t, e_norm)
    ax3.set_title(f'Наблюдатель, спектр {sp["name"]}: $||e(t)||$')
    ax3.set_xlabel('t')
    ax3.set_yscale('log')
    ax3.set_xlim(0, t_max2)
    fig3.tight_layout()
    fig3.savefig(os.path.join(img_dir, f'task2_obs_{sp["name"]}_enorm.png'), bbox_inches='tight')
    plt.close(fig3)

print("\nЗадание 2: графики сохранены")

# ============================================================
#   ЗАДАНИЕ 3: МОДАЛЬНОЕ УПРАВЛЕНИЕ ПО ВЫХОДУ
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 3: МОДАЛЬНОЕ УПРАВЛЕНИЕ ПО ВЫХОДУ")
print("=" * 60)

A3 = np.array([[5., -7, -5, 1],
               [-7., 5, -1, 5],
               [-5., -1, 5, 7],
               [1., 5, 7, 5]])
B3 = np.array([[5.], [7.], [1.], [9.]])
C3 = np.array([[0., 0, 2, 2],
               [1., 1, -1, 1]])
D3 = np.array([[4.], [2.]])
n3 = 4

V("T3_A", mat2typ(A3))
V("T3_B", mat2typ(B3))
V("T3_C", mat2typ(C3))
V("T3_D", mat2typ(D3))

# Eigenvalues
eigs3 = np.linalg.eigvals(A3)
print(f"Eigenvalues of A: {sorted(eigs3, key=lambda x: (x.real, x.imag))}")
V("T3_eigs", eigs_str(eigs3))

# Controllability
U3 = np.hstack([np.linalg.matrix_power(A3, i) @ B3 for i in range(n3)])
rank_U3 = np.linalg.matrix_rank(U3)
print(f"Controllability rank: {rank_U3}")
V("T3_rank_U", str(rank_U3))

# Observability
O3 = np.vstack([C3 @ np.linalg.matrix_power(A3, i) for i in range(n3)])
rank_O3 = np.linalg.matrix_rank(O3)
print(f"Observability rank: {rank_O3}")
V("T3_rank_O", str(rank_O3))

# Hautus tests
unique_eigs3 = []
seen3 = set()
for e in sorted(eigs3, key=lambda x: (x.real, x.imag)):
    key = (round(e.real, 6), round(e.imag, 6))
    if key not in seen3:
        unique_eigs3.append(e)
        seen3.add(key)
        if abs(e.imag) > 1e-8:
            seen3.add((round(e.real, 6), -round(e.imag, 6)))

for e in unique_eigs3:
    rc = hautus_ctrl_rank(A3, B3, e)
    ro = hautus_obs_rank(A3, C3, e)
    print(f"  λ={fmt(e)}: ctrl_rank={rc}, obs_rank={ro}")

V("T3_fully_ctrl", "да" if rank_U3 == n3 else "нет")
V("T3_fully_obs", "да" if rank_O3 == n3 else "нет")

# Choose desired spectra for regulator and observer
# System is fully controllable but NOT fully observable (rank_O3 = 3)
# Unobservable eigenvalue:
unobs_eigs3 = []
for e in eigs3:
    if hautus_obs_rank(A3, C3, e) < n3:
        unobs_eigs3.append(e)
        print(f"  Unobservable eigenvalue: {fmt(e)}")

# Stabilizability: fully controllable => yes
# Detectability: unobs eigenvalue must be stable
detectable = all(e.real < 0 for e in unobs_eigs3)
print(f"  Detectable: {'yes' if detectable else 'NO'}")
V("T3_detectable", "да" if detectable else "нет")

# Regulator: all eigenvalues are controllable, place all 4
reg_poles_3 = np.array([-3., -4., -5., -6.])
# Observer: unobs eigenvalue stays fixed, place the remaining 3
# Unobs eigenvalue is -8 (stable), so observer spectrum must contain -8
obs_poles_3_ctrl = np.array([-15., -16., -17.])  # for observable part
obs_poles_3_full = np.sort(np.concatenate([obs_poles_3_ctrl, [e.real for e in unobs_eigs3]]))

print(f"\nRegulator desired poles: {reg_poles_3}")
print(f"Observer desired poles: {obs_poles_3_full} (including unobs {[fmt(e) for e in unobs_eigs3]})")

V("T3_reg_poles", eigs_str(reg_poles_3))
V("T3_obs_poles", eigs_str(obs_poles_3_full))

# Synthesize K (regulator) - full controllability
K3 = place_K(A3, B3, reg_poles_3)
eigs_cl3 = np.linalg.eigvals(A3 + B3 @ K3)
print(f"K = {K3}")
print(f"eig(A+BK) = {sorted(eigs_cl3, key=lambda x: (x.real, x.imag))}")
V("T3_K", mat2typ(K3))
V("T3_eigs_cl", eigs_str(eigs_cl3))

# Synthesize L (observer) - partial observability
# Use place_K on dual system (A^T, C^T)
K_dual3 = place_K(A3.T, C3.T, obs_poles_3_full)
L3 = K_dual3.T  # L is (4, 2)
eigs_obs3 = np.linalg.eigvals(A3 + L3 @ C3)
print(f"L = {L3}")
print(f"eig(A+LC) = {sorted(eigs_obs3, key=lambda x: (x.real, x.imag))}")
V("T3_L", mat2typ(L3))
V("T3_eigs_obs", eigs_str(eigs_obs3))

# Simulate the output feedback system
# System: x_dot = A x + B u, y = C x + D u
# Observer: x_hat_dot = A x_hat + (B + L D) u + L (C x_hat - y)
# Control: u = K x_hat
x0_3 = np.array([1., 1., 1., 1.])
xhat0_3 = np.array([0., 0., 0., 0.])
t_sim3 = np.linspace(0, 5, 2000)

def task3_dynamics(t, state):
    x = state[:n3]
    xhat = state[n3:]
    u = (K3 @ xhat).item()
    dx = A3 @ x + B3.flatten() * u
    y = C3 @ x + D3.flatten() * u
    # Observer: xhat_dot = A xhat + (B + LD) u + L(C xhat - y)
    innovation = C3 @ xhat - y
    dxhat = A3 @ xhat + (B3 + L3 @ D3).flatten() * u + (L3 @ innovation.reshape(-1, 1)).flatten()
    return np.concatenate([dx, dxhat])

sol3 = solve_ivp(task3_dynamics, [0, t_sim3[-1]],
                 np.concatenate([x0_3, xhat0_3]),
                 t_eval=t_sim3, method='RK45', rtol=1e-10, atol=1e-12)

x3_t = sol3.y[:n3]
xhat3_t = sol3.y[n3:]
e3_t = x3_t - xhat3_t
u3_t = np.array([(K3 @ xhat3_t[:, i:i+1]).item() for i in range(xhat3_t.shape[1])])
y3_t = np.array([(C3 @ x3_t[:, i:i+1] + D3 * u3_t[i]).flatten() for i in range(x3_t.shape[1])]).T

t_max3 = find_visual_end(sol3.t, np.vstack([x3_t, e3_t]))

# u(t)
fig_u, ax_u = plt.subplots(figsize=(8, 4))
ax_u.plot(sol3.t, u3_t)
ax_u.set_title('Задание 3: Управление $u(t)$')
ax_u.set_xlabel('t')
ax_u.set_xlim(0, t_max3)
fig_u.tight_layout()
fig_u.savefig(os.path.join(img_dir, 'task3_u.png'), bbox_inches='tight')
plt.close(fig_u)

# y(t)
fig_y, ax_y = plt.subplots(figsize=(8, 4))
for j in range(C3.shape[0]):
    ax_y.plot(sol3.t, y3_t[j], label=f'$y_{j+1}$')
ax_y.set_title('Задание 3: Выход $y(t)$')
ax_y.set_xlabel('t')
ax_y.set_xlim(0, t_max3)
ax_y.legend()
fig_y.tight_layout()
fig_y.savefig(os.path.join(img_dir, 'task3_y.png'), bbox_inches='tight')
plt.close(fig_y)

# x(t) vs x_hat(t)
fig_x, ax_x = plt.subplots(figsize=(8, 4))
for j in range(n3):
    ax_x.plot(sol3.t, x3_t[j], label=f'$x_{j+1}$')
    ax_x.plot(sol3.t, xhat3_t[j], '--', label=f'$\\hat{{x}}_{j+1}$')
ax_x.set_title('Задание 3: $x(t)$ и $\\hat{x}(t)$')
ax_x.set_xlabel('t')
ax_x.set_xlim(0, t_max3)
ax_x.legend(loc='best', fontsize=7)
fig_x.tight_layout()
fig_x.savefig(os.path.join(img_dir, 'task3_x.png'), bbox_inches='tight')
plt.close(fig_x)

# e(t)
fig_e, ax_e = plt.subplots(figsize=(8, 4))
for j in range(n3):
    ax_e.plot(sol3.t, e3_t[j], label=f'$e_{j+1}$')
ax_e.set_title('Задание 3: Ошибка $e(t)$')
ax_e.set_xlabel('t')
ax_e.set_xlim(0, t_max3)
ax_e.legend()
fig_e.tight_layout()
fig_e.savefig(os.path.join(img_dir, 'task3_e.png'), bbox_inches='tight')
plt.close(fig_e)

print("\nЗадание 3: графики сохранены")

# ============================================================
#   ЗАДАНИЕ 4: НАБЛЮДАТЕЛЬ ПОНИЖЕННОГО ПОРЯДКА
# ============================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 4: НАБЛЮДАТЕЛЬ ПОНИЖЕННОГО ПОРЯДКА")
print("=" * 60)

# Same A, B, D as Task 3
A4 = A3.copy()
B4 = B3.copy()
D4 = D3.copy()
# C from Table 5, №4
C4 = np.array([[0., 1, 0, 0],
               [1., 0, 0, 0]])
n4 = 4
p4 = C4.shape[0]  # 2 outputs
q4 = n4 - p4  # 2 reduced-order observer states

V("T4_C", mat2typ(C4))

# Observability with new C
O4 = np.vstack([C4 @ np.linalg.matrix_power(A4, i) for i in range(n4)])
rank_O4 = np.linalg.matrix_rank(O4)
print(f"Observability rank (new C): {rank_O4}")
V("T4_rank_O", str(rank_O4))

# Hautus observability with new C
for e in unique_eigs3:
    ro = hautus_obs_rank(A4, C4, e)
    print(f"  λ={fmt(e)}: obs_rank={ro}")

# Use K from Task 3
K4 = K3.copy()
V("T4_K", mat2typ(K4))

# Reduced-order observer design
# C4 measures y1=x2, y2=x1 (from C4 = [[0,1,0,0],[1,0,0,0]])
# So measured states: x1, x2. Unmeasured: x3, x4.

# Partition: let's find transformation
# y = C x => we can extract x_measured from y
# With C4 = [[0,1,0,0],[1,0,0,0]], y = [x2, x1]^T
# So x_measured = [x1, x2] (after reordering) and x_unmeasured = [x3, x4]

# Build permutation/transformation
# We need T such that C4 @ T = [I_p | 0]
# C4 @ [col2, col1, col3, col4] = [[1,0,0,0],[0,1,0,0]] ✓
# So T permutes columns: [2,1,3,4] -> [1,2,3,4]
# Actually C4 = [[0,1,0,0],[1,0,0,0]]
# C4 @ e1 = [0,1], C4 @ e2 = [1,0], C4 @ e3 = [0,0], C4 @ e4 = [0,0]
# So if we permute to [e2, e1, e3, e4]:
# C4 @ [e2,e1,e3,e4] = [[1,0,0,0],[0,1,0,0]] = [I | 0]

perm = np.array([[0,1,0,0],
                  [1,0,0,0],
                  [0,0,1,0],
                  [0,0,0,1]], dtype=float)
T_perm = perm.T  # columns of T are permuted basis vectors

# Check: C4 @ T_perm should give [I_2 | 0]
CT_check = C4 @ T_perm
print(f"C @ T = {CT_check}")  # should be [[1,0,0,0],[0,1,0,0]]

# Transform system: x = T_perm @ x_new
# A_new = T_perm^{-1} @ A @ T_perm
# B_new = T_perm^{-1} @ B
A_new = np.linalg.inv(T_perm) @ A4 @ T_perm
B_new = np.linalg.inv(T_perm) @ B4

# Partition A_new into blocks
# A_new = [[A11, A12], [A21, A22]] where A11 is p×p, A22 is q×q
A11 = A_new[:p4, :p4]
A12 = A_new[:p4, p4:]
A21 = A_new[p4:, :p4]
A22 = A_new[p4:, p4:]
B1_new = B_new[:p4]
B2_new = B_new[p4:]

print(f"A11 = {A11}")
V("T4_A11", mat2typ(A11))
V("T4_A12", mat2typ(A12))
V("T4_A21", mat2typ(A21))
V("T4_A22", mat2typ(A22))
V("T4_T", mat2typ(T_perm))
print(f"A12 = {A12}")
print(f"A21 = {A21}")
print(f"A22 = {A22}")

# Reduced-order observer:
# z_hat_dot = Gamma @ z_hat - Y @ y + (Q @ B_new + Y @ D_new) @ u
# where Gamma = A22 + Q @ A12 is the observer dynamics matrix
# Need to place eigenvalues of Gamma
# This is dual to placing eigenvalues via feedback:
# Gamma = A22 + Q @ A12 where Q is 2×2 and A12 is 2×2

# Choose desired spectrum for reduced-order observer
# Should be faster than the full-order observer
ro_poles = np.array([-20., -25.])
print(f"\nReduced-order observer desired poles: {ro_poles}")
V("T4_ro_poles", eigs_str(ro_poles))

# Place poles of Gamma = A22 + Q @ A12
# This is: eig(A22 + Q @ A12) = ro_poles
# Transpose: eig(A22^T + A12^T @ Q^T) = ro_poles
# This is a state feedback problem: (A22^T, A12^T) with gain Q^T
# Place poles of Gamma = A22 + Q @ A12
# Dual: eig(A22^T + A12^T Q^T) = desired
try:
    res4 = place_poles(A22.T, A12.T, ro_poles, method='YT')
    Q = -res4.gain_matrix.T
except:
    K_ro = place_K(A22.T, A12.T, ro_poles)
    Q = K_ro.T

Gamma = A22 + Q @ A12
Y = A21 + Q @ A11  # Y matrix for observer
eigs_ro = np.linalg.eigvals(Gamma)
print(f"Q = {Q}")
print(f"Gamma = A22 + Q A12 = {Gamma}")
print(f"eig(Gamma) = {sorted(eigs_ro, key=lambda x: (x.real, x.imag))}")

V("T4_Q", mat2typ(Q))
V("T4_Gamma", mat2typ(Gamma))
V("T4_Y", mat2typ(Y))
V("T4_eigs_ro", eigs_str(eigs_ro))

# Simulate reduced-order observer
# Full state reconstruction:
# x_hat = [C; Q]^{-1} @ [y - D u; z_hat]
# In our transformed coordinates: x_new_measured = y (from C_new = [I | 0])
# x_new_unmeasured = z_hat - Q @ y (approximately)
# Actually, the standard reduced-order observer:
# z_hat_dot = Gamma z_hat - Y y + (Q B_new_measured + B_new_unmeasured + (Y D_new - something) u)
# Let me implement this carefully.

# In the transformed coordinates where C_new = [I_p | 0]:
# x_m = y (measured part, first p components)
# x_u (unmeasured part, last q components)
#
# Define w = x_u + Q x_m (auxiliary variable)
# w_dot = (A22 + Q A12) w + (A21 + Q A11 - (A22 + Q A12) Q) y
#         + (B2_new + Q B1_new) u
#       = Gamma w + (A21 + Q A11 - Gamma Q) y + (B2_new + Q B1_new) u
#
# x_u_hat = w - Q y
# x_hat = T_perm @ [y; x_u_hat]   (in original coords)

# For the system y = Cx + Du, so in transformed coords:
# y = x_m (since C_new = [I|0] and D contribution handled separately)
# But we have D != 0, so y = C x + D u = x_m + D_new u...
# Wait, actually in transformed coordinates:
# C_new = C4 @ T_perm = [I_p | 0]
# y = C_new @ x_new + D u = x_new_measured + D u
# So x_m = y - D u

# Let me redo this properly for the D != 0 case.
# The system is:
# x_dot = A x + B u
# y = C x + D u
#
# In transformed coordinates x_new = T_perm^{-1} x:
# x_new_dot = A_new x_new + B_new u
# y = C_new x_new + D u = [I | 0] x_new + D u
# => x_m = y - D u, where x_m = x_new[:p]

D_new = D4  # D doesn't change under state transformation

x0_4 = np.array([1., 1., 1., 1.])
zhat0_4 = np.array([0., 0.])
t_sim4 = np.linspace(0, 5, 2000)

# Precompute matrices for reduced-order observer
BQ = B2_new + Q @ B1_new  # input matrix for w dynamics
Gamma_Q_term = A21 + Q @ A11 - Gamma @ Q  # coefficient for y in w dynamics
DQ_term = Q @ D_new  # for correcting x_m = y - Du

def task4_dynamics(t, state):
    x = state[:n4]  # true state
    w = state[n4:]  # auxiliary variable w = x_u + Q x_m

    # Reconstruct x_hat
    x_new = np.linalg.inv(T_perm) @ x

    # Control: u = K x_hat
    # First get y
    # We need to avoid algebraic loop: u depends on x_hat, x_hat depends on y, y depends on u
    # Resolve: y = C x + D u, u = K x_hat
    # x_hat depends on w and y, y depends on u
    # For D != 0: need to solve algebraic loop

    # x_m_hat = y - D u (in transformed coords)
    # x_u_hat = w - Q x_m_hat = w - Q(y - Du)
    # x_new_hat = [x_m_hat; x_u_hat]
    # x_hat = T_perm @ x_new_hat
    # u = K x_hat = K T_perm [y - Du; w - Q(y-Du)]
    # y = Cx + Du

    # Let v = y - Du = Cx (the "clean" output)
    # Then x_m_hat = v = Cx
    # x_u_hat = w - Q v
    # x_new_hat = [v; w - Qv]
    # x_hat = T_perm @ [v; w - Qv]
    # u = K @ T_perm @ [v; w - Qv]
    # But v = Cx, so:
    # u = K @ T_perm @ [Cx; w - Q Cx]
    # This is linear in x and w, no algebraic loop!

    v = C4 @ x  # v = Cx (p-vector)
    x_u_hat = w - Q @ v
    x_new_hat = np.concatenate([v, x_u_hat])
    x_hat = T_perm @ x_new_hat
    u_val = (K4 @ x_hat).item()

    # True system dynamics
    dx = A4 @ x + B4.flatten() * u_val

    # w dynamics
    y = C4 @ x + D4.flatten() * u_val
    x_m = y - D_new.flatten() * u_val  # = Cx = v
    dw = Gamma @ w + Gamma_Q_term @ x_m + BQ.flatten() * u_val

    return np.concatenate([dx, dw])

sol4 = solve_ivp(task4_dynamics, [0, t_sim4[-1]],
                 np.concatenate([x0_4, zhat0_4]),
                 t_eval=t_sim4, method='RK45', rtol=1e-10, atol=1e-12)

x4_t = sol4.y[:n4]
w4_t = sol4.y[n4:]

# Reconstruct x_hat for each time step
xhat4_t = np.zeros_like(x4_t)
u4_t = np.zeros(x4_t.shape[1])
zhat4_t = np.zeros((q4, x4_t.shape[1]))

for i in range(x4_t.shape[1]):
    x_i = x4_t[:, i]
    w_i = w4_t[:, i]
    v_i = C4 @ x_i
    x_u_hat = w_i - Q @ v_i
    x_new_hat = np.concatenate([v_i, x_u_hat])
    xhat4_t[:, i] = T_perm @ x_new_hat
    u4_t[i] = (K4 @ xhat4_t[:, i]).item()
    zhat4_t[:, i] = x_u_hat

e4_t = x4_t - xhat4_t

t_max4 = find_visual_end(sol4.t, np.vstack([x4_t, e4_t]))

# u(t)
fig_u, ax_u = plt.subplots(figsize=(8, 4))
ax_u.plot(sol4.t, u4_t)
ax_u.set_title('Задание 4: Управление $u(t)$')
ax_u.set_xlabel('t')
ax_u.set_xlim(0, t_max4)
fig_u.tight_layout()
fig_u.savefig(os.path.join(img_dir, 'task4_u.png'), bbox_inches='tight')
plt.close(fig_u)

# z_hat(t)
fig_z, ax_z = plt.subplots(figsize=(8, 4))
for j in range(q4):
    ax_z.plot(sol4.t, zhat4_t[j], label=f'$\\hat{{z}}_{j+1}$')
ax_z.set_title('Задание 4: Состояние наблюдателя $\\hat{z}(t)$')
ax_z.set_xlabel('t')
ax_z.set_xlim(0, t_max4)
ax_z.legend()
fig_z.tight_layout()
fig_z.savefig(os.path.join(img_dir, 'task4_zhat.png'), bbox_inches='tight')
plt.close(fig_z)

# x(t) vs x_hat(t)
fig_x, ax_x = plt.subplots(figsize=(8, 4))
for j in range(n4):
    ax_x.plot(sol4.t, x4_t[j], label=f'$x_{j+1}$')
    ax_x.plot(sol4.t, xhat4_t[j], '--', label=f'$\\hat{{x}}_{j+1}$')
ax_x.set_title('Задание 4: $x(t)$ и $\\hat{x}(t)$')
ax_x.set_xlabel('t')
ax_x.set_xlim(0, t_max4)
ax_x.legend(loc='best', fontsize=7)
fig_x.tight_layout()
fig_x.savefig(os.path.join(img_dir, 'task4_x.png'), bbox_inches='tight')
plt.close(fig_x)

# e(t)
fig_e, ax_e = plt.subplots(figsize=(8, 4))
for j in range(n4):
    ax_e.plot(sol4.t, e4_t[j], label=f'$e_{j+1}$')
ax_e.set_title('Задание 4: Ошибка $e(t) = x(t) - \\hat{x}(t)$')
ax_e.set_xlabel('t')
ax_e.set_xlim(0, t_max4)
ax_e.legend()
fig_e.tight_layout()
fig_e.savefig(os.path.join(img_dir, 'task4_e.png'), bbox_inches='tight')
plt.close(fig_e)

# ||e(t)||
fig_en, ax_en = plt.subplots(figsize=(8, 4))
e4_norm = np.linalg.norm(e4_t, axis=0)
ax_en.plot(sol4.t, e4_norm)
ax_en.set_title('Задание 4: $||e(t)||$')
ax_en.set_xlabel('t')
ax_en.set_yscale('log')
ax_en.set_xlim(0, t_max4)
fig_en.tight_layout()
fig_en.savefig(os.path.join(img_dir, 'task4_enorm.png'), bbox_inches='tight')
plt.close(fig_en)

# y(t)
fig_y, ax_y = plt.subplots(figsize=(8, 4))
y4_t = np.array([(C4 @ x4_t[:, i] + D4.flatten() * u4_t[i]) for i in range(x4_t.shape[1])]).T
for j in range(p4):
    ax_y.plot(sol4.t, y4_t[j], label=f'$y_{j+1}$')
ax_y.set_title('Задание 4: Выход $y(t)$')
ax_y.set_xlabel('t')
ax_y.set_xlim(0, t_max4)
ax_y.legend()
fig_y.tight_layout()
fig_y.savefig(os.path.join(img_dir, 'task4_y.png'), bbox_inches='tight')
plt.close(fig_y)

print("\nЗадание 4: графики сохранены")

# ============================================================
#   СОХРАНЕНИЕ ПЕРЕМЕННЫХ
# ============================================================
import re
with open(vars_file, 'w', encoding='utf-8') as f:
    for name, val in sorted(variables.items()):
        val_str = str(val).strip()
        if val_str.startswith('$'):
            f.write(f'#let {name} = {val_str}\n')
        elif re.search(r'[a-hj-zA-HJ-Zа-яА-Я]', val_str): # text, avoid matching 'i' for imaginary
            f.write(f'#let {name} = "{val_str}"\n')
        else:
            f.write(f'#let {name} = ${val_str}$\n')

print(f"\nВсе переменные сохранены в {vars_file}")
print(f"Графики сохранены в {img_dir}")
print("ГОТОВО!")
