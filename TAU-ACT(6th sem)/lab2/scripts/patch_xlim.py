import sys, os
import re

path = r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab2\scripts\solve_lab2.py"
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Inject the helper function right after imports
helper_def = """
def find_visual_end(t, signals, tol=1e-3):
    sig = np.atleast_2d(signals)
    env = np.max(np.abs(sig), axis=0)
    above = np.where(env > tol)[0]
    if len(above) == 0: return t[0] + 0.5
    idx = above[-1]
    tend = t[idx] * 1.15
    return min(t[-1], max(tend, t[0]+0.1))

"""
if helper_def not in text:
    text = text.replace("plt.rcParams.update({", helper_def + "plt.rcParams.update({")

# We apply it to Task 1:
# We need to collect u_t to find global max t for ax_u.
if "u_ts = []" not in text:
    text = text.replace(
        "fig_u, ax_u = plt.subplots(1, 1, figsize=(10, 5))",
        "fig_u, ax_u = plt.subplots(1, 1, figsize=(10, 5))\nu_ts = []"
    )
    text = text.replace(
        "    ax_u.plot(sol.t, u_t, label=f\"{sp['name']}: {sp['label']}\")",
        "    ax_u.plot(sol.t, u_t, label=f\"{sp['name']}: {sp['label']}\")\n    u_ts.append(u_t)"
    )
    # Apply to ax_u
    text = text.replace(
        "ax_u.legend(loc='best')\nfig_u.tight_layout()",
        "ax_u.legend(loc='best')\nif u_ts:\n    t_max_u = find_visual_end(sol.t, u_ts)\n    ax_u.set_xlim(0, t_max_u)\nfig_u.tight_layout()"
    )
    
    # Task 1 single figures
    text = text.replace(
        "    ax_single[-1].set_xlabel('t')",
        "    ax_single[-1].set_xlabel('t')\n    t_max_x = find_visual_end(sol.t, x_t)\n    ax_single[-1].set_xlim(0, t_max_x)"
    )

# Task 2
# e_t is a good indicator of settling.
if "t_max2 =" not in text:
    text = text.replace(
        "    fig.suptitle(f'Наблюдатель, спектр {sp[\"name\"]}')",
        "    t_max2 = find_visual_end(sol.t, e_t, tol=1e-3)\n    for ax in axes:\n        ax.set_xlim(0, t_max2)\n    fig.suptitle(f'Наблюдатель, спектр {sp[\"name\"]}')"
    )

# Task 3
if "t_max3 =" not in text:
    text = text.replace(
        "fig3.suptitle('Задание 3: Модальное управление по выходу')",
        "t_max3 = find_visual_end(sol3.t, np.vstack([x3_t, e3_t]))\nfor ax in axes3.flatten():\n    ax.set_xlim(0, t_max3)\nfig3.suptitle('Задание 3: Модальное управление по выходу')"
    )

# Task 4
if "t_max4 =" not in text:
    text = text.replace(
        "fig4.suptitle('Задание 4: Наблюдатель пониженного порядка')",
        "t_max4 = find_visual_end(sol4.t, np.vstack([x4_t, e4_t]))\nfor ax in axes4.flatten():\n    ax.set_xlim(0, t_max4)\nfig4.suptitle('Задание 4: Наблюдатель пониженного порядка')"
    )


with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Xlim patch applied to solve_lab2.py successfully!")
