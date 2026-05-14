import sys, re
path = r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab2\scripts\solve_lab2.py"
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

orig = """with open(vars_file, 'w', encoding='utf-8') as f:
    for name, val in sorted(variables.items()):
        f.write(f'#let {name} = {val}\\n')"""

new = """import re
with open(vars_file, 'w', encoding='utf-8') as f:
    for name, val in sorted(variables.items()):
        val_str = str(val).strip()
        if val_str.startswith('$'):
            f.write(f'#let {name} = {val_str}\\n')
        elif re.search(r'[a-zA-HJ-Zа-яА-Я]', val_str): # text, avoid matching 'i' for imaginary
            f.write(f'#let {name} = "{val_str}"\\n')
        else:
            f.write(f'#let {name} = ${val_str}$\\n')"""

if orig in text:
    text = text.replace(orig, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Patched variables.typ dumping successfully.")
else:
    print("Could not find the target block.")
