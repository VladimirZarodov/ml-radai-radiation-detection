# -*- coding: utf-8 -*-
"""Compile-check every code cell of the built notebook (IPython magics skipped)."""
import nbformat
nb = nbformat.read("Matched_Filter_Detection.ipynb", as_version=4)
bad = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    src = "\n".join(l for l in c.source.split("\n") if not l.strip().startswith("%"))
    try:
        compile(src, "cell%d" % i, "exec")
    except SyntaxError as e:
        bad += 1
        print("SYNTAX ERR cell", i, ":", e)
print("cells:", len(nb.cells), "bad:", bad)
