# -*- coding: utf-8 -*-
"""Execute the notebook, then dump every cell's stdout text to nb_outputs3.txt."""
import subprocess, sys, io
import nbformat

r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                    "--execute", "--inplace", "Matched_Filter_Detection.ipynb"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("rc:", r.returncode)
tail = (r.stderr or r.stdout or "").strip().splitlines()
for l in tail[-6:]:
    print("  |", l[:160])
nb = nbformat.read("Matched_Filter_Detection.ipynb", as_version=4)
with io.open("_scratch/nb_outputs3.txt", "w", encoding="utf-8") as o:
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        o.write(f"{'='*18} cell {i}\n")
        for out in c.get("outputs", []):
            if out.get("output_type") == "stream":
                o.write(out.get("text", ""))
            elif out.get("output_type") == "error":
                o.write(f"!!! ERROR {out['ename']}: {out['evalue']}\n")
                o.write("\n".join(out.get("traceback", [])[-4:]) + "\n")
            elif "text/plain" in out.get("data", {}):
                o.write(out["data"]["text/plain"] + "\n")
        if not c.get("outputs"):
            o.write("(no output)\n")
print("dumped _scratch/nb_outputs3.txt; cells:", len(nb.cells))
