import nbformat
nb = nbformat.read("Matched_Filter_Detection.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    outs = c.get("outputs", [])
    kinds = ",".join(o.get("output_type", "?") + (":" + o.get("evalue", "")[:90] if o.get("output_type") == "error" else "") for o in outs)
    src1 = (c.source.splitlines() or [""])[0][:58]
    print(f"cell {i:2d} | {src1:<58s} | {kinds if kinds else 'NO OUTPUT'}")
