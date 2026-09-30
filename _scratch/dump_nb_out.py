import nbformat
nb = nbformat.read("Matched_Filter_Detection.ipynb", as_version=4)
want = {9, 13, 14, 15, 19, 22, 3, 7}
for i, c in enumerate(nb.cells):
    if i not in want:
        continue
    print(f"\n========== CELL {i} ==========")
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(o["text"][:2500])
        elif o.get("output_type") == "execute_result":
            t = o["data"].get("text/plain", "")
            print(t[:2500])
        elif o.get("output_type") == "error":
            print("ERROR", o["evalue"])
