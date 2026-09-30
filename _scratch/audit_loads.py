# -*- coding: utf-8 -*-
import re
import nbformat
nb = nbformat.read('Matched_Filter_Detection.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for m in re.finditer(r'pickle\.load\(open\("([^"]+)"', c.source):
        pre = c.source[:m.start()]
        g = 'GUARD' if 'os.path.exists' in pre else 'none '
        print(i, g, m.group(1))
