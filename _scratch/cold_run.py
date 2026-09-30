# -*- coding: utf-8 -*-
"""Official cold-run: execute the notebook from an empty cache dir and time it.

Pre-conditions (set up by the shell before launching):
  - _scratch/ contains ONLY scripts (.py/.out/.csv/.png), NO .pkl caches;
  - previous _scratch was moved aside to _scratch_coldbak.
Writes _scratch/cold_time.txt with total wall-clock minutes.
"""
import io, os, subprocess, sys, time, glob

pkls = glob.glob('_scratch/*.pkl')
assert not pkls, f'cold run must start without caches, found: {pkls}'
t0 = time.time()
r = subprocess.run([sys.executable, '_scratch/run_exec.py'])
mins = (time.time() - t0) / 60.0
with io.open('_scratch/cold_time.txt', 'w', encoding='utf-8') as o:
    o.write(f'{mins:.1f}\n')
print(f'COLD RUN done in {mins:.1f} min, wrapper rc={r.returncode}')
