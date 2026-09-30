# -*- coding: utf-8 -*-
"""Bit-compare the notebook-rebuilt item3 caches against the originals."""
import pickle, numpy as np, pandas as pd
new = pickle.load(open('_scratch/item3_ml_proba.pkl', 'rb'))
old = pickle.load(open('_scratch/item3_ml_proba_truth.pkl', 'rb'))
same_keys = set(new) == set(old)
diffs = []
for r in sorted(old):
    if r not in new:
        diffs.append((r, 'missing')); continue
    if new[r]['nw'] != old[r]['nw']:
        diffs.append((r, 'nw')); continue
    for nm in ('logreg', 'mlp'):
        a, b = new[r][nm], old[r][nm]
        if len(a) != len(b) or not np.array_equal(a, b):
            md = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if len(a) == len(b) else float('nan')
            diffs.append((r, nm, md))
print('keys same:', same_keys, '| runs compared:', len(old), '| diffs:', diffs[:10] or 'NONE (bit-identical)')
snew = pickle.load(open('_scratch/item3_hybrid.pkl', 'rb'))
sold = pickle.load(open('_scratch/item3_hybrid_truth.pkl', 'rb'))
cols = ['hybrid', 'target', 'thr_p', 'thr_f']
m = sold[cols].merge(snew[cols], on=['hybrid', 'target'], suffixes=('_old', '_new'))
bad = m[(m.thr_p_old != m.thr_p_new) | (m.thr_f_old != m.thr_f_new)]
print('SEL rows old/new:', len(sold), len(snew), '| mismatched pairs:', len(bad))
if len(bad):
    print(bad.to_string())
