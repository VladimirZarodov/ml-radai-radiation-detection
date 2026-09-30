# -*- coding: utf-8 -*-
"""Compare a fresh nb_outputs3.txt dump against nb_out_truth.txt.

Blocks that print nothing (the 7 new plot cells, `%matplotlib` etc.) are dropped
on both sides, so the sequences should be 1:1. Tolerated diffs:
  - extra R=100 line in the §3.6 table block;
  - fallback messages ('пересборка', 'запускаю обучение', 'обучение завершено');
  - cold-run cache 'built ...s' timing lines.
"""
import io, re, sys

def load(p):
    txt = io.open(p, encoding='utf-8').read()
    blocks, cur = [], None
    for ln in txt.splitlines():
        if re.match(r'^=+ cell \d+$', ln):
            cur = []
            blocks.append(cur)
        elif cur is not None:
            ln = ln.rstrip()
            if ln and ln != '(no output)':
                cur.append(ln)
    return [b for b in blocks if b and not all(re.match(r'^<Figure size .*>', l) for l in b)]

tol = re.compile(r'(пересборка|запускаю обучение|обучение завершено|proba\+SEL из кэша|built \d|scores built|свежие логи|^\s*кэш |^\s*\d+/123 \(\d+s\)|mf cache: \d+ runs|baseline windows: cache)')

a = load('_scratch/nb_out_truth.txt')
b = load(sys.argv[1] if len(sys.argv) > 1 else '_scratch/nb_outputs3.txt')
print(f'non-empty blocks: truth {len(a)}, new {len(b)}')
nd = 0
for k in range(max(len(a), len(b))):
    if k >= len(a):
        print(f'NEW extra block {k}: {b[k][:2]}'); nd += 1; continue
    if k >= len(b):
        print(f'MISSING block {k}: {a[k][:2]}'); nd += 1; continue
    x, y = a[k], b[k]
    if x == y:
        continue
    xt = [l for l in x if not tol.search(l)]
    yt = [l for l in y if not tol.search(l)]
    if xt == yt:
        continue  # only tolerated lines differ
    if len(yt) == len(xt) + 1 and (yt[1:] == xt or yt[:-1] == xt or all(p == q for p, q in zip(xt, yt[:2]))):
        extra = [l for l in yt if l not in xt]
        print(f'block {k}: +1 line (expected R=100): {extra[:1]}')
        continue
    nd += 1
    print(f'DIFF block {k} (truth {len(x)} / new {len(y)} lines):')
    for m in range(max(len(x), len(y))):
        lx = x[m] if m < len(x) else '<none>'
        ly = y[m] if m < len(y) else '<none>'
        if lx != ly:
            print('  A:', lx[:130]); print('  B:', ly[:130]); break
print('hard diffs:', nd)
