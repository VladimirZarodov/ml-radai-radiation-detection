"""Independent verification pass #1: structure + metadata of training_v4.3.h5."""
import json
import h5py
import numpy as np

H5 = "training_v4.3.h5"

def walk(group, prefix="", depth=0, maxdepth=3):
    for k in group.keys():
        item = group[k]
        path = f"{prefix}/{k}"
        if isinstance(item, h5py.Group):
            if depth < maxdepth:
                print(f"{'  '*depth}[G] {path}  attrs={dict(item.attrs)}")
                walk(item, path, depth + 1, maxdepth)
        else:
            shp = item.shape
            dt = str(item.dtype)
            preview = None
            if item.ndim == 1 and item.size and item.size <= 12:
                preview = item[:]
            elif item.ndim == 0:
                preview = item[()]
            print(f"{'  '*depth}[D] {path}  shape={shp} dtype={dt} attrs={dict(item.attrs)}"
                  + (f" val={preview}" if preview is not None else ""))

f = h5py.File(H5, "r")
print("== ROOT attrs ==")
for k, v in f.attrs.items():
    s = str(v)
    print(f"  {k}: {s[:200]}")

print("\n== root keys ==", list(f.keys()))
print("n runs:", len(f["runs"]), "first:", list(f["runs"].keys())[:5], "last:", list(f["runs"].keys())[-3:])

print("\n== structure of run3 (one full run) ==")
walk(f["runs/run3"], "run3", maxdepth=3)

print("\n== run3 attrs ==", dict(f["runs/run3"].attrs))

f.close()
