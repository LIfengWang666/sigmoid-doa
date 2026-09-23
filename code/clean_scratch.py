# -*- coding: utf-8 -*-
"""
clean_scratch.py -- run ONCE to remove the exploratory scratch scripts from code/,
keeping only the files needed to reproduce the paper figures.

Keeps:
  code/partricnntran.py     (model definition)
  code/toolbox.py           (helpers)
  code/val_data_gen.py      (reference data generator)
  code/figures/             (the 4 figure scripts + common.py)
  code/train/               (training code)

Run:  python code/clean_scratch.py
"""
from pathlib import Path

CODE = Path(__file__).resolve().parent
KEEP = {"partricnntran.py", "toolbox.py", "val_data_gen.py", "clean_scratch.py"}
KEEP_DIRS = {"figures", "train"}

removed = 0
for p in sorted(CODE.iterdir()):
    if p.name in KEEP or p.name in KEEP_DIRS:
        continue
    if p.is_file() and p.suffix == ".py":
        p.unlink()
        print("removed", p.name)
        removed += 1
print(f"done -- removed {removed} scratch .py files; kept {sorted(KEEP)} + dirs {sorted(KEEP_DIRS)}")
