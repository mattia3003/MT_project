"""E0b stages: data audit and mask quality control.

One module per topic, each holding that topic's review stage (evidence and a proposed config
value) and production stage (applies the decision), with the figures and tables they need:

- `common`: helpers shared by every stage (loading, decisions, overwrite guard, slice access).
- `brain`: `brain-review` (decision A) and `brain`.

The command line is `scripts/e0b_audit.py`; every stage function here takes the config and
keyword options with the same names as its command-line options, so it can also be called
from a notebook.
"""
