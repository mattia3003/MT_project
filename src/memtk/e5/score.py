"""E5 scoring (`docs/e5-plan.md`): both detectors on a query set against T, under the main settings
and every sensitivity variant of decision O, with settings and thresholds from the frozen E3 file
and the E5 file. Raw tables only: no rate is computed here.

`score_query` is the one function that scores: the `score` stage (step 4) runs it once on S, and
`rehearse` runs it on mock query sets first, so the code run on S is the code rehearsed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, lesionmatch
from memtk.audit import class_table
from memtk.e5.common import variant_frozen
from memtk.sets import SliceSet


# --- scoring ----------------------------------------------------------------------------------

def query_table(q: SliceSet, clusters: np.ndarray | Sequence[str], mask_class: Sequence[str],
                scheme: Sequence[str]) -> pd.DataFrame:
    """One row per query slice: `key`, `cluster` (the bootstrap unit: the sample itself for S,
    the patient for real slices), `mask_class`, `scheme` (naming scheme or source set),
    `n_lesions` (lesions >= the minimum area) and `l1_area` (area of the largest lesion, strays
    included: the size used for near-exact copies and the size-confound check)."""
    flat = q.labels.reshape(len(q.labels), -1)
    return pd.DataFrame({"key": q.keys, "cluster": list(clusters), "mask_class": list(mask_class),
                         "scheme": list(scheme), "n_lesions": flat.max(axis=1).astype(int),
                         "l1_area": (flat == 1).sum(axis=1).astype(int)})


def score_query(q: SliceSet, drop_labels: np.ndarray, meta: pd.DataFrame, t: SliceSet, t_drop: SliceSet,
                e3: dict, e5: dict, variants: Sequence[str], out_dir: Path) -> pd.DataFrame:
    """Score query set `q` against T (main and each variant) and write the raw tables to `out_dir`:
    `queries.csv` (`meta`), `classes_<variant>.csv` for main and every variant (columns of
    `audit.class_table`), `e1_main.csv`, `e1_drop_only.csv`, and the four E2 tables of every run
    that scores E2 (`e2_<variant>_*`). Percentile variants re-threshold the main tables; sigma and
    bound variants run E2 again; drop-only runs both on `drop_labels` against `t_drop`. Returns
    one row per run: variant, seconds, scored slices, lesion pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "queries.csv", index=False)
    log = []

    start = time.perf_counter()
    e1m, e2m, e2cm = calibration.run_frozen(q.labels, q.keys, t, e3)
    e1m.to_csv(out_dir / "e1_main.csv", index=False)
    lesionmatch.save_tables(e2m, out_dir, "main")
    class_table(q, e1m, e2m, e2cm).to_csv(out_dir / "classes_main.csv", index=False)
    log.append(run_row("main", start, e1m, e2m))

    for name in variants:
        start = time.perf_counter()
        variant = e5["variants"][name]
        vf = variant_frozen(e3, variant)
        if variant["kind"] == "percentile":
            e1c = calibration.e1_classes(e1m, e1m["n_lesions"].to_numpy(), vf)
            table = class_table(q, e1c, e2m, calibration.e2_classes(e2m, vf))
            e2 = None
        elif variant["kind"] == "drop_only":
            qd = SliceSet(q.name, q.keys, q.patients, drop_labels, dict(q.left_out))
            e1c, e2, e2c = calibration.run_frozen(drop_labels, q.keys, t_drop, vf)
            e1c.to_csv(out_dir / f"e1_{name}.csv", index=False)
            table = class_table(qd, e1c, e2, e2c)
        else:
            _, settings = calibration.frozen_settings(vf)
            e2 = lesionmatch.match(q.labels, q.keys, t.labels, t.keys, **settings)
            e1c = e1m
            table = class_table(q, e1m, e2, calibration.e2_classes(e2, vf))
        if e2 is not None:
            lesionmatch.save_tables(e2, out_dir, name)
        table.to_csv(out_dir / f"classes_{name}.csv", index=False)
        log.append(run_row(name, start, e1c, e2))
    return pd.DataFrame(log)


def run_row(name: str, start: float, e1: pd.DataFrame, e2: lesionmatch.LesionMatch | None) -> dict:
    """One line of the scoring log."""
    return {"variant": name, "seconds": round(time.perf_counter() - start, 1), "e1_scored": len(e1),
            "e2_scored": int((e2.queries["n_informative"] > 0).sum()) if e2 is not None else np.nan,
            "lesion_pairs": len(e2.lesion_pairs) if e2 is not None else np.nan}
