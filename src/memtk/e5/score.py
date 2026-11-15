"""E5 stage `score` (`docs/e5-plan.md`, step 4) and the scoring it runs: both detectors on a query
set against T, under the main settings and every sensitivity variant of decision O, with settings
and thresholds from the frozen E3 file and the E5 file. Raw tables only: no rate is computed or
printed here.

`score_query` is the one function that scores: `score` runs it once on S (the main and noisy
masks, decision L), and `rehearse` ran it on mock query sets first, so the code run on S is the code
rehearsed. `score` is the only E5 stage that reads S. It refuses to run again once it has completed
(the completion marker), and it records the SHA-256 of the `audit:` section and both frozen files,
which `analyze` checks (the lock).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, config, lesionmatch
from memtk.audit import class_table
from memtk.e4.common import gate_status
from memtk.e5.common import audit_config, audit_hash, check_prepared, drop_only_t, score_marker, variant_frozen
from memtk.io import MaskStack, read_table
from memtk.sets import SliceSet, load_pool, load_set, relabel
from memtk.stages import require
from memtk.synthetic import name_scheme


# --- stages -----------------------------------------------------------------------------------

def run_score(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Score S once: the main and noisy masks of the pool (decision L) against T, main and every
    variant (decision O). Needs the E4 gate (PASS or ACCEPTED), `prepare` and `rehearse` made with
    the current `audit:` section and frozen files. Writes `<output_root>/<pool>/e5/` (raw tables,
    `score.complete.json`); refuses once that marker exists. `--overwrite` only replaces the tables
    of a run that stopped before its marker (log such a re-run in docs/process-log.md)."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    if pool_cfg.get("trained_on") != name:
        raise SystemExit(f"Pool '{pool_name}' was trained on '{pool_cfg.get('trained_on')}', not on '{name}'.")
    data_dir = config.output_dir(cfg, pool_name) / "e5"
    marker = score_marker(cfg, pool_name)
    if marker.exists():
        raise SystemExit(f"S has already been scored ({marker}); E5 scores S once. Run 'e5_audit.py analyze'.")
    if (data_dir / "queries.csv").exists() and not overwrite:
        raise SystemExit(f"{data_dir} holds tables of a run that did not complete; pass --overwrite to score again "
                         f"and record the re-run in docs/process-log.md.")
    verdict = gate_status(cfg, name)
    if verdict not in ("PASS", "ACCEPTED"):
        raise SystemExit(f"The E4 gate is {verdict}; E5 runs only on PASS or ACCEPTED.")
    audit = audit_config(cfg)
    e3, e3_hash = calibration.load_frozen(cfg, name)
    e5, e5_hash = calibration.load_frozen(cfg, name, "e5")
    check_prepared(cfg, e5, e3_hash)
    rehearsed = config.output_dir(cfg, name) / "e5" / "rehearse" / "positive" / "rehearse.run_info.json"
    require(rehearsed, "e5_audit.py rehearse")
    if json.loads(rehearsed.read_text(encoding="utf-8")).get("e5_sha256") != e5_hash:
        raise SystemExit("'rehearse' ran with another E5 file; run 'e5_audit.py rehearse --overwrite' first.")
    hashes = {"audit_sha256": audit_hash(cfg), "frozen_sha256": e3_hash, "e5_sha256": e5_hash}

    t = load_set(cfg, name, "T")
    t_drop = drop_only_t(cfg, name, t)
    classes = [*audit["sets"]["main"], *audit["sets"]["noisy"]]
    s = load_pool(cfg, pool_name, classes)
    pool_dir = config.output_dir(cfg, pool_name)
    mask_class = read_table(pool_dir / "qc.csv").set_index("key")["mask_class"]
    raw = MaskStack(pool_dir / "masks_raw.npz")
    min_area = int(e3["min_lesion_area"])
    print(f"Drop-only label maps of {len(s.keys)} synthetic masks (from the masks as saved) ...")
    drop = relabel(np.stack([raw[k] for k in s.keys]), min_area, 0)
    meta = query_table(s, s.keys, mask_class.loc[s.keys].tolist(), [name_scheme(k) for k in s.keys])
    print(f"Scoring S: {len(s.keys)} masks ({meta['mask_class'].value_counts().to_dict()}), main and "
          f"{len(audit['sensitivity'])} variants. No rate is computed or shown here.")
    start = time.perf_counter()
    log = score_query(s, drop, meta, t, t_drop, e3, e5, list(audit["sensitivity"]), data_dir)
    log.to_csv(data_dir / "score_log.csv", index=False)
    run_extra = {"stage": "score", "inputs": ["T", "S"], "pool": pool_name, **hashes, "verdict_e4": verdict,
                 "left_out": s.left_out}
    config.write_run_info(data_dir, cfg, script=__file__, filename="score.run_info.json", extra=run_extra)
    marker.write_text(json.dumps({"completed": datetime.now().isoformat(timespec="seconds"), **hashes,
                                  "queries": len(s.keys), "by_mask_class": meta["mask_class"].value_counts().to_dict(),
                                  "variants": ["main", *audit["sensitivity"]],
                                  "minutes": round((time.perf_counter() - start) / 60, 1)}, indent=2), encoding="utf-8")
    print(f"\nScoring log:\n{log.to_string(index=False)}\nLeft out: {s.left_out}\nWrote {data_dir}\n"
          f"S is scored; '{marker.name}' locks the E5 file and audit:. Next: 'e5_audit.py analyze'.")


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
