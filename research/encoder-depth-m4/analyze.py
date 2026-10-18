"""Phase 4: turn raw probe / latency outputs into results.json numbers: seed means, bootstrap CIs, paired layer contrasts.

Bootstrap resamples *test utterances* (paired across conditions: the same resample is applied to every layer). For Common Voice,
whose test speakers are many, a speaker-cluster bootstrap is added; ASCEND test has only a couple of speakers, so a cluster
interval would be meaningless and is not reported (the limitation is recorded instead)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = ["L3", "L6", "L9", "L12", "L12-Proj"]
SUBSETS = {"CV": ("CV", None), "FLEURS": ("FLEURS", None), "ASCEND": ("ASCEND", None), "ASCEND-mixed": ("ASCEND", "mixed")}
BOOT = 2000


def masks(rows):
    out = {}
    for key, (name, lang) in SUBSETS.items():
        out[key] = np.array([r["set"] == name and (lang is None or r.get("language") == lang) for r in rows])
    return out


def ratio(e, n, idx):
    return e[idx].sum(-1) / n[idx].sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probes", type=Path, default=Path("research/encoder-depth-m4/runs/probes"))
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/analysis.json"))
    a = p.parse_args()
    res = json.loads((a.probes / "results.json").read_text(encoding="utf-8"))
    arrays = np.load(a.probes / "test_errors.npz")
    rows = json.loads((a.probes / "test_rows.json").read_text(encoding="utf-8"))
    seeds = sorted({r["seed"] for r in res["results"]})
    mk = masks(rows)
    rng = np.random.default_rng(0)
    out = {"seeds": seeds, "bootstrap_draws": BOOT, "selected_lr": res["selected_lr"], "subsets": {}}
    for sub, m in mk.items():
        idxs = np.flatnonzero(m)
        boot = rng.integers(0, len(idxs), (BOOT, len(idxs)))
        spk = np.array([rows[i]["speaker"] or "" for i in idxs])
        clusters = np.unique(spk)
        cluster_ok = sub == "CV" and len(clusters) >= 10
        cboot = None
        if cluster_ok:
            members = [np.flatnonzero(spk == c) for c in clusters]
            pick = rng.integers(0, len(clusters), (BOOT, len(clusters)))
            cboot = [np.concatenate([members[j] for j in row]) for row in pick]
        data = {}
        for variant in ("base", "adapted"):
            for layer in LAYERS:
                per_seed = np.stack([arrays[f"{variant}|{layer}|{s}"][idxs] for s in seeds])  # [S, N, 2]
                units = per_seed[0, :, 1].astype(float)
                if (per_seed[:, :, 1] != per_seed[0, :, 1]).any():
                    raise RuntimeError("reference units differ across seeds")
                data[variant, layer] = (per_seed[:, :, 0].mean(0), units, per_seed[:, :, 0].sum(1) / units.sum())
        table = {}
        for (variant, layer), (e, n, seed_vals) in data.items():
            d = ratio(e, n, boot)
            table[f"{variant}/{layer}"] = {
                "error": float(e.sum() / n.sum()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "seed_values": seed_vals.tolist(),
                "seed_std": float(seed_vals.std(ddof=1)) if len(seeds) > 1 else None,
                "utts": int(len(idxs)),
                "ref_units": int(n.sum()),
            }
        contrasts = {}

        def contrast(name, A, B):
            ea, na, _ = data[A]
            eb, nb, _ = data[B]
            diff = ratio(ea, na, boot) - ratio(eb, nb, boot)
            r = {
                "abs_diff": float(ea.sum() / na.sum() - eb.sum() / nb.sum()),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            }
            r["rel_diff"] = r["abs_diff"] / float(eb.sum() / nb.sum())
            if cboot is not None:
                dc = np.array([ea[i].sum() / na[i].sum() - eb[i].sum() / nb[i].sum() for i in cboot])
                r["ci95_speaker_cluster"] = [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]
            contrasts[name] = r

        for variant in ("base", "adapted"):
            for layer in ("L3", "L6", "L9"):
                contrast(f"{variant}: {layer} - L12", (variant, layer), (variant, "L12"))
            contrast(f"{variant}: L12-Proj - L12", (variant, "L12-Proj"), (variant, "L12"))
        for layer in LAYERS:
            contrast(f"LoRA effect (adapted - base) at {layer}", ("adapted", layer), ("base", layer))
        out["subsets"][sub] = {
            "table": table,
            "contrasts": contrasts,
            "speaker_cluster_bootstrap": cluster_ok,
            "test_speakers": int(len(clusters)),
        }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for sub, v in out["subsets"].items():
        print(sub, "speakers", v["test_speakers"])
        for variant in ("base", "adapted"):
            print("  ", variant, " ".join(f"{layer}={v['table'][f'{variant}/{layer}']['error']:.3f}" for layer in LAYERS))


if __name__ == "__main__":
    main()
