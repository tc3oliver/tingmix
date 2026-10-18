"""Phase 5: assemble the tracked results.json from the gitignored runs/ outputs. No number is typed by hand except the dataset
provenance below (revisions / licences read from the Hugging Face dataset cards when the data was downloaded)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATASETS = {
    "FLEURS": {"hf": "google/fleurs", "config": "en_us", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd", "license": "cc-by-4.0"},
    "ASCEND": {
        "hf": "CAiRE/ASCEND",
        "revision": "737e9800ae31be9932ba8464c80366559bd28424",
        "license": "cc-by-sa-4.0",
        "note": "Simplified transcripts converted to Traditional with OpenCC s2t",
    },
    "CV": {
        "hf": "fsicoli/common_voice_22_0",
        "config": "zh-TW",
        "revision": "ae911de250fe9375d08e2daa2105b671d659d446",
        "license": "cc0-1.0",
        "note": "third-party mirror of Mozilla Common Voice 22.0; not byte-compared with the official release",
    },
}
LAYERS = ["L3", "L6", "L9", "L12", "L12-Proj"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def convergence(res):
    """Mean (over seeds) pooled dev CTC at epochs 5/10/15/20 per condition, and the last-5-epoch change."""
    out = {}
    for variant in ("base", "adapted"):
        for layer in LAYERS:
            hs = [r["history"] for r in res["results"] if r["variant"] == variant and r["layer"] == layer]
            dev = [statistics.mean(h[i]["dev_ctc"] for h in hs) for i in range(len(hs[0]))]
            train = [statistics.mean(h[i]["train_ctc"] for h in hs) for i in range(len(hs[0]))]
            out[f"{variant}/{layer}"] = {
                "epochs": [h["epoch"] for h in hs[0]],
                "dev_ctc": dev,
                "train_ctc": train,
                "dev_change_epoch15_to_20": dev[-1] - dev[-2],
                "dev_change_epoch10_to_20": dev[-1] - dev[-3],
                "head_parameters": hs and next(r["head_parameters"] for r in res["results"] if r["layer"] == layer),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("research/encoder-depth-m4/runs"))
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/results.json"))
    a = p.parse_args()
    parity, latency = load(a.runs / "parity.json"), load(a.runs / "latency.json")
    extract, manifests = load(a.runs / "features/extract_results.json"), load(a.runs / "manifests/report.json")
    probes, analysis = load(a.runs / "probes/results.json"), load(a.runs / "analysis.json")
    lat = {
        k: {x: v[x] for x in ("p50_ms", "p95_ms", "mean_ms", "std_ms", "n", "rtf_p50", "executed_parameters") if x in v}
        | {x: v[x] for x in v if x.startswith("p50_reduction")}
        for k, v in latency["measures"].items()
    }
    est = {}
    for secs in latency["durations_s"]:
        e12, full = lat[f"E12/{secs}s"]["p50_ms"], lat[f"e2e_full/{secs}s"]["p50_ms"]
        est[f"{secs}s"] = {
            "reader_share_of_e2e_p50": lat[f"reader_greedy/{secs}s"]["p50_ms"] / full,
            **{f"{e}_e2e_saving_estimate": (e12 - lat[f"{e}/{secs}s"]["p50_ms"]) / full for e in ("E6", "E9")},
        }
    sub = analysis["subsets"]
    gate = {
        f"{v}/L{d}": {
            s: {
                "rel_diff_vs_L12": sub[s]["contrasts"][f"{v}: L{d} - L12"]["rel_diff"],
                "ci95_abs": sub[s]["contrasts"][f"{v}: L{d} - L12"]["ci95"],
            }
            for s in ("CV", "FLEURS", "ASCEND")
        }
        for v in ("adapted", "base")
        for d in (6, 9)
    }
    out = {
        "study": "tingmix-large-encoder-depth-m4",
        "status": "MEASURED_ON_M4",
        "environment": parity["environment"],
        "release": parity["origin"],
        "datasets": {
            n: DATASETS[n]
            | {
                "splits": {
                    s: {k: manifests["sets"][f"{n}/{s}"].get(k) for k in ("utterances", "hours", "text_overlap_removed", "speaker_overlap")}
                    for s in ("train", "dev", "test")
                }
            }
            for n in DATASETS
        },
        "sampling": {
            "seed": manifests["seed"],
            "rule": "official split kept; seeded shuffle, first N eligible; dev/test text overlap with train removed",
        },
        "mps_parity": {k: parity[k] for k in parity if k not in ("audio", "environment", "origin")} | {"status": parity["status"]},
        "feature_extraction": {
            "status": extract["status"],
            "storage_dtype": extract["storage_dtype"],
            "frames": {k: v["frames"] for k, v in extract["sets"].items()},
            "fp16_roundtrip_rel_l2_max": max(
                x for v in extract["sets"].values() for t in v["fp16_roundtrip_rel_l2"].values() for x in t.values()
            ),
            "acoustic_neighbour_audit": extract["nn_audit"],
            "exact_audio_duplicates": extract["overlap"]["exact_duplicates"],
        },
        "latency": {
            "status": latency["status"],
            "batch_size": latency["batch_size"],
            "precision": latency["precision"],
            "device": latency["device"],
            "audio": "real FLEURS en_us speech, concatenated and cut to 5/15/25 s",
            "pruned": latency["pruned"],
            "loaded_encoder_params": latency["loaded_encoder_params"],
            "measures": lat,
            "e2e_estimate": {
                "kind": "ESTIMATE, not measured: encoder saving (ms) / measured full e2e (ms), assuming an unchanged Reader and a compatible head",
                **est,
            },
            "memory_note": latency["memory_note"],
        },
        "decision": {
            "value": "INVESTIGATE",
            "basis": "E9 clears the encoder-latency screen (-23% P50) and its Chinese probes match L12, but its English (FLEURS) probe is 13-16% worse "
            "(CI excludes 0), the estimated end-to-end saving is only ~7-10% because the Reader is 52-68% of end-to-end time, and a frozen "
            "linear probe is not evidence about a retrained early-exit model. E6 fails the accuracy screen in every set. "
            "Not GO (language-specific regression; e2e saving below the screen); not NO-GO (a trained early-exit model was not tested).",
            "screen": "research screen: encoder P50 -15% and probe error <= +5% vs L12 with no material per-language regression",
            "probe_rel_diff_vs_L12": gate,
        },
        "linear_probe": {
            "scope": probes["scope"],
            "selected_lr": probes["selected_lr"],
            "lr_pilot": probes["pilot"],
            "args": probes["args"],
            "counts": probes["counts"],
            "seeds": analysis["seeds"],
            "bootstrap_draws": analysis["bootstrap_draws"],
            "known_limits": probes["known_limits"],
            "subsets": analysis["subsets"],
            "convergence": convergence(probes),
        },
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
