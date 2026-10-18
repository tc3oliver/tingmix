# TingMix-Large M4 Encoder Depth Research Report

**Research decision: INVESTIGATE.** Measured on the Mac Studio M4 Max (MPS, fp32, no CPU fallback). This is a research decision, not a
product decision. GO would only have meant "worth investing in next-stage early-exit model training"; it would never have meant that the
existing Reader can be deployed on a 9-layer encoder.

Every number below is in [results.json](results.json) (assembled by `collect.py` from the gitignored `runs/` outputs). Three kinds of
statements are kept apart throughout: **measured** (latency, probe error), **estimated** (end-to-end saving), and **not tested** (anything
about a retrained early-exit model).

## Environment and sources

- Mac Studio M4 Max 64 GB, macOS 27.0, Python 3.12.14, torch 2.14.1, transformers 5.19.0, 12 threads, `PYTORCH_ENABLE_MPS_FALLBACK=0`,
  fp32, commit `e11ab939` as the starting point.
- Released model: `tc3oliver/TingMix-Large@75071a1260fb21874f150df6ebd53716ac423f13` (LoRA r16/alpha32 on all 12 layers). SHA256: encoder delta
  `1b155cfe…d9cb`, Reader `67a06c21…e7b`, tokenizer `dbc1aba9…7756`. Encoder `google/embeddinggemma-2@914f7f89…`.
  Params: base tower 304,824,608; with LoRA 310,329,632 (LoRA 5,505,024); Reader 22,171,041. No published weight was modified.
- Phase 0 gate **PASS**: MPS vs CPU-fp32 relative L2 per stage 2.6e-7 (L3) to 2.0e-6 (L12-Proj), logits max abs diff 5.7e-5, identical greedy transcript,
  no NaN/Inf, MPS CTC forward+backward works. Caveat: parity used one 6.2 s synthetic (TTS) Chinese clip and fp32 only; bf16/fp16 were not tested.
- Data (public only; no audio, feature or manifest is committed; see README for provenance): FLEURS en_us, ASCEND, Common Voice zh-TW.
  Official splits kept, seeded subsample. train 6,000 utterances (CV 2,000 / FLEURS 1,000 / ASCEND 3,000; ~7.6 h), dev 700, test 1,500.
  Dev/test text that also occurs in train was removed; there is no speaker overlap and no identical audio between splits, and the
  acoustic nearest-neighbour audit found 0 test/dev utterances with cosine > 0.98 to any train utterance (max 0.90).

## Linear CTC probe (measured)

Frozen features → parameter-free LayerNorm → one Linear layer → CTC (blank 0, 5,025-unit tokenizer). 20 epochs, batch 16, AdamW wd 0.01, 5 % warmup +
cosine, 3 seeds (1337/1338/1339), identical data order and targets for every condition. One learning rate for all conditions
(3e-3, chosen once on pooled dev CTC from {1e-3, 3e-3, 1e-2} using 6-epoch runs); test was scored once. Metrics: CER (CV), WER (FLEURS),
MER (ASCEND), Embed2ASR normalisation. Values are seed means; brackets are 95 % utterance-level bootstrap CIs (2000 draws, the same
resample for every condition). L12-Proj has a 1536→5025 head (7.72 M params vs 5.15 M), so it is a different budget and is not a depth point.

**ASR-adapted encoder (LoRA)**

| | L3 | L6 | L9 | L12 | L12-Proj |
|---|---|---|---|---|---|
| CV CER | .772 [.758,.788] | .566 [.545,.586] | .452 [.432,.472] | .460 [.438,.485] | .490 [.470,.511] |
| FLEURS WER | .901 [.894,.908] | .780 [.769,.790] | .680 [.668,.692] | .587 [.573,.602] | .600 [.586,.614] |
| ASCEND MER (all) | .684 [.670,.699] | .508 [.491,.525] | .448 [.430,.467] | .465 [.444,.485] | .461 [.442,.479] |
| ASCEND code-switch subset | .659 [.639,.681] | .495 [.473,.518] | .441 [.418,.465] | .438 [.409,.466] | .439 [.415,.463] |

**Base (un-adapted) encoder**

| | L3 | L6 | L9 | L12 | L12-Proj |
|---|---|---|---|---|---|
| CV CER | .772 [.757,.787] | .569 [.548,.590] | .468 [.447,.488] | .494 [.469,.521] | .496 [.476,.516] |
| FLEURS WER | .902 [.895,.909] | .782 [.771,.792] | .684 [.673,.696] | .606 [.593,.620] | .622 [.608,.636] |
| ASCEND MER (all) | .684 [.670,.699] | .507 [.490,.524] | .452 [.433,.471] | .477 [.457,.497] | .468 [.450,.487] |
| ASCEND code-switch subset | .659 [.638,.681] | .495 [.471,.519] | .446 [.422,.471] | .452 [.425,.479] | .447 [.424,.472] |

Seed standard deviations are 0.000–0.006 (0.011 for base/L12 on CV), far below the layer gaps discussed below.

**Paired contrasts to L12, adapted encoder** (abs difference, relative to L12, 95 % CI of the abs difference):

| | L6 − L12 | L9 − L12 |
|---|---|---|
| CV | +.105 (+22.9 %) [+.088,+.122] | −.008 (−1.8 %) [−.025,+.007] |
| FLEURS | +.192 (+32.8 %) [+.180,+.203] | +.093 (+15.8 %) [+.082,+.103] |
| ASCEND | +.043 (+9.2 %) [+.030,+.057] | −.016 (−3.5 %) [−.028,−.004] |
| ASCEND code-switch | +.058 (+13.1 %) [+.039,+.076] | +.004 (+0.9 %) [−.014,+.020] |

The base encoder gives the same pattern (L9 − L12: CV −5.4 %, ASCEND −5.2 %, FLEURS +12.9 %). For CV the speaker-cluster bootstrap (390 test speakers)
agrees with the utterance bootstrap. ASCEND has only two test speakers, so its intervals understate speaker variability and are not claimed to
generalise; FLEURS has no speaker field.

**Convergence.** For L3, L6, L9 and L12 the pooled dev CTC is flat or slightly rising over epochs 10→20 (L9: −0.0 to +0.02; L12: ≤ ±0.01) while train CTC keeps falling
(L12 0.25 vs dev 2.65), i.e. these heads are converged and mildly over-fitted at 20 epochs. Only the L12-Proj heads were still creeping down
(−0.02 to −0.03 over the last 5 epochs). The planned 40-epoch sensitivity run was triggered only if L9 clearly lost to L12 *and* dev loss was still falling; the first
holds for English only and the second does not hold for L9/L12, so **no extra run was made** and no hyper-parameter was revisited. Caveat: only pooled dev CTC
was recorded, so a language-specific dev trend cannot be excluded.

## Latency (measured, encoder-only vs end-to-end)

MPS, fp32, batch 1, real FLEURS speech cut to 5/15/25 s, 6 warm-ups + 50 interleaved iterations, `torch.mps.synchronize` around each call, every call starts from the
input mel (no feature cache). P50 ms (P95 within 1–3 %):

| | 5 s | 15 s | 25 s |
|---|---|---|---|
| CPU preprocessing | 1.0 | 2.6 | 4.2 |
| E6 (subsample + 6 layers, 1024-d) | 16.7 | 29.1 | 40.2 |
| E9 | 23.5 | 41.6 | 57.4 |
| E12 incl. projection | 30.4 | 54.0 | 74.9 |
| Reader + greedy decode | 34.9 | 104.4 | 169.9 |
| Full model end-to-end | 66.8 | 161.8 | 251.1 |

- **Encoder only (measured):** E6 is 45–46 % and E9 23 % faster than E12 at every length; the output projection costs ≈0.2–0.5 %. Executed params: E6 154.9 M, E9 231.8 M, E12 310.3 M.
- **Reader (measured):** the Reader plus greedy decode is 52 % (5 s) to 68 % (25 s) of end-to-end time and grows faster with length than the encoder.
- **End-to-end (ESTIMATE, not measured):** with an unchanged Reader and a compatible head, the saving would be about 10 % (E9) / 21 % (E6) at 5 s and 7 % (E9) / 14 % (E6) at 25 s.
  No early-exit end-to-end system exists, so no end-to-end gain is claimed; encoder-only gains must not be quoted as end-to-end gains.
- Early exit skips compute only: all 12 layers and the LoRA stay resident, so **no model-size or memory reduction is claimed** (only torch.mps allocator counters were recorded, not a true peak).
- Reader optimisation (the largest remaining cost) is **Future Work only**; nothing was implemented.

## The seven research questions

1. **Where does ASR ability emerge?** At the layer granularity measured (L3/L6/L9/L12): L3 is poor (CER 0.77, WER 0.90), L6 recovers most of it, and the curve flattens at L9–L12. A finer location cannot be
   given. These are linear-probe error rates on 2–3 h per language, so absolute values are high for every layer.
2. **How much do L6 and L9 retain?** L6 is clearly worse than L12 in every set (+23 % CV, +33 % FLEURS, +9 % ASCEND, adapted; CIs exclude 0). L9 equals or beats L12 on Chinese-dominant sets
   (CV −2 %, CI includes 0; ASCEND −3.5 %) but is 16 % worse on English FLEURS.
3. **Is L9's gap stable?** Yes, per set: seed SD ≤ 0.006 (0.011 for base/L12 on CV), and wherever a CI excludes 0 the base and adapted encoders agree in sign. The sign depends on language (English worse; Chinese equal or better),
   so there is no single "L9 retains X %". A hypothesis (not tested) is that the final layers of an encoder trained for sequence embeddings specialise in ways that matter more for English in this setup.
4. **Effect of LoRA per layer.** None measurable at L3 and L6 (|Δ| ≤ 0.6 %, CIs include 0); small at L9 (CV −3.3 %, FLEURS −0.6 %, ASCEND n.s.); largest at L12 (−6.9 % CV, −3.1 % FLEURS, −2.5 % ASCEND, all CIs exclude 0).
   This matches the representation shift measured in parity (relative L2 vs base 0.02 at L3 growing to 0.145 at L12). The LoRA's original training data is unknown and may overlap the public sets used here,
   which could inflate the LoRA effect at L12.
5. **Best depth per language.** Chinese (CV): L9 ≈ L12 (tie). ASCEND: L9 slightly better than L12; code-switch subset: tie. English (FLEURS): L12 clearly best. L12-Proj is not better than L12 anywhere (worse on CV and FLEURS), with the head-size caveat.
6. **Real MPS latency reduction.** Encoder-only P50: −45–46 % (E6), −23 % (E9). End-to-end estimate −7…−10 % (E9) and −14…−21 % (E6); not measured.
7. **Hand a true early-exit model to MI300X?** Not on this evidence alone; see the decision.

## Decision: INVESTIGATE

Why not GO: (a) the English probe loses 13–16 % at L9, outside the ≤5 % / no-language-regression screen; (b) the estimated end-to-end saving of E9 (7–10 %) is below the 15 % screen because the Reader dominates;
(c) E6, the only option with a large saving, fails the accuracy screen in every set.
Why not NO-GO: L9 is on par with or better than L12 for Chinese under the same probe, the encoder saving is real, and a frozen linear probe on features from an encoder that was not trained to exit early
says little about a model trained for it — **these probe results are not the accuracy of a retrained early-exit model, and nothing here shows that a specially trained 6- or 9-layer model cannot do ASR.**
What would change the decision: a trained-exit study that includes English, or an end-to-end cost that includes a cheaper Reader.

## Limitations

- Frozen linear CTC on ~7.6 h of public data; absolute errors (≈0.45–0.9) say little about the released system, which was not scored here. Only relative depth comparisons are meaningful. Not an MLP, GRU Reader or fine-tune (out of scope by design).
- The original TingMix training/evaluation manifests were not available locally, so public corpora were used. The LoRA's training data may overlap them; this is not audited.
- Common Voice comes from a third-party mirror (`fsicoli/common_voice_22_0`); consistency with the official Mozilla release was not verified.
- ASCEND transcripts are Simplified Chinese converted to Traditional with OpenCC; the scoring normalisation therefore differs from the other sets and ASCEND absolute numbers should not be compared directly with CV or FLEURS. Dev has ~3 and test ~2 speakers.
- One learning rate for all conditions (picked once on pooled dev); a layer-specific rate was not tried by design. Pooled dev CTC only was logged. No 40-epoch run (reason above).
- L12-Proj uses a larger head and its dev loss was still decreasing slightly; treat its comparison to L12 as indicative.
- Latency: one machine, fp32, batch 1, FLEURS-derived audio, thermal state recorded but no sustained-load test; the end-to-end saving is a calculation from separate measurements (their sum matches the measured end-to-end within ~1 %).
- Parity covers one synthetic clip and fp32 only.

## Reproduce

See [README.md](README.md).
