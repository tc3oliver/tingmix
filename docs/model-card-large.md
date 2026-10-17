# TingMix-Large model card

Released: [tc3oliver/TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large) (the Hugging Face card is the
authoritative copy). The released version is additionally fine-tuned with the NTUML2021 train split.

## Model

- Encoder: `google/embeddinggemma-2` audio tower (~304.8M parameters, 25 frames/s, 1536-d `last_hidden_state`),
  same as [Embed2ASR-v1](https://huggingface.co/tc3oliver/Embed2ASR-v1), base weights unchanged.
- Reader: GRU-CTC, d 768 / BiGRU 384 x 2 / 4 conv blocks (22.2M parameters; CJK characters + 400 English BPE
  pieces, 5,025 units), trained from scratch on frozen encoder features (Embed2ASR v0.1.0 trainer, 80 epochs).
- Adaptation: LoRA (rank 16, alpha 32) on q/k/v/post and both feed-forward blocks of all 12 encoder layers
  (~5.5M parameters), trained jointly with the reader (encoder LR 1e-4, reader LR 2e-5, 12k-step cosine). A release
  would contain the LoRA delta + reader (tools/export_release.py), not the encoder.
- Decoding: greedy CTC, or CTC prefix beam search with a character 5-gram LM (CJK units scored, English pieces
  LM-neutral).
- Streaming: buffered re-decoding of the open speech segment (energy VAD, 600 ms endpoint). It is not a native
  streaming model; partial results can change, final results never do.

## Training data

Common Voice zh-TW, Taiwan Legislative Yuan (Parliament) speech, LibriSpeech, VoxPopuli,
AMI, FLEURS en_us, ASCEND (Mandarin-English code-switching). No training audio, transcripts, manifests or features are
distributed. Test sets used for the public benchmark were excluded from training and LM text by audio path and by
sentence (non-trivial sentences).

## Evaluation

Model selection and LM weights use development sets only. Test results are produced once per release candidate with
`evaluation/accuracy.py`, from one checkpoint, and reported per metric: CER (Taiwanese Mandarin), MER
(code-switching: CJK characters + English words), WER (English). Results of the single test run: [benchmark.md](benchmark.md).

Known limitations: NTUML2021 is evaluation-only (licence not stated); ASCEND speakers are from Hong Kong, so it is not a
Taiwanese-accent code-switch test; Parliament references are not verbatim (deletion/insertion floor); English accuracy
remains well behind Whisper large-v3.

## Licence review (to complete before any weight release)

- Parliament speech: CC BY-NC 4.0 — weights and the Parliament-derived LM are likely non-commercial.
- ASCEND: CC BY-SA 4.0 — attribution; conservative reading: ShareAlike for derived weights/LM.
- People's Speech / AMI / FLEURS: CC BY (attribution). Common Voice / VoxPopuli: CC0 (VoxPopuli audio: EP notice).
- EmbeddingGemma 2: Google's terms and prohibited-use policy apply to any use of the encoder and its adapters.
