# Licensing of TingMix artifacts (review draft)

Not legal advice. This is the checklist used before any model or LM is published; publishing needs an explicit
maintainer decision. Training data itself is never redistributed.

| Artifact | Built from | Proposed terms | Status |
|---|---|---|---|
| Source code (this repository) | — | Apache-2.0 | published |
| TingMix-Large (reader + LoRA delta) | Common Voice zh-TW/en (CC0), VoxPopuli (CC0), LibriSpeech, AMI, FLEURS, People's Speech clean (CC BY), ASCEND (CC BY-SA 4.0), **Taiwan Parliament speech (CC BY-NC 4.0)** | non-commercial (CC BY-NC-SA 4.0 is the conservative reading) | not published |
| TingMix-Large-Open (reader + LoRA delta) | same, **without Parliament** | CC BY-SA 4.0 (ShareAlike because of ASCEND), with attribution to every source | in training |
| LM-C (character 5-gram) | CV + Parliament + ASCEND transcripts | non-commercial (contains Parliament n-grams) | not published |
| LM-O (character 5-gram) | Common Voice zh-TW transcripts only (CC0) | CC0-derived; can ship with either model | built |

Encoder: `google/embeddinggemma-2` is not redistributed. Users download it from Google and accept its terms and the
Gemma prohibited-use policy; the LoRA delta is only usable together with it.

Attribution required for CC BY / CC BY-SA sources: LibriSpeech (Panayotov et al. 2015), AMI (Carletta et al.),
FLEURS (Conneau et al. 2022), People's Speech (Galvez et al. 2021), ASCEND (Lovenia et al., LREC 2022), VoxPopuli
audio (© European Union – Source: European Parliament). Common Voice: speakers must not be identified.

Evaluation-only data: NTUML2021 (no licence stated) is never used for training or the LM; only scores are published.
