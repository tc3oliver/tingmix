# Licensing notes for TingMix artifacts

Not legal advice. The maintainer's position: dataset licences govern redistribution of the data, which TingMix never
does; trained weights and n-gram count tables are published as models. Sources are attributed below.

| Artifact | Built from | Proposed terms | Status |
|---|---|---|---|
| Source code (this repository) | — | Apache-2.0 | published |
| TingMix-Large (reader + LoRA delta) | Common Voice zh-TW/en (CC0), VoxPopuli (CC0), LibriSpeech, AMI, FLEURS, People's Speech clean (CC BY), ASCEND (CC BY-SA 4.0), Taiwan Parliament speech (CC BY-NC 4.0), NTUML2021 train (no licence stated) | Apache-2.0 (model card) | published (HF tc3oliver/TingMix-Large) |
| LM-CN (character 5-gram) | CV + Parliament + ASCEND + NTUML-train transcripts (dev/test sentences removed) | shipped with the model | published |
| LM-O (character 5-gram) | Common Voice zh-TW transcripts only (CC0) | CC0-derived; can ship with either model | built |

Encoder: `google/embeddinggemma-2` is not redistributed. Users download it from Google and accept its terms and the
Gemma prohibited-use policy; the LoRA delta is only usable together with it.

Attribution required for CC BY / CC BY-SA sources: LibriSpeech (Panayotov et al. 2015), AMI (Carletta et al.),
FLEURS (Conneau et al. 2022), People's Speech (Galvez et al. 2021), ASCEND (Lovenia et al., LREC 2022), VoxPopuli
audio (© European Union – Source: European Parliament). Common Voice: speakers must not be identified.

NTUML2021: the released model uses its train split; its test split is used only for evaluation.
