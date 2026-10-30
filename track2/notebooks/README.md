| notebook | what it is |
| --- | --- |
| 00_phase2_submission_dictionary_all | **the leaderboard submission (994952).** Rebuilds the residual bank from all 695 released synthetic validation pairs, searches every Phase-2 clip, writes the dictionary-all package. It also builds a second, hybrid package (dictionary where matched, V36 elsewhere); that one was not the scored entry. |
| teacher_v23_phase1_dictionary | Phase-1 dictionary system (rank 3 in Phase 1). Same family as 00. |
| 01_v32 ... 05_v36 | the always-on neural enhancer, trained in five stages. Not the scored entry; see the report. |

Stage summary for the neural model:

| notebook | stage | starts from | selected |
| --- | --- | --- | --- |
| 01_v32_ttspairs_demucs | dns64 fine-tuned on 5,000 pseudo pairs (Indic Parler-TTS speech + validation residuals matched by BEATs) | dns64 | epoch 6 |
| 02_v33_multievent | 1-3 event mixtures, event-centred crops | V32 | best dev |
| 03_v34_noise_adaptive | adds conditioner, FiLM, strength head (zero-init) | V33 | epoch 2 |
| 04_v35_asr_constrained | genuine pairs, frozen acoustic encoder constraint | V34 | epoch 3 |
| 05_v36_teacher_distillation | gold + teacher pairs with remixing | V35 | epoch 1 |
