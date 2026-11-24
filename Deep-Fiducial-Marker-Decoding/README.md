# Deep Fiducial Marker Decoding under Occlusion

A diagnostic study of how a published learned ArUco decoder, the DeepArUco++ decoder (Berral Soler et al., Image and Vision Computing 2024), reads markers that are partly covered. All experiments run in Google Colab notebooks N0 to N17. Every notebook in this repository is saved with its outputs, so the numbers can be read without running anything.

The full write up is in [`report/Project2_Technical_Report_v5.pdf`](report/Project2_Technical_Report_v5.pdf). Every number the notebooks saved, N0 to N17, is collected in [`results/all_results.xlsx`](results/all_results.xlsx) (28 sheets; N17 is in `n17_dev_and_heldout`).

## Main findings

Development cameras unless stated. P is the published decoder. Trained arms are means over 5 seeds (N14 to N17). Findings 1, 1b, 3, 4 and 6 were rerun once on the held out camera in N17 and hold there.

| # | Finding | Key number | Notebooks |
|---|---|---|---|
| 1 | Painting only the border ring, with all 36 data cells visible, damages decoding, and the damage grows with paint brightness. The classical reader falls even further, but refuses these markers instead of misreading them. Confirmed on the held out camera. | P 92.36% (black paint) to 48.63% (white paint); held out 84.46% to 47.77%; classical 73.1% to 0% (100% refused) | N8, N10, N16, N17 |
| 1b | The border effect is a training gap: a decoder trained with painted borders loses it, with no cost on clean markers. | spread (highest minus lowest accuracy over five paint brightnesses) 55.58 for N against 4.50 for BB, difference −51.08 [−56.53, −45.63]; held out confirmed | N16, N17 |
| 2 | Using the brightness scale of the clean image reduces the border spread by a similar amount in the neural and the classical decoder. | 44.6% and 41.7% | N11, N16 |
| 3 | Training on covered markers gives large robustness to grey and white rectangles, also with the training recipe held fixed. | 22 to 50 points; held out 28 to 49 | N5, N16, N17 |
| 4 | Training on grey rectangles makes black rectangles slightly worse. | about 5 points at 3.2 to 11.5 hidden cells; held out +5.5 to +7.4 at 6 and 12 cells | N5, N16, N17 |
| 5 | Black rectangles are a separate problem: only an arm trained on black alone detects them, and it then fails on lighter rectangles. | 89.6 to 94.7% of hidden black cells flagged | N15 |
| 6 | The published decoder is right about half the time on hidden cells, with confidence near 1. Wrong answers survive the paper's limit 9 reject check; limit 3 removes them only by refusing almost every covered marker. | white border: 52.6% wrong, 48.2% at limit 9, 0% at limit 3 with 92.5% refused | N2, N12, N17 |
| 7 | A separate hidden cell output can be learned; whether it improves reading is unresolved with 5 seeds. | D against B inconclusive | N3b, N14, N15 |
| 8 | Withdrawn. The earlier soft decoding gain came from the released code breaking Hamming ties toward low marker ids. | ties in 3 to 47% of markers | N12 |
| 9 | Shadow ArUco uses ARUCO_MIP_36h12 (in OpenCV as `DICT_ARUCO_MIP_36h12`), while the paper's wording points to `DICT_6X6_250`, which reads nothing on this data. | 9 of 9 against 0 of 9 | N0 |

Side results: structured erasure decoding over 21 edge strip hypotheses (a known method family: GMD, Chase, CompositionalNets) gains up to 29.6 points on edge rectangles and loses up to 17.1 on centre rectangles (N15). Random brightness training, max against average pooling and the hidden cell output showed no difference that 5 seeds could detect (N14).

## Notebooks

| Notebook | Purpose | Key result |
|---|---|---|
| [N00](notebooks/N00_data_preparation.ipynb) | Download and MD5 check both datasets, verify the dictionary on all six cameras, make the camera split and the crc32 seed table | 8,652 frames; ARUCO_MIP_36h12 confirmed, DICT_6X6_250 control zero |
| [N01a](notebooks/N01a_erasure_headroom.ipynb) | Published decoder on clean markers with dataset corners | 99.43% hard, room for erasure decoding measured |
| [N02](notebooks/N02_occlusion.ipynb) | Occlusion sweep and per cell confidence | confidence about 0.99 on hidden cells |
| [N03](notebooks/N03_learned_erasure.ipynb) | First trained arms B and C with a hidden cell output | apparent 21.8 point gain, later traced to an undertrained arm B |
| [N03b](notebooks/N03b_arm_D.ipynb) | Arms B, C, D trained properly | hidden cell output F1 0.971, gain at most 1.71 points |
| [N04](notebooks/N04_controlled_occlusion.ipynb) | Rectangle brightness and position with fixed geometry | brightness changes accuracy with hidden cells fixed |
| [N05](notebooks/N05_paired_significance.ipynb) | Paired cluster bootstrap with Holm correction, including held out camera | trained arm loses to published decoder on black, 12 of 12 conditions |
| [N06](notebooks/N06_mechanism_and_mitigation.ipynb) | Brightness sweep, arm E with random brightness | arm E does not fix black |
| [N07](notebooks/N07_error_localisation.ipynb) | Where the errors land, hidden against visible cells | errors spill onto visible cells |
| [N08](notebooks/N08_border_causal.ipynb) | Paint only the border or only the data region, dev and held out | border only collapses decoding |
| [N09](notebooks/N09_pooling_causal.ipynb) | Max against average pooling, arms M and G | no difference |
| [N10](notebooks/N10_classical_control.ipynb) | Classical OpenCV style decoder on the same crops | collapses harder than the neural decoder |
| [N11](notebooks/N11_normalisation_test.ipynb) | Clean image brightness scale and adaptive threshold | spread reduced by 45% and 42% |
| [N12](notebooks/N12_tie_breaking.ipynb) | Tie breaking in hard Hamming decoding and per cell calibration | finding 8 withdrawn |
| [N13](notebooks/N13_seeds_SUPERSEDED.ipynb) | **Superseded.** Seed study that mixed old and new checkpoints. Kept for the record, not cited. | none |
| [N14](notebooks/N14_retrain_mechanism.ipynb) | All arms (B, C, D, E, F, M, G) with 5 seeds, deterministic training, oracle decoding | most single run differences become inconclusive |
| [N15](notebooks/N15_structured_decoding.ipynb) | Arm H (black only) and structured erasure decoding | H detects black but fails on grey and white |
| [N16](notebooks/N16_border_and_control.ipynb) | Border training arms BB and BD, and the no rectangle control arm N | border effect removed; finding 4 holds with recipe fixed |
| [N17](notebooks/N17_heldout_and_reject_check.ipynb) | No training. Held out camera used once, results per camera, and the paper's reject check at limits 9 and 3 | every main result holds on video_6; limit 9 lets wrong answers through, limit 3 refuses almost every covered marker |

`notebooks/extras/` holds the early dictionary notebooks that found ARUCO_MIP_36h12. Note that their first OpenCV sweep ran on a broken mixed OpenCV install, which is why it found zero with every built in dictionary.

## Data

| Dataset | Use | Source |
|---|---|---|
| Shadow ArUco | testing only, 6 fixed cameras, 8,652 frames, marker ids 238 to 249 | [Zenodo 10791293](https://zenodo.org/records/10791293), CC BY 4.0 |
| Flying ArUco v2 | training our arms, 2,500 synthetic images | [Zenodo 14053985](https://zenodo.org/records/14053985) |

See [`DATASET_SOURCES.md`](DATASET_SOURCES.md). Data are not included in this repository.

## Evaluation protocol

* Development cameras video_1 to video_5, every 6th frame, 12,276 markers per condition. Camera video_6 is held out; it was used in N5 and N8 and once in N17 after all choices were fixed.
* Dataset corners are used throughout, so only the decoder is tested; no detector is involved.
* Rectangles are drawn on the photograph before cropping, at brightness 0, 64, 127, 192 or 255, on one edge or in the centre. Placement uses crc32 seeds, so every run is identical.
* Severity is calibrated to 3.2, 5.7, 8.5, 11.5, 15.6 and 18.0 hidden data cells.
* Statistics: paired cluster bootstrap by frame with Holm correction (N5 to N12); from N14 on, 5 training seeds per arm, paired t intervals over seeds, and a verdict of differs, equivalent (within 1 point) or inconclusive.
* Training from N14 on is deterministic. Checkpoints are tagged `n14-v1` with signatures.

## Running the notebooks

1. Open a notebook in Colab with a GPU runtime (an L4 was used).
2. The notebooks mount Google Drive and use this layout:
   ```
   MyDrive/aruco_p2/archives/    dataset archives (downloaded by N0)
   MyDrive/aruco_p2/artifacts/   splits.json, seed table, results JSON, all_results.xlsx
   MyDrive/aruco_p2/ckpt_n14/    trained checkpoints from N14 to N16
   ```
3. Run N0 first. N12 onwards are self contained and only need N0's artifacts; N15 and N16 reuse N14 checkpoints if present and retrain them otherwise.
4. The published weights `dec_new.h5` come from the authors' repository, which the notebooks clone at run time.

`src/deeparuco_port.py` is a PyTorch port of the published decoder (32 by 32 grey input, 6 by 6 output, layers of 8 and 4 filters with per element PReLU, 7,353 parameters) that loads `dec_new.h5`. `src/mip36h12.py` builds the ARUCO_MIP_36h12 dictionary and checks it.

## Limitations

* One dataset, one room, 12 marker identities.
* Only six cameras. Camera level intervals (N17) are about 2.3 times wider than frame intervals; every key result holds on each camera.
* The held out evidence is a single camera.
* Uniform single brightness rectangles, not real objects.
* One small decoder design. Dataset corners, so no full system numbers.
* The reject check was tested only at the paper's limits 3 and 9.

## Licence and credits

The upstream DeepArUco++ code ([AVAuco/deeparuco](https://github.com/AVAuco/deeparuco)) is AGPL 3.0. It is cloned at run time and not copied here. Shadow ArUco and Flying ArUco v2 belong to their authors.

If you use this work, please cite the DeepArUco++ paper:

> R. Berral Soler, R. Muñoz Salinas, R. Medina Carnicer, M. J. Marín Jiménez. DeepArUco++: Improved detection of square fiducial markers in challenging lighting conditions. Image and Vision Computing 152, 2024.
