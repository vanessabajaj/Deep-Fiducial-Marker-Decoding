# Project 2 — Dataset sources

All three components are open access. Nothing here requires a request or a login.

---

## 1. TRAINING DATA — Flying-ArUco v2 (synthetic)

**Landing page:** https://zenodo.org/records/14053985
**DOI:** 10.5281/zenodo.14053985

Synthetic images built by overlaying ArUco markers onto backgrounds sampled from the
MS COCO 2017 training set. This is the dataset DeepArUco++ was trained on, so using it
keeps our training protocol directly comparable to the published baseline.

**Base archive:** `flyingarucov2.tar.gz` — 2500 images, each with a `.json` giving the
2D corner coordinates, the marker ID, and the rotation (how many 90-degree counter-clockwise
turns the marker has undergone).

The record also provides pre-built detection datasets at increasing augmentation levels
(`det_luma`, `det_luma_b`, `det_luma_bc`, and further variants — check the record page for
the current complete list):

| suffix | augmentation applied |
|---|---|
| `luma`   | simulated lighting changes |
| `_b`     | + blur |
| `_bc`    | + colour shift |

The authors recommend the pre-generated archives over rebuilding, because their generation
script uses random operations and results will vary otherwise.

**Direct download:**
```
https://zenodo.org/records/14053985/files/flyingarucov2.tar.gz
```

---

## 2. TEST DATA — Shadow-ArUco (real)

**Landing page:** https://zenodo.org/records/10791293
**DOI:** 10.5281/zenodo.10791293
**License:** CC-BY-4.0
**Total:** 4.6 GB, 8652 frames at 800x600

Real photographs: ArUco markers attached to the walls in the corner of a darkened room,
with a video of moving shadows and lighting patterns projected over the corner, recorded
from 6 different points of view. Ground truth was tagged using a method based on the
classic ArUco technique, with the positions and IDs of undetected markers added manually.

Each `video_N.tar.gz` contains the frames plus a `corrected_annotations/` subdirectory
holding one JSON per frame with the corner coordinates and ID of every marker.

| file | size | md5 |
|---|---|---|
| `video_1.tar.gz` | 760.0 MB | `1c8bd58fb067a453487dfff783416674` |
| `video_2.tar.gz` | 701.6 MB | `1df7a6a568db96caf4e205a26050598f` |
| `video_3.tar.gz` | 709.9 MB | `a21caebb578b5760c0efdd30472298bf` |
| `video_4.tar.gz` | 812.2 MB | `d1d82d6054a1e61a6998e0069d8626e7` |
| `video_5.tar.gz` | 687.0 MB | `3e8acd8de355941aac414ef8ad44eb38` |
| `video_6.tar.gz` | 893.2 MB | `be62ecc5969faec0e1a466b69e7cdda8` |

**Direct download pattern:**
```
https://zenodo.org/records/10791293/files/video_1.tar.gz
```

Verify the md5 before extracting. A truncated download still partially extracts, which
produces corrupt frames that analyse without raising any error.

**Contents (verified on video_1):** 1433 frames, exactly 12 markers per frame,
IDs 238-249 throughout.

---

## 3. DICTIONARY — ARUCO_MIP_36h12

**Source:** https://github.com/AVAuco/deeparuco — file `impl/aruco.py`

Not downloadable as a dataset; it is a 252-entry table of hexadecimal codewords
hardcoded in the reference implementation (250 markers plus an all-black and an
all-white entry used as negatives during training).

| property | value |
|---|---|
| markers | 250 |
| bits | 36 (6x6 grid) |
| border | 1 cell, black |
| minimum Hamming distance | 12 |
| max error correction | 5 bits |
| max erasure correction | 11 cells |

**This is not any OpenCV predefined dictionary.** Every `cv2.aruco` predefined dictionary
returns zero detections on this data. The DeepArUco++ paper describes the markers as being
sampled from "the ArUco DICT 6X6 250 dictionary", which denotes *6x6 bits, 250 markers* —
not the OpenCV constant `DICT_6X6_250`, whose codewords are different. Verified: rendering
markers from MIP36h12 and decoding them gives 9/9 with a dictionary built from that table
and 0/9 with `DICT_6X6_250` on identical images.

Use `mip36h12.py` (in this project) to construct a working `cv2.aruco.Dictionary`.

---

## Experimental design

Train on synthetic (Flying-ArUco v2), test on real (Shadow-ArUco). This is the protocol
DeepArUco++ used and it is what the data supports: Shadow-ArUco is six fixed camera
positions, so adjacent frames are near-duplicates and its effective scene diversity is 6,
not 8652. It is an evaluation set, not a training set.

Splits are by viewpoint sequence, never by frame. One viewpoint (`video_6`) is held out
entirely and not evaluated until final reporting.

---

## Citation

```bibtex
@article{berral2024imavis,
  title   = {DeepArUco++: Improved detection of square fiducial markers in
             challenging lighting conditions},
  journal = {Image and Vision Computing},
  pages   = {105313},
  year    = {2024},
  issn    = {0262-8856},
  doi     = {10.1016/j.imavis.2024.105313},
  author  = {Rafael Berral-Soler and Rafael Mu\~{n}oz-Salinas and
             Rafael Medina-Carnicer and Manuel J. Mar\'{i}n-Jim\'{e}nez}
}
```

Both datasets are by Berral-Soler, Sarmadi, Muñoz-Salinas, Medina-Carnicer and
Marín-Jiménez (University of Córdoba, Artificial Vision Applications group).
