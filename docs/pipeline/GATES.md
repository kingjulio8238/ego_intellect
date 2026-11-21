# Gates: triage (stage 3) and reconstruction QA (stage 5)

There are two CPU gates. Each one stops a clip before money is spent on it:

| Gate | Runs | Saves when it rejects |
|---|---|---|
| **Triage** (`tools/gates/triage.py`) | before `reconstruct`, on the source video | the whole GPU reconstruction (~$0.6–1.2 per clip) plus the robot phase |
| **Reconstruction QA** (`tools/gates/recon_qa.py`) | right after `reconstruct`, on the bundle (inside `run_e2e.sh inspect`) | the robot phase (retarget + simulate, one rg GPU container per clip), and keeps a bad reconstruction from being packaged |

All limits live in one file, `tools/gates/limits.json`. Each limit carries a one-line reason, and
that reason is what a rejected clip's report says. A gate that ran always writes its verdict
as data:

- `RUN_ROOT/qa/triage.json`
- `RUN_ROOT/qa/reconstruction.json`

`run_e2e.sh triage` and `run_e2e.sh inspect` exit 0 when the clip is accepted and **3 when it
is rejected**. A rejected clip is not a failure. Batch mode reports rejected clips separately
(`rejected` in `pipeline.json`, `REJECTED` in the summary), runs no further steps for them,
and does not count them as failed. The environment package records both verdicts
(`qa.gates`). The builder refuses a rejected clip unless `--allow-flagged` is passed.

Batch manifests get both gates by default. `"triage": false` or `"recon_qa": false` turns one
off.

## Triage checks

| Check | Limit | Catches |
|---|---|---|
| `format_errors` | 0 | anything outside the intake envelope ([INTAKE.md](INTAKE.md)): container, codec, resolution, frame rate, duration |
| `hand_frame_fraction` | ≥ 0.25 | no hands: robot footage, screen recordings, scenery |
| `hand_size_median` | ≥ 0.08 of the short side | hands too small for a close-up egocentric view (third-person footage) |
| `sharp_frame_fraction` | ≥ 0.6 (Laplacian variance ≥ 60 at 640 px) | motion blur, out of focus |
| `exposure_ok_fraction` | ≥ 0.7 (mean luma 40–215, ≤ 25% clipped) | too dark, blown out |
| `frozen_fraction` | ≤ 0.3 (consecutive frames nearly identical) | stalled capture, static camera, slideshows |
| `cut_count` | 0 (colour-histogram jump between consecutive frames) | edited clips with more than one shot |

Hands are detected with the MediaPipe hand landmarker (CPU) on frames sampled at 2 fps, at most
60. The model is baked into the control image with a checked sha256. One clip takes about 4 s on
a laptop CPU.

## Reconstruction QA checks

The bundle-only checks run on any bundle, including imported ones:

| Check | Limit | Catches |
|---|---|---|
| `camera_valid_fraction` / `object_valid_fraction` / `hand_valid_fraction` | ≥ 0.95 / 0.9 / 0.5 | tracking lost |
| `camera_speed_max_mps` / `camera_turn_max_radps` | ≤ 2.0 / 6.0 | SLAM jumps (faster than a head moves) |
| `object_accel_p95_mps2` / `wrist_accel_p95_mps2` | ≤ 2.5 / 3.0 | jittery trajectories |
| `object_extent_max_m` | 0.03–1.0 | scale or unit errors |
| `hand_object_gap_p20_m` | ≤ 0.06 | hands that never reach the object, which points to a hand or object depth/scale error |
| `wrist_inside_object_fraction` | ≤ 0.15 (deeper than 1 cm inside the object's convex hull) | object pose or scale wrong |

These checks compare against the clip's own SAM2 masks, when the run root has them:

| Check | Limit | Catches |
|---|---|---|
| `object_mask_fraction` | ≥ 0.8 | object not segmented |
| `object_iou_median` / `object_iou_p10` | ≥ 0.80 / 0.77 | the posed object mesh does not cover the object's mask. Hand pixels are excluded, so occlusion does not count |
| `wrist_on_hand_mask_fraction` | ≥ 0.8 (within the hand mask's box + 30%) | wrists off their hands, or left and right swapped |

A check that cannot be measured is listed as `skipped` in the report and never counted as
passed. Without masks, all four mask checks are skipped. One clip takes 1.5–4 s.

## Calibration (2026-11-21)

Reproduce with `python3 -m tools.gates.calibrate recon|triage ...`. It exits non-zero on any
mismatch.

**Reconstruction QA: 52/52 labels matched.**

- **Good (14 runs, all pass):** every tissue_box reconstruction with masks on our volumes:
  b2_tb1–3, b5_tb1–5, b5_tbseed, h2/h3/h4 (A100, H100), r4_tbseed, tissue_box_fast3.
  - Good ranges: object IoU median 0.864–0.881, p10 0.837–0.852; camera ≤ 0.13 m/s and
    ≤ 0.11 rad/s; wrist accel p95 ≤ 1.14 m/s²; wrist on hand mask 1.0.
- **Real bad (2 runs, both rejected):** both airplane reconstructions, b2_ap1 and b5_ap2.
  - Their cameras jump up to 25 rad/s (48° between two frames) while the video shows a still
    camera. That is a DROID-SLAM failure on the mostly white scene.
  - Wrists land on their hand masks only 47–48% of the time. Object IoU p10 is 0.41–0.53.
  - These clips had been through throughput batches but never through a quality check. This
    gate is the first thing to catch them.
- **Synthetic (12 defects × 3 good runs, all rejected).** Each defect changes one thing:
  - object 3 cm off;
  - object trajectory 10 frames late;
  - object jitter 5 mm;
  - wrist jitter 1 cm;
  - hands 50% too deep;
  - camera jump 30 cm;
  - object pose lost on 40% of frames;
  - left and right hands swapped;
  - object scale ×1.3 and ×0.7;
  - units ×0.1;
  - the segmenter tracked a hand instead of the object.

**Triage: 32/32 labels matched.**

- **Good (6 clips, all accepted):** tissue_box, airplane, two head-camera clips (ep00 boxes,
  ep01 paper), two HOT3D sequences (fisheye, grey).
  - The lowest margins are HOT3D's hand fraction (0.4 against a limit of 0.25) and
    tissue_box's frozen fraction (0.14 against 0.3; the clip has real duplicate frames).
- **Real bad (5 clips, all rejected):** a third-person clip (no close-up hands, static), a UI
  screen recording, two robot rollouts (no human hands, and below the envelope), and a
  multi-view wrist-camera strip (format).
- **Synthetic (7 defects × 3 good clips, all rejected):** blurred, dark, overexposed, frozen
  (each frame repeated ×10), scene cut, 1.2 s long, 320 px wide.

**On Modal (newacc control container, 2026-11-21):**

- `run_e2e.sh triage` on r4_tbseed: ACCEPT in 3.7 s.
- `run_e2e.sh inspect` on r4_tbseed: PASS in 3 s, exit 0.
- `run_e2e.sh inspect` on b2_ap1: REJECT in 5 s, exit 3. The reasons printed and stored in
  `qa/reconstruction.json` are the camera jumps, the wrists off their hands and the object
  IoU.
- The first smoke found that MediaPipe needs `libEGL.so.1`, even on CPU. The control image
  now installs it, and the image build fails unless it can create the hand detector and run
  it once.

## Known limits

- **The reconstruction QA good set is one object in one scene** (tissue_box, 14 runs). The IoU
  limits (0.80 / 0.77) sit about 0.07 below that object's range. Thin, small or shiny objects
  may score lower when correct. Recalibrate when the first new objects come through: run
  `calibrate recon` with them added as `--good`.
- **Detection floors:**
  - Hand depth errors below about +40% are not caught on tissue_box. At +25% the wrists
    stay within 3 cm of the box. The physics QA's grasp-contact checks are the backstop.
  - A 10-frame object lag is caught only through IoU p10, with a 0.02 margin.
- **Triage does not judge the object or the scene's suitability for SLAM.** The feature-count
  metric `texture_corners_median` is recorded but not gated. Airplane is the low outlier
  (157 against 435–818), and the airplane reconstructions do fail on SLAM. One example is not
  enough to set a limit, and reconstruction QA catches the failure anyway, at the cost of one
  reconstruction.
- **Third-person footage with large, visible hands** would pass triage. The hand-size limit
  only rejects distant hands.
