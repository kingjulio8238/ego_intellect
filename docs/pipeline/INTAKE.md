# Vendor intake contract (`intake/v1`)

What a vendor delivers. A delivery is a folder with `intake.json` at its root, plus the files it
lists. Schema: `tools/envpkg/schemas/intake_v1.schema.json`. Check a delivery with:

```
python3 -m tools.envpkg intake <delivery>/intake.json
```

The check needs `jsonschema` and `ffprobe`. `--no-probe` skips the video checks and is not
acceptable for acceptance.

## Delivery manifest

| Field | Required | Rule |
|---|---|---|
| `delivery_id`, `vendor.id`, `vendor.name` | yes | ids: letters, digits, `_ . -` |
| `license.terms` | yes | the contract or licence reference |
| `license.commercial_use`, `derivatives_allowed`, `consent_obtained`, `pii_reviewed` | yes | must all be `true` |
| `clips[]` | yes | one or more; `clip_id` unique within the delivery |

## Per clip

| Field | Required | Rule |
|---|---|---|
| `video.path`, `sha256`, `bytes` | yes | relative to the delivery folder; the hash and size must match |
| `task.object_prompt` | yes | noun phrase naming the one manipulated object ("a tissue box"); it prompts the object detector |
| `task.instruction` | yes | what the hands do ("lift the box and hold it") |
| `task.category` | no | `pick_place`, `lift_hold`, `push`, `open_close`, `pour`, `handover`, `other` |
| `capture.device`, `capture.mount` | yes | mount: `head`, `glasses`, `chest`, `handheld`, `other` |
| `capture.captured_at` | no | ISO 8601 |
| `capture.intrinsics` | no, preferred | `pinhole` / `opencv` / `opencv_fisheye`; `fx fy cx cy` in pixels at the video's `width × height` (which must match); up to 8 distortion coefficients; pinhole means zero distortion |
| `capture.imu` | no, preferred | CSV `imu_csv/1`: header `t_s,ax,ay,az,gx,gy,gz` (s, m/s², rad/s, camera frame), strictly increasing time, ≥ 100 Hz; `time_offset_s` = IMU time of the first video frame; must cover the whole video |
| `content.hands_visible` | yes | `both`, `left`, `right` |
| `content.single_rigid_object` | yes | must be `true` (one rigid object per clip) |
| `content.faces_visible` | yes | must be `false` |

## Video envelope

| Property | Accepted | Run end to end so far |
|---|---|---|
| Container / codec | MP4 or MOV / H.264 or H.265 | MP4 / H.264 |
| Resolution | short side ≥ 480, long side ≤ 3840 | 1280×720, 1280×800 |
| Frame rate | 24–60 fps, constant | 30 fps |
| Duration | 2–60 s | 13 s, 26 s |
| Content | egocentric, hands and the object in view for most of the clip | tissue_box, airplane |

Reconstruction cost and time grow with the frame count. Clips that pass intake can still be
rejected by the triage gate (stage 3), which checks content rather than format.

## Who uses the optional fields

The pipeline currently estimates intrinsics (AnyCalib / MoGe) and gravity (GeoCalib) from the
video. Vendor intrinsics and IMU are checked and recorded at intake, but are not yet used. Using
them is a reconstruction change (fewer estimation steps, metric gravity), queued behind the
build stages. Asking for them now avoids a second request to vendors later.

## Checks

1. Schema, including the licence and content attestations.
2. Each listed file exists inside the delivery, is not a symlink, and matches its size and
   sha256.
3. ffprobe: container, codec, resolution (rotation-aware), constant frame rate, duration.
4. Intrinsics: principal point inside the image; size equal to the video's.
5. IMU: header, numeric rows, strictly increasing time, rate, and coverage of the video span.

Verified on the two sample clips (tissue_box 1280×720, 30 fps, 13 s, 390 frames; airplane
1280×800, 30 fps, 26 s): both are inside the envelope.
