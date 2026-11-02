# Presenter notes — four-slide MuJoCo SLAM presentation

Planned length: about 3–4 minutes, including the 24-second video. The PPTX contains exactly four slides and the same speaker notes. The PDF is a static four-page handout.

## Before presenting

- Open `MuJoCo_SLAM_4_slides.pptx` in PowerPoint and test the embedded video on slide 3. Enter Slide Show and click the video; it is not configured to autoplay.
- The MP4 is embedded, so PowerPoint does not need an internet connection or CUDA. Compatibility with other presentation applications varies. Keep `../blackout_preview.mp4` available as a fallback, or use the online video link.
- For the PDF, open the MP4 separately. No simulator, Python environment or GPU is required during the talk.
- State that the video shows inputs, and that the numerical example is an oracle-initialized sensor check. Do not present it as a completed MM3DGS or tight-coupling experiment.

## Slide 1 — Testing SLAM when vision fails

Timing: 0:00–0:35

### Suggested script

SLAM estimates camera motion while building a map. MM3DGS combines vision, depth, and inertial measurements with a 3D Gaussian map. We built a controlled MuJoCo environment to test what happens when the visual observations become weak or disappear. The aim is a repeatable comparison of tracking methods, especially methods using the IMU.

### Technical backup / likely questions

The reference is Sun et al., MM3DGS SLAM, 2024, arXiv:2404.00923. Section III-D in the attached paper uses open-loop inertial propagation to initialize visual tracking and leaves closed-loop fusion with bias estimation as future work. The checked official repository also contains an optional inertial pose-prior loss; it is not a complete bias-estimating inertial factor graph. A blackout motivates testing inertial robustness; it does not establish that tightly coupled fusion is necessary.

## Slide 2 — A reproducible visual–inertial testbed

Timing: 0:35–1:25

### Suggested script

The scene is a furnished studio made procedurally in Python. Furniture and plants use MuJoCo primitives, and six textures are generated locally. The same 24-second motion is recorded with a 30-hertz RGB-D camera and a 100-hertz IMU. We export calibrated, synchronized measurements, keep ground truth separate for scoring, and replay exactly the same data into each estimator. The three conditions are rich visuals, low texture with blur and dimming, and a complete visual blackout.

### Technical backup / likely questions

Scene: studio-v2; no downloaded room, scan, mesh pack, or generated-image service was used. There are six deterministic PNG textures, exported beside scene.xml. Motion: prescribed 6-DoF loop with changing acceleration, yaw, roll, pitch, and height; stationary from 0–2 s and 22–24 s. MuJoCo inverse dynamics supplies the wrench and forward dynamics produces the accelerometer and gyro readings. A 600 Hz sampling grid gives integer strides for 30 Hz cameras and 100 Hz IMU. Both endpoint timestamps are included: 721 images and 2,401 raw IMU samples. A 100 Hz / 30 Hz bridge groups IMU intervals 3/3/4; the initial t=0 sample has no preceding integration interval.

Calibration: T_A_B maps B into A; world z is up; body/IMU axes are x forward, y left, z up; camera axes are x right, y down, z forward. The camera is 0.08 m forward and 0.04 m above the IMU. Accelerometer readings are specific force R_W_B.T @ (a_W - g_W), so a stationary, level sensor reads +9.81 m/s² on body z. Timestamps are integer nanoseconds on a shared simulation clock; IMU is delivered before an image at the same time.

Sensor model: nonzero biases and random walks; seed 7 in the shown run. Acceleration white-noise density is 0.002 m/s²/√Hz and gyro density is 0.0002 rad/s/√Hz. Bias random-walk coefficients are 0.0003 (acceleration) and 0.00002 (gyro), with discrete increments scaled by √dt. Initial planted biases are (0.025,-0.015,0.020) m/s² and (0.001,-0.0006,0.0008) rad/s. Sample noise sigma is density × √rate. Replay hides planted bias values, seed, scenario timing, and ground truth from the estimator.

Depth is optical-axis z in metres, zero invalid, with 5 mm independent Gaussian noise and a 0.1–20 m validity range. This is a simple axial-depth model, not stereo reconstruction. Images use instantaneous global exposure; low-texture blur and dimming are image-space approximations. Rolling shutter and clock offsets are not simulated. This is a sensor benchmark rather than a locomotion/controller environment.

## Slide 3 — Demo: four seconds without vision

Timing: 1:25–2:10, including 24-second video

### Suggested script

The left panel is a clean reference so we can see that the rig keeps moving. It is never given to the estimator. The right panel is the actual camera input. At eight seconds it goes black, and depth is also invalid until twelve seconds. The IMU keeps measuring throughout. When vision returns, we can test whether tracking recovers. This is a video of simulated sensor inputs, not a reconstructed SLAM map.

### Technical backup / likely questions

Click the embedded video while presenting slide 3. It is H.264, 960×420, with 721 frames at 30 fps and duration 24.033 seconds including the endpoint frame. Play once without pausing. Point out the beginning of the blackout at 8 s and recovery at 12 s. There are exactly 120 black RGB frames and all corresponding depth pixels are invalid. Outside this interval the scene remains visually rich; the separate low_texture sequence removes material variation and adds blur, dimming, and pixel noise over the same interval.

Depth must also be gated because healthy RGB-D depth can permit tracking despite poor RGB. For other experiments, depth_mode=off provides a monocular sequence and depth_mode=clean provides a depth-assisted control. All matched scenarios share geometry, trajectory, calibration, and IMU noise. The full studio run has byte-identical raw IMU and camera ground truth to the original checkerboard run. Scene appearance changes visual observability, so comparisons should use the same scene version and exported sequence.

Playback fallback: open ../blackout_preview.mp4 from this folder, or use the online MP4 link on the slide. The embedded video is self-contained in PowerPoint, but video playback varies by presentation application. PDF is a static four-page handout: use its external video link. No CUDA or running simulator is needed to show this clip.

## Slide 4 — Verified simulator. Estimator comparison comes next.

Timing: 2:10–3:15

### Suggested script

Thirty checks passed for the simulator, including physical IMU output, timing, calibration, depth geometry, and blackout handling. As a sensor sanity check, we initialized both predictors with the true pose and velocity at the start of the outage. Constant velocity ended about 4.44 metres away, while noisy IMU propagation ended about 5 centimetres away. These are not SLAM accuracy results. We still need to run the full MM3DGS pipeline on CUDA and compare visual-only tracking, IMU-initialized tracking, and tightly coupled fusion over multiple seeds and outage lengths.

### Technical backup / likely questions

Verification: Linux x86_64, Python 3.12, MuJoCo 3.3.7, EGL/Mesa CPU rendering; pytest reported 30 passed in 167.13 s. Checks include actual accelerometer/gyro versus rigid-body equations; analytic trajectory derivatives; camera extrinsics and depth versus ray intersections; deterministic resets; matched geometry and sensor noise; frame/IMU ordering; ground-truth isolation; metric trajectory scoring; saved scene/texture reload; furniture clearance along the path; and the official loader/result-conversion bridge.

The sanity check is single-seed inertial dead reckoning, not a visual/inertial estimator. It starts at t=8 s with the oracle (ground-truth) boundary pose and velocity. Biases are estimated from the first 1.5 stationary seconds with known initial orientation; planted bias values are not supplied to propagation. IMU endpoint position error at 12 s is 0.05105 m versus 4.43730 m for constant velocity; outage position RMSE is 0.02199 m versus 2.09482 m. Neither result validates a 3D Gaussian map, reconstruction quality, closed-loop bias estimation, or tight coupling.

A well-initialized loosely coupled filter or inertial propagator can also bridge a short blackout. Tight coupling is most meaningfully tested against such baselines when weak visual constraints remain and during recovery. Even tightly coupled VIO drifts during complete visual loss. Compare: (1) visual tracking plus constant velocity, (2) the paper’s IMU-initialized tracking, (3) joint pose/velocity/bias estimation with visual and inertial residuals. Use identical data, several seeds and outage durations, and the rich and low_texture controls.

Report metric ATE, 1-second translation/rotation RPE, overall and per-interval pose coverage, outage drift, and return-to-vision recovery. The supplied evaluator fixes SE(3) alignment at the first matched pose, does not fit scale, and does not realign after the outage. The official CUDA tracking/mapping pipeline was not run in the CPU-only workspace. Its current unmodified tracker may fail on empty visual/depth masks; this is a benchmark outcome, not a reason to invent measurements.

## Reproduce or extend the demonstration

The scene snapshot used to prepare this deck is commit `3073c7b5fc443c7adaf081013325ef04fd883b06` (`studio-v2`). Use the same scene and measurements across estimators.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
mm3dgs-bench generate --scenario suite --out runs/presentation --width 480 --height 360
python scripts/make_preview.py runs/presentation/blackout --out runs/presentation_preview.mp4
pytest -q
```

For the official CUDA pipeline, follow the upstream MM3DGS environment instructions and this repository’s README. The simulator environment and official CUDA environment are separate. The supplied runner can prepare configs with `--prepare-only`; keep `tracking.use_gt_pose=false` and compare CV versus IMU initialization. Add a separate tightly coupled estimator through the documented replay interface.

## Sources and attribution

- Paper: Lisong C. Sun et al., *MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements* (2024), https://arxiv.org/abs/2404.00923.
- Official project: https://vita-group.github.io/MM3DGS-SLAM/; code: https://github.com/VITA-Group/MM3DGS-SLAM. The data bridge was checked against upstream commit `1fae9f5d1ad50a6159717b711ae54325422202e2`.
- Simulator and implementation details: https://github.com/WhiteLion1212/MuJoCoTest/blob/3073c7b5fc443c7adaf081013325ef04fd883b06/README.md.
- Recorded validation: https://github.com/WhiteLion1212/MuJoCoTest/blob/3073c7b5fc443c7adaf081013325ef04fd883b06/VALIDATION.md.
- Video: https://github.com/WhiteLion1212/MuJoCoTest/blob/main/docs/blackout_preview.mp4. All studio geometry and six textures are generated by the repository; there is no external graphics pack.

## Rebuild the slides

`python scripts/build_presentation.py` requires optional authoring packages `python-pptx` and `Pillow`, plus `ffmpeg` for the poster frame. To regenerate the PDF with LibreOffice, export `work/presentation/MuJoCo_SLAM_handout_source.pptx` (the static poster copy created by the builder), then save the four-page PDF as `docs/presentation/MuJoCo_SLAM_4_slides.pdf`. The editable master contains the embedded movie.
