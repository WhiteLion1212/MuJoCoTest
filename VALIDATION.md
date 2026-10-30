# Validation

Validated on Linux x86_64, Python 3.12, MuJoCo 3.3.7, EGL/Mesa software rendering. No CUDA GPU is exposed by this workspace.

- `pytest -q`: **30 passed** (167.13 seconds). Includes the official dataset bridge exercised with CPU PyTorch 2.6.0+cpu.
- Generated a full 24-second blackout sequence at 480×360: **721 camera frames, 2,401 raw IMU samples**.
- Verified all **120 frames** in the 8–12 second interval contain no RGB signal and no valid depth.
- Loaded the full sequence through the official-code bridge: 721 frames, 2,400 inter-frame IMU samples grouped 3/3/4 at 30 Hz camera / 100 Hz IMU. The initial t=0 IMU sample has no preceding integration interval.
- Exported studio XML and all six textures reload independently and reproduce the original RGB/depth frames. The original checkerboard dataset also reloads with identical RGB.
- Furniture clearance is checked along the complete camera trajectory.
- The full studio sequence has byte-identical raw IMU and camera ground-truth files to the original sequence; changing the scene preserves the prescribed motion and inertial measurements.
- Exported studio XML and all six textures reload independently and reproduce the original RGB/depth frames. The original checkerboard dataset also reloads with identical RGB.
- Furniture clearance is checked along the complete camera trajectory.
- The full studio sequence has byte-identical raw IMU and camera ground-truth files to the original sequence; changing the scene preserves the prescribed motion and inertial measurements.
- Physical accelerometer/gyro checks match analytic rigid-body kinematics to numerical precision. Rendered axial depth is checked independently using MuJoCo ray intersections and exported optical calibration.
- Tested official w2c/wxyz result conversion to c2w/xyzw and metric trajectory scoring against synthetic known poses.
- Complete official MM3DGS CUDA tracking/mapping was **not run**. Its native behavior on empty visual/depth observations may fail; this is deliberately exposed by the stress test.

The current scene version is `studio-v2`. Geometry and photometric constraints differ from the original checkerboard scene; compare estimators on the same version. All rendering checks and the preview video use CPU/EGL, without CUDA.

The current scene version is `studio-v2`. Geometry and photometric constraints differ from the original checkerboard scene; compare estimators on the same version. All rendering checks and the preview video use CPU/EGL, without CUDA.

## Sensor sanity result, seed 7

The dead-reckoning check uses a ground-truth boundary pose and velocity, and biases estimated from the first 1.5 stationary seconds. It is not a SLAM or tight-coupling result.

| Predictor | Error at 12 s | RMSE during 8–12 s |
| --- | --- | --- |
| Noisy IMU propagation | 0.05105 m | 0.02199 m |
| Constant velocity | 4.43730 m | 2.09482 m |

The blackout demonstrates lost visual observability and the utility of inertial information. Necessity or superiority of tight fusion over loosely coupled fusion requires an actual estimator comparison.
