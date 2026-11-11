"""CPU RGB-D odometry reference with optional inertial optimization residuals.

This is a local tracking/prediction baseline, not a replacement for the target
SLAM system: no loop closure, global bundle adjustment, or online bias factors.
The first second must be stationary for observable gyro/gravity initialization.
No ground truth, planted biases, scenario timing, or future samples are used.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class Estimator:
    use_imu_residuals = True

    def reset(self, calibration):
        cv2.setRNGSeed(0)
        cv2.setNumThreads(1)
        self.K = np.array(calibration['camera']['K'])
        self.Tbc = np.array(calibration['T_B_C'])
        self.Tcb = np.linalg.inv(self.Tbc)
        self.imu = None
        self.imu_origin = None
        self.warmup_accel, self.warmup_gyro = [], []
        self.bg = np.zeros(3)
        self.gravity = self.Tcb[:3, :3] @ np.array([0., 0., -9.81])
        self.body = None
        self.last_pose = self.last_time = None
        self.camera_velocity = np.zeros(3)
        self.camera_omega = np.zeros(3)
        self.gray = self.points = self.landmarks = None
        self.status = {'visual_update': False, 'inliers': 0}

    def _propagate(self, body, timestamp_ns, sample, previous=None):
        p, r, v, ts = body
        dt = (timestamp_ns-ts)/1e9
        if dt < 0:
            raise ValueError('Noncausal IMU/camera delivery')
        f, w = sample['accel'], sample['gyro']-self.bg
        if previous is not None:
            f = (f+previous['accel'])/2
            w = (w+previous['gyro']-self.bg)/2
        mid = r*Rotation.from_rotvec(w*dt/2)
        a = mid.apply(f)+self.gravity
        return p+v*dt+.5*a*dt**2, r*Rotation.from_rotvec(w*dt), v+a*dt, timestamp_ns

    def process_imu(self, sample):
        ts = sample['timestamp_ns']
        if self.imu is not None and ts <= self.imu['timestamp_ns']:
            raise ValueError('IMU samples must be strictly ordered')
        if self.imu_origin is None:
            self.imu_origin = ts
        if ts-self.imu_origin <= 1_000_000_000:
            self.warmup_accel.append(sample['accel'])
            self.warmup_gyro.append(sample['gyro'])
            mean_accel = np.mean(self.warmup_accel, axis=0)
            self.bg = np.mean(self.warmup_gyro, axis=0)
            # Gravity/tilt and transverse accelerometer bias are not separately
            # observable at rest; do not silently use the planted bias values.
            self.gravity = self.Tcb[:3, :3] @ (-9.81*mean_accel/np.linalg.norm(mean_accel))
        if self.body is not None:
            self.body = self._propagate(self.body, ts, sample, self.imu)
        self.imu = sample

    def _predicted_body(self, timestamp_ns):
        if self.body is None:
            return None
        if self.imu is None:
            pose = self._visual_prediction(timestamp_ns)
            T = pose @ self.Tcb
            return T[:3, 3], Rotation.from_matrix(T[:3, :3]), self.body[2], timestamp_ns
        return self._propagate(self.body, timestamp_ns, self.imu)

    def _visual_prediction(self, timestamp_ns):
        dt = (timestamp_ns-self.last_time)/1e9
        if dt < 0:
            raise ValueError('Cannot predict into the past')
        T = self.last_pose.copy()
        T[:3, 3] += self.camera_velocity*dt
        T[:3, :3] = (Rotation.from_matrix(T[:3, :3])*Rotation.from_rotvec(self.camera_omega*dt)).as_matrix()
        return T

    def predict_pose(self, timestamp_ns):
        """Pose BEFORE seeing the next camera; uses only IMU delivered so far."""
        if self.last_pose is None:
            return None
        if self.imu is None:
            return self._visual_prediction(timestamp_ns)
        p, r, _, _ = self._predicted_body(timestamp_ns)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = r.as_matrix(), p
        return T @ self.Tbc

    def tracking_status(self):
        return dict(self.status)

    def _correspondences(self, gray):
        if self.points is None or len(self.points) < 8:
            return np.empty((0, 3)), np.empty((0, 2))
        tracked, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None,
                                                winSize=(21, 21), maxLevel=3)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, tracked, None,
                                                 winSize=(21, 21), maxLevel=3)
        good = ok.ravel().astype(bool) & ok_back.ravel().astype(bool)
        good &= np.linalg.norm(back[:, 0]-self.points[:, 0], axis=1) < 1.
        return self.landmarks[good], tracked[good, 0].astype(float)

    def _optimize(self, predicted, world, pixels, dt, velocity):
        Rcw = predicted[:3, :3].T
        rvec = Rotation.from_matrix(Rcw).as_rotvec()
        tvec = -Rcw @ predicted[:3, 3]
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            world, pixels, self.K, None, rvec=rvec.copy(), tvec=tvec.copy(),
            useExtrinsicGuess=True, iterationsCount=100, reprojectionError=2.5,
            confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 8:
            return predicted, velocity, 0
        world, pixels = world[inliers[:, 0]], pixels[inliers[:, 0]]
        visual_R = Rotation.from_rotvec(rv.ravel()).inv()
        visual_p = -visual_R.apply(tv.ravel())
        base_R = Rotation.from_matrix(predicted[:3, :3])
        inertial = self.imu is not None and self.use_imu_residuals
        initial = np.r_[(base_R.inv()*visual_R).as_rotvec(), visual_p-predicted[:3, 3]]
        if inertial:
            initial = np.r_[initial, np.zeros(6)]

        def residual(x):
            r = base_R*Rotation.from_rotvec(x[:3])
            p = predicted[:3, 3]+x[3:6]
            camera_points = r.inv().apply(world-p)
            z = np.maximum(camera_points[:, 2], .05)
            uv = camera_points[:, :2]/z[:, None]
            uv = uv*self.K[[0, 1], [0, 1]]+self.K[:2, 2]
            visual = (uv-pixels).ravel()
            # Huber pixel residuals; inertial factors keep their quadratic form.
            visual = np.sign(visual)*np.sqrt(np.where(abs(visual) <= 1.5,
                                           visual**2, 3*abs(visual)-2.25))
            if not inertial:
                return visual
            # A two-state preintegration factor with fixed preceding pose and
            # jointly optimized previous/current velocities. The velocity prior
            # carries information from the preceding tracking update.
            dv_previous, dv_current = x[6:9], x[9:12]
            lever = r.apply(self.Tcb[:3, 3])-base_R.apply(self.Tcb[:3, 3])
            position = (x[3:6]+lever-dv_previous*dt)/(.001+.002*dt)
            rotation = x[:3]/(.0008+.003*dt)
            speed = (dv_current-dv_previous)/(.005+.02*dt)
            return np.r_[visual, position, rotation, speed, dv_previous/.06]

        solution = least_squares(residual, initial, max_nfev=20, ftol=1e-5, xtol=1e-5)
        x = solution.x
        T = predicted.copy()
        T[:3, :3] = (base_R*Rotation.from_rotvec(x[:3])).as_matrix()
        T[:3, 3] += x[3:6]
        return T, velocity+(x[9:12] if inertial else 0), len(inliers)

    def _new_landmarks(self, gray, depth, pose, world=None, pixels=None):
        # Keep tracked landmarks in their existing world frame. Re-lifting all
        # pixels from noisy depth at every image needlessly accumulates drift.
        kept_world, kept_pixels = np.empty((0, 3)), np.empty((0, 2))
        mask = (depth > .1).astype(np.uint8)*255
        if world is not None and len(world):
            pc = (world-pose[:3, 3]) @ pose[:3, :3]
            uv = pc[:, :2]/np.maximum(pc[:, 2:3], .05)
            uv = uv*self.K[[0, 1], [0, 1]]+self.K[:2, 2]
            good = (pc[:, 2] > .1) & (np.linalg.norm(uv-pixels, axis=1) < 2.5)
            kept_world, kept_pixels = world[good][:350], pixels[good][:350]
            for u, v in np.rint(kept_pixels).astype(int):
                cv2.circle(mask, (u, v), 6, 0, -1)
        count = 350-len(kept_world)
        corners = (cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=.008,
                                         minDistance=5, blockSize=5, mask=mask) if count else None)
        if corners is None:
            self.points = kept_pixels.astype(np.float32).reshape(-1, 1, 2)
            self.landmarks = kept_world
            return
        uv = corners[:, 0]
        ij = np.rint(uv).astype(int)
        z = depth[ij[:, 1], ij[:, 0]]
        good = (z > .1) & np.isfinite(z)
        self.points = np.concatenate([kept_pixels[:, None].astype(np.float32), corners[good]])
        points = np.c_[uv[good], np.ones(good.sum())] @ np.linalg.inv(self.K).T
        points *= z[good, None]
        self.landmarks = np.concatenate([kept_world, points @ pose[:3, :3].T+pose[:3, 3]])

    def process_frame(self, frame):
        ts = frame['timestamp_ns']
        gray = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2GRAY)
        retained_world = retained_pixels = None
        if self.last_pose is None:
            pose = np.eye(4)
            T = pose @ self.Tcb
            self.body = (T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]), np.zeros(3), ts)
            self.status = {'visual_update': True, 'inliers': 0}
        else:
            if ts <= self.last_time:
                raise ValueError('Camera frames must be strictly ordered')
            dt = (ts-self.last_time)/1e9
            predicted = self.predict_pose(ts)
            body = self._predicted_body(ts)
            world, pixels = self._correspondences(gray)
            pose, velocity, count = (self._optimize(predicted, world, pixels, dt, body[2])
                                     if len(world) >= 8 else (predicted, body[2], 0))
            self.status = {'visual_update': count >= 8, 'inliers': count}
            if count >= 8:
                retained_world, retained_pixels = world, pixels
            self.camera_velocity = (pose[:3, 3]-self.last_pose[:3, 3])/dt
            self.camera_omega = (Rotation.from_matrix(self.last_pose[:3, :3]).inv()*
                                 Rotation.from_matrix(pose[:3, :3])).as_rotvec()/dt
            T = pose @ self.Tcb
            if self.imu is None or not self.use_imu_residuals:
                velocity = (T[:3, 3]-self.body_at_frame[:3, 3])/dt
            self.body = (T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]), velocity, ts)
        self.body_at_frame = pose @ self.Tcb
        self._new_landmarks(gray, frame['depth'], pose, retained_world, retained_pixels)
        self.gray = gray
        self.last_pose, self.last_time = pose, ts
        # Propagated poses remain visible in ATE; tracking_status separately
        # exposes whether image constraints actually supported each update.
        return pose


class InitializationOnlyEstimator(Estimator):
    use_imu_residuals = False
