"""Timestamped streaming camera and IMU API, with isolated ground truth."""
from dataclasses import dataclass, asdict
import os
# CPU-only Linux machines can select osmesa before import instead.
if os.name == 'posix' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/mm3dgs-mesa-cache')
import cv2
import mujoco
import numpy as np
from .scene import build_scene, build_assets, SCENE_VERSION, R_B_C, P_B_C
from .trajectory import state
from .effects import environmental_effect

SCENARIOS = ('rich', 'low_texture', 'blackout', 'motion_blur', 'dust', 'rain')


@dataclass
class Config:
    scenario: str = 'blackout'
    duration: float = 24.0
    width: int = 640
    height: int = 480
    fovy: float = 60.0
    physics_hz: int = 600
    camera_hz: int = 30
    imu_hz: int = 100
    seed: int = 7
    blackout_start: float = 8.0
    blackout_end: float = 12.0
    depth_mode: str = 'gated'  # gated, off, clean (explicit oracle-depth ablation)
    accel_noise_density: float = 0.002  # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 0.0002  # rad/s / sqrt(Hz)
    accel_bias_rw: float = 0.0003  # m/s^3 / sqrt(Hz)
    gyro_bias_rw: float = 0.00002  # rad/s^2 / sqrt(Hz)
    accel_bias: tuple = (0.025, -0.015, 0.02)
    gyro_bias: tuple = (0.001, -0.0006, 0.0008)
    depth_noise: float = 0.005  # constant axial depth sigma, metres
    blur_exposure_s: float = 0.030
    blur_samples: int = 5
    blur_gain: float = 0.60
    blur_pixel_noise: float = 1.5
    blur_depth_keep: float = 0.60

    def __post_init__(self):
        if self.scenario not in SCENARIOS or self.depth_mode not in ('gated', 'off', 'clean'):
            raise ValueError('Unknown scenario or depth mode')
        if self.duration <= 4 or self.width < 32 or self.height < 32 or not 1 < self.fovy < 170:
            raise ValueError('Require duration > 4 s, dimensions >= 32, and 1 < fovy < 170')
        rates = (self.physics_hz, self.camera_hz, self.imu_hz)
        if any(not isinstance(r, int) or r <= 0 for r in rates) or any(self.physics_hz % r for r in rates[1:]):
            raise ValueError('Positive integer rates; physics_hz must divide into integer sensor strides')
        if not 0 <= self.blackout_start < self.blackout_end:
            raise ValueError('Invalid blackout interval')
        if self.scenario in ('blackout', 'motion_blur', 'dust', 'rain') and not 2 < self.blackout_start < self.blackout_end < self.duration-2:
            raise ValueError('Degradation must follow initialization and precede final stationary period')
        if not np.isfinite(self.blur_exposure_s) or self.blur_exposure_s <= 0 or (self.scenario == 'motion_blur' and self.blur_exposure_s > 1/self.camera_hz):
            raise ValueError('Blur exposure must be positive and no longer than one camera period')
        if not isinstance(self.blur_samples, int) or self.blur_samples < 2:
            raise ValueError('Blur requires at least two exposure samples')
        if not 0 < self.blur_gain <= 1 or not 0 < self.blur_depth_keep <= 1:
            raise ValueError('Blur gain and depth keep fraction must be in (0, 1]')
        noises = (self.accel_noise_density, self.gyro_noise_density, self.accel_bias_rw, self.gyro_bias_rw, self.depth_noise, self.blur_pixel_noise)
        if any(not np.isfinite(v) or v < 0 for v in noises):
            raise ValueError('Noise parameters must be finite and nonnegative')


class SLAMEnvironment:
    """Prescribed sensor trajectory, not a robot locomotion/control environment.

    next_tick() returns only public measurements. Truth is accessed explicitly
    with ground_truth(), and must never be passed to the estimator.
    """
    def __init__(self, config=None, render=True, scene_xml=None, assets=None):
        self.cfg = config or Config()
        c = self.cfg
        self.xml = scene_xml if scene_xml is not None else build_scene(c.width, c.height, c.fovy, c.physics_hz, False)
        self.assets = dict(assets or {}) if scene_xml is not None else dict(build_assets())
        self.scene_version = 'custom' if scene_xml is not None else SCENE_VERSION
        self.model = mujoco.MjModel.from_xml_string(self.xml, assets=self.assets)
        self.data = mujoco.MjData(self.model)
        self._rich_materials = self.model.geom_matid.copy()
        self._rich_colors = self.model.geom_rgba.copy()
        self.renderer = mujoco.Renderer(self.model, height=c.height, width=c.width) if render else None
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        # Separate streams ensure disabling depth does not change IMU noise.
        self.camera_rng = np.random.default_rng(self.cfg.seed+1000)
        self.image_rng = np.random.default_rng(self.cfg.seed+2000)
        self.degradation_rng = np.random.default_rng(self.cfg.seed+3000)
        self.ba = np.array(self.cfg.accel_bias, dtype=float)
        self.bg = np.array(self.cfg.gyro_bias, dtype=float)
        self.tick = -1
        self._set_state(0)

    def calibration(self, include_generation=False):
        c = self.cfg
        fy = c.height/(2*np.tan(np.deg2rad(c.fovy)/2))
        T = np.eye(4); T[:3, :3] = R_B_C; T[:3, 3] = P_B_C
        result = {
            'schema_version': 1, 'scene_version': self.scene_version,
            'camera': {'width': c.width, 'height': c.height, 'rate_hz': c.camera_hz,
                       'K': [[fy, 0, (c.width-1)/2], [0, fy, (c.height-1)/2], [0, 0, 1]],
                       'distortion': [0, 0, 0, 0, 0], 'model': 'pinhole',
                       'shutter': (f'global; trailing exposure up to {c.blur_exposure_s:g} s; timestamp at exposure end'
                                   if c.scenario == 'motion_blur' else 'global; instantaneous exposure'),
                       'depth': 'optical z, metres; zero invalid'},
            'imu': {'rate_hz': c.imu_hz,
                    'noise': {k: v for k, v in asdict(c).items() if k in ('accel_noise_density','gyro_noise_density','accel_bias_rw','gyro_bias_rw')}, 'accelerometer': 'specific force R_W_B.T @ (a_W - g_W), m/s^2',
                    'gyroscope': 'body angular velocity, rad/s',
                    'noise_discretization': 'sample sigma = density * sqrt(rate_hz); bias increment sigma = random_walk / sqrt(rate_hz)'},
            'T_B_C': T.tolist(), 'T_B_I': np.eye(4).tolist(), 'gravity_W': [0, 0, -9.81],
            'conventions': 'T_A_B maps B coordinates into A; W z up; B=I x forward/y left/z up; C optical x right/y down/z forward; quaternions xyzw',
            'timestamps': 'integer nanoseconds since simulation start; inclusive endpoints; IMU before camera at equal time',
        }

        if include_generation:
            result['config'] = asdict(c)
        return result

    def _set_state(self, t):
        p, rot, v, acc, omega, alpha = state(t, self.cfg.duration)
        q = rot.as_quat()
        self.data.qpos[:] = np.r_[p, q[[3, 0, 1, 2]]]
        self.data.qvel[:] = np.r_[v, omega]
        self.data.qacc[:] = np.r_[acc, alpha]
        self.data.time = t
        # Inverse dynamics supplies exactly the wrench for prescribed acceleration.
        # Forward dynamics then computes actual MuJoCo accelerometer/gyro outputs.
        mujoco.mj_inverse(self.model, self.data)
        self.data.qfrc_applied[:] = self.data.qfrc_inverse
        mujoco.mj_forward(self.model, self.data)

    def ground_truth(self):
        t = max(self.tick, 0)/self.cfg.physics_hz
        p, rot, v, acc, omega, _ = state(t, self.cfg.duration)
        T = np.eye(4); T[:3, :3] = rot.as_matrix(); T[:3, 3] = p
        Tbc = np.array(self.calibration()['T_B_C'])
        return {'timestamp_ns': round(t*1e9), 'T_W_B': T, 'T_W_C': T @ Tbc,
                'velocity_W': v, 'acceleration_W': acc, 'omega_B': omega,
                'accel_bias': self.ba.copy(), 'gyro_bias': self.bg.copy()}

    def _camera(self, t):
        if self.renderer is None:
            raise RuntimeError('Camera requested without renderer')
        c = self.cfg
        bad = c.blackout_start <= t < c.blackout_end
        if c.scenario == 'low_texture' and bad:
            self.model.geom_matid[:] = self.model.mat('plain').id
            self.model.geom_rgba[:] = [.65, .65, .65, 1.]
        else:
            self.model.geom_matid[:] = self._rich_materials
            self.model.geom_rgba[:] = self._rich_colors
        self.renderer.update_scene(self.data, camera='rgbd')
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        valid = (depth > .1) & (depth < 20)
        if c.depth_noise:
            depth += self.camera_rng.normal(0, c.depth_noise, depth.shape).astype(np.float32)
        depth[~valid] = 0
        bad = c.blackout_start <= t < c.blackout_end
        if c.scenario == 'blackout' and bad:
            rgb[:] = 0
        elif c.scenario == 'low_texture' and bad:
            rgb = cv2.GaussianBlur(rgb, (21, 21), 5)
            rgb = np.clip(rgb.astype(float)*.16 + self.image_rng.normal(0, 2, rgb.shape), 0, 255).astype(np.uint8)
        elif c.scenario in ('dust', 'rain') and bad:
            ramp = min(.3, (c.blackout_end-c.blackout_start)/4)
            strength = np.clip(min(t-c.blackout_start, c.blackout_end-t)/ramp, 0, 1)
            if strength:
                rgb, invalid = environmental_effect(rgb, depth, c.scenario, strength, t, c.seed)
                if c.depth_mode == 'gated':
                    depth[invalid] = 0
        elif c.scenario == 'motion_blur' and bad:
            # Walking/turning through a dim patch: longer exposure, reduced SNR,
            # and partial stereo confidence loss. Textures and geometry stay put.
            ramp = min(.3, (c.blackout_end-c.blackout_start)/4)
            strength = np.clip(min(t-c.blackout_start, c.blackout_end-t)/ramp, 0, 1)
            if strength:
                exposure = c.blur_exposure_s*strength
                # Integrate actual camera motion over PAST poses. Averaging in
                # display RGB is an approximation to a radiometric sensor model.
                images = [rgb.astype(np.float32)*.5]
                try:
                    for i, offset in enumerate(np.linspace(exposure, 0, c.blur_samples)[:-1]):
                        self._set_state(max(0, t-offset))
                        self.renderer.update_scene(self.data, camera='rgbd')
                        weight = .5 if i == 0 else 1.
                        images.append(self.renderer.render().astype(np.float32)*weight)
                finally:
                    self._set_state(t)
                integrated = np.sum(images, axis=0)/(c.blur_samples-1)
                gain = 1-(1-c.blur_gain)*strength
                noise = self.degradation_rng.normal(0, c.blur_pixel_noise*strength, rgb.shape)
                rgb = np.clip(integrated*gain+noise, 0, 255).astype(np.uint8)
                if c.depth_mode == 'gated':
                    # Block confidence dropout is a simple passive-stereo proxy;
                    # clean-depth mode tests the independent RGB-D sensor case.
                    keep = 1-(1-c.blur_depth_keep)*strength
                    blocks = self.degradation_rng.random(((c.height+7)//8, (c.width+7)//8)) < keep
                    mask = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:c.height, :c.width]
                    depth[~mask] = 0
        if c.depth_mode == 'off' or (c.depth_mode == 'gated' and bad and c.scenario in ('blackout', 'low_texture')):
            depth[:] = 0
        return {'timestamp_ns': round(t*1e9), 'rgb': rgb, 'depth': depth}

    def next_tick(self):
        self.tick += 1
        if self.tick > round(self.cfg.duration*self.cfg.physics_hz):
            raise StopIteration
        c = self.cfg
        t = self.tick/c.physics_hz
        self._set_state(t)
        result = {}
        if self.tick % (c.physics_hz//c.imu_hz) == 0:
            dt = 1/c.imu_hz
            if self.tick:
                self.ba += self.rng.normal(0, c.accel_bias_rw*np.sqrt(dt), 3)
                self.bg += self.rng.normal(0, c.gyro_bias_rw*np.sqrt(dt), 3)
            result['imu'] = {
                'timestamp_ns': round(t*1e9),
                'accel': self.data.sensor('accel').data.copy()+self.ba+self.rng.normal(0, c.accel_noise_density/np.sqrt(dt), 3),
                'gyro': self.data.sensor('gyro').data.copy()+self.bg+self.rng.normal(0, c.gyro_noise_density/np.sqrt(dt), 3)}
        if self.renderer is not None and self.tick % (c.physics_hz//c.camera_hz) == 0:
            result['camera'] = self._camera(t)
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
