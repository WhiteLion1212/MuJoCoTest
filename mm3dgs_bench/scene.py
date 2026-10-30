"""All assets are procedural; rich and low-texture worlds share geometry."""
import numpy as np
from scipy.spatial.transform import Rotation

R_B_C = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
P_B_C = np.array([0.08, 0.0, 0.04])


def build_scene(width, height, fovy, physics_hz, low_texture=False):
    rng = np.random.default_rng(123)  # scene independent of sensor noise seed
    gl_rot = R_B_C @ np.diag([1., -1., -1.])
    q = Rotation.from_matrix(gl_rot).as_quat()
    camera_quat = ' '.join(map(str, q[[3, 0, 1, 2]]))
    material = 'plain' if low_texture else 'detail'
    objects = []
    # Room surfaces and distinct wall panels produce repeatable visual structure.
    for i in range(32):
        x = -6.4 + (i % 16)*0.84
        y = -5.85 if i < 16 else 5.85
        color = [0.6, 0.6, 0.6] if low_texture else rng.uniform(0.1, 0.95, 3)
        objects.append(f'<geom type="box" pos="{x} {y} 1.8" size=".3 .02 .5" material="{material}" rgba="{color[0]} {color[1]} {color[2]} 1"/>')
    # Perimeter clutter stays clear of the prescribed central trajectory.
    for i in range(28):
        x, y = rng.uniform(-6, 6), rng.choice([-1, 1])*rng.uniform(3.4, 5.1)
        z = rng.uniform(.2, .8)
        color = [0.65]*3 if low_texture else rng.uniform(.15, .9, 3)
        objects.append(f'<geom type="box" pos="{x} {y} {z}" size=".25 .3 {z}" material="{material}" rgba="{color[0]} {color[1]} {color[2]} 1"/>')
    return f'''<mujoco model="mm3dgs_sensor_room">
      <compiler angle="radian"/>
      <option timestep="{1/physics_hz}" gravity="0 0 -9.81"/>
      <visual><global offwidth="{width}" offheight="{height}"/><map znear=".01" zfar="50"/><quality shadowsize="2048"/></visual>
      <asset>
        <texture name="checks" type="2d" builtin="checker" rgb1=".15 .22 .32" rgb2=".88 .82 .66" width="512" height="512" mark="random" markrgb=".9 .1 .1" random=".2"/>
        <material name="detail" texture="checks" texrepeat="6 6" texuniform="true" reflectance="0"/>
        <material name="plain" rgba=".65 .65 .65 1" reflectance="0"/>
      </asset>
      <default><geom friction=".8 .1 .1"/></default>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse=".8 .8 .8" ambient=".3 .3 .3"/>
        <geom name="floor" type="box" pos="0 0 -.1" size="7 6 .1" material="{material}"/>
        <geom type="box" pos="7.1 0 1.5" size=".1 6 1.5" material="{material}"/>
        <geom type="box" pos="-7.1 0 1.5" size=".1 6 1.5" material="{material}"/>
        <geom type="box" pos="0 6.1 1.5" size="7 .1 1.5" material="{material}"/>
        <geom type="box" pos="0 -6.1 1.5" size="7 .1 1.5" material="{material}"/>
        {''.join(objects)}
        <body name="rig" pos="0 0 1.2"><freejoint/>
          <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <site name="imu" pos="0 0 0" size=".01"/>
          <camera name="rgbd" pos=".08 0 .04" quat="{camera_quat}" fovy="{fovy}"/>
        </body>
      </worldbody>
      <sensor><accelerometer name="accel" site="imu"/><gyro name="gyro" site="imu"/></sensor>
    </mujoco>'''
