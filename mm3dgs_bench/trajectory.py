"""Smooth prescribed 6-DoF motion with exact translational derivatives."""
import numpy as np
from scipy.spatial.transform import Rotation


def _angles(t, duration):
    span = duration - 4.0
    u = np.clip((t - 2.0) / span, 0.0, 1.0)
    s = 10*u**3 - 15*u**4 + 6*u**5
    sd = (30*u**2 - 60*u**3 + 30*u**4) / span
    sdd = (60*u - 180*u**2 + 120*u**3) / span**2
    a, ad, add = 2*np.pi*s, 2*np.pi*sd, 2*np.pi*sdd
    # Yaw turns and translational accelerations continue through the blackout.
    e = np.array([0.07*np.sin(3*a), 0.12*np.sin(2*a), 0.7*np.sin(a)+0.25*np.sin(3*a)])
    ed = np.array([0.21*np.cos(3*a), 0.24*np.cos(2*a), 0.7*np.cos(a)+0.75*np.cos(3*a)])*ad
    return a, ad, add, e, ed


def angular_velocity(t, duration):
    _, _, _, (r, p, _), (rd, pd, yd) = _angles(t, duration)
    return np.array([rd-yd*np.sin(p), pd*np.cos(r)+yd*np.sin(r)*np.cos(p),
                     -pd*np.sin(r)+yd*np.cos(r)*np.cos(p)])


def state(t, duration):
    a, ad, add, e, _ = _angles(t, duration)
    p = np.array([2.5*np.sin(a), 1.3*np.sin(2*a), 1.2+0.15*np.sin(3*a)])
    pa = np.array([2.5*np.cos(a), 2.6*np.cos(2*a), 0.45*np.cos(3*a)])
    paa = np.array([-2.5*np.sin(a), -5.2*np.sin(2*a), -1.35*np.sin(3*a)])
    v, acc = pa*ad, paa*ad**2+pa*add
    rot = Rotation.from_euler('xyz', e)
    omega = angular_velocity(t, duration)
    h = 1e-5
    alpha = (angular_velocity(t+h, duration)-angular_velocity(t-h, duration))/(2*h)
    return p, rot, v, acc, omega, alpha
