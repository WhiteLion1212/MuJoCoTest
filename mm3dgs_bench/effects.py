"""Image-space environmental sensor models; never change scene geometry or IMU."""
import cv2
import numpy as np


def environmental_effect(rgb, depth, scenario, strength, time_s, seed):
    """Return impaired RGB and an invalid-depth mask, preserving usable regions.

    Dust uses depth-dependent Beer-Lambert attenuation and drifting particles.
    Rain represents water on the lens after entering the building, rather than
    rainfall inside the studio. Droplets refract/blur only their covered pixels.
    These are controlled approximations, not fluid or optical simulations.
    """
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    rng = np.random.default_rng(seed+4000)
    invalid = np.zeros(depth.shape, bool)
    if scenario == 'dust':
        distance = np.where(depth > 0, depth, 20.)
        cloud = .8+.2*np.sin(xx/width*7+time_s*.6)*np.cos(yy/height*5)
        transmission = np.exp(-.12*strength*distance*cloud)
        airlight = np.array([195., 177., 148.])
        image = rgb*transmission[..., None]+airlight*(1-transmission[..., None])
        for x, y, radius, speed in rng.uniform(size=(35, 4)):
            u = ((x+time_s*(.025+.035*speed)) % 1)*width
            v = ((y+time_s*.012) % 1)*height
            sigma = max(1., (.002+.006*radius)*width)
            alpha = .35*strength*np.exp(-((xx-u)**2+(yy-v)**2)/(2*sigma**2))
            image = image*(1-alpha[..., None])+airlight*alpha[..., None]
            invalid |= alpha > .23
        # Passive stereo confidence drops first at heavily attenuated pixels.
        invalid |= transmission < .32
    elif scenario == 'rain':
        map_x, map_y = xx.copy(), yy.copy()
        alpha = np.zeros(depth.shape, np.float32)
        rim = np.zeros(depth.shape, np.float32)
        for x, y, radius in rng.uniform(size=(22, 3)):
            rx = width*(.018+.035*radius)
            ry = rx*(1.2+.6*radius)
            dx, dy = xx-x*width, yy-y*height
            rho = (dx/rx)**2+(dy/ry)**2
            inside = rho < 1
            bend = .45*(1-rho[inside])*strength
            map_x[inside] += dx[inside]*bend
            map_y[inside] += dy[inside]*bend
            alpha[inside] = strength
            rim = np.maximum(rim, np.exp(-((rho-.83)/.12)**2)*strength)
        refracted = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        refracted = cv2.GaussianBlur(refracted, (5, 5), 1.2)
        image = rgb*(1-alpha[..., None])+refracted*alpha[..., None]
        image = image*(1-.25*rim[..., None])+245.*.25*rim[..., None]
        invalid = alpha > .2
    else:
        raise ValueError(f'Unknown environmental effect: {scenario}')
    return np.clip(image, 0, 255).astype(np.uint8), invalid
