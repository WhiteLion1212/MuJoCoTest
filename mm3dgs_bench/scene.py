"""Procedural studio interior and portable texture assets for the sensor benchmark."""
from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

SCENE_VERSION = 'studio-v2'
R_B_C = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
P_B_C = np.array([0.08, 0.0, 0.04])


def _png(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


@lru_cache(maxsize=1)
def build_assets():
    """Original deterministic textures; no downloads or external asset licenses."""
    rng = np.random.default_rng(123)
    y, x = np.mgrid[:768, :768]
    grain = 2*np.sin(x*.16 + 2*np.sin(y*.009)) + rng.normal(0, 1.4, x.shape)
    wood = np.empty((768,768,3), dtype=np.uint8)
    for i in range(12):
        strip = slice(i*64,(i+1)*64)
        tone = np.array([186,150,108]) + rng.uniform(-11,11)
        wood[strip] = np.clip(tone + grain[strip,...,None],0,255)
        wood[i*64:i*64+2] = [105,81,57]
        for joint in range(int(rng.integers(60,250)),768,256):
            wood[strip,joint:joint+2] = [112,86,60]
    assets = {'assets/oak.png': _png(Image.fromarray(wood))}
    textile = np.clip(np.array([75,119,115]) + rng.normal(0,3,(256,256,1)),0,255).astype(np.uint8)
    textile[::3] = np.clip(textile[::3].astype(int)+4,0,255)
    assets['assets/fabric.png'] = _png(Image.fromarray(textile))
    palette = [(222,149,96),(62,110,113),(197,179,145)]
    for i, accent in enumerate(palette):
        image = Image.new('RGB',(384,512),(237,230,213))
        d = ImageDraw.Draw(image)
        d.rectangle((28,28,356,484),outline=(77,80,73),width=2)
        d.ellipse((55+i*17,65,300+i*8,310),fill=accent)
        d.rectangle((70,225-i*12,312,405),fill=(39+i*12,69+i*8,74+i*8))
        for j in range(7):
            d.arc((48+j*13,174+j*7,330-j*8,450-j*7),0,175,fill=(235,218,181),width=3)
        d.rectangle((50,448,155,453),fill=(72,80,75))
        d.rectangle((50,462,225,465),fill=(150,150,135))
        assets[f'assets/art_{i}.png'] = _png(image)
    image = Image.new('RGB',(512,384))
    sky = np.zeros((384,512,3),np.uint8)
    for row in range(384):
        sky[row] = np.array([176,207,218])*(1-row/1200)+np.array([238,233,210])*(row/1200)
    image = Image.fromarray(sky)
    d = ImageDraw.Draw(image)
    d.ellipse((350,45,402,97),fill=(249,237,190))
    d.polygon([(0,310),(90,170),(190,300),(295,205),(420,305),(512,235),(512,384),(0,384)],fill=(125,156,151))
    d.polygon([(0,360),(120,270),(235,360),(385,275),(512,345),(512,384),(0,384)],fill=(98,134,128))
    assets['assets/window.png'] = _png(image)
    return assets


def build_scene(width, height, fovy, physics_hz, low_texture=False):
    rng = np.random.default_rng(321)
    gl_rot = R_B_C @ np.diag([1., -1., -1.])
    q = Rotation.from_matrix(gl_rot).as_quat()
    camera_quat = ' '.join(map(str, q[[3, 0, 1, 2]]))
    objects = []

    def geom(kind, pos, size, material, **attrs):
        material = 'plain' if low_texture else material
        params = {'type':kind, 'pos':' '.join(map(str,pos)), 'size':' '.join(map(str,size)), 'material':material, **attrs}
        objects.append('<geom '+ ' '.join(f'{k}="{v}"' for k,v in params.items()) + '/>')

    def box(pos, size, material, **attrs):
        geom('box',pos,size,material,**attrs)

    # A closed, comfortably lit room replaces the high-contrast checker enclosure.
    box((0,0,-.1),(7,6,.1),'oak',name='floor')
    box((0,0,3.5),(7,6,.1),'ceiling',name='ceiling')
    for x in (-7.1,7.1):
        box((x,0,1.7),(.1,6,1.7),'plaster')
        box((x*.985,0,.10),(.025,6,.10),'trim')
    for y in (-6.1,6.1):
        box((0,y,1.7),(7,.1,1.7),'plaster')
        box((0,y*.985,.10),(7,.025,.10),'trim')
    # Accent wall, three framed prints, and vertical timber battens.
    box((6.985,.3,1.7),(.012,3.15,1.7),'accent')
    for i, y in enumerate((-1.5,.15,1.8)):
        box((6.94,y,1.95),(.035,.64,.84),'frame')
        geom('box',(6.900,y,1.95),(.59,.79,.001),f'art{i}',xyaxes='0 -1 0 0 0 1')
    for y in np.arange(3.65,5.91,.105):
        box((6.92,y,1.7),(.055,.023,1.7),'oak')
    # Large north-wall windows; opaque illustrated glazing keeps depth defined.
    for x in (-4.9,-2.15,.6,3.35):
        box((x,5.93,2.05),(1.20,.055,.95),'trim')
        geom('box',(x,5.865,2.05),(1.13,.88,.001),'window',xyaxes='1 0 0 0 0 1')
        box((x,5.84,2.05),(.018,.025,.9),'trim')
        box((x,5.84,2.05),(1.16,.025,.018),'trim')
        box((x,5.78,1.08),(1.29,.18,.045),'oak')
    # Work table, chairs, monitor, books and ceramic objects on the east side.
    box((5.10,.1,.78),(.70,1.65,.055),'oak')
    for x in (4.55,5.65):
        for y in (-1.35,1.55):
            box((x,y,.36),(.035,.035,.36),'metal')
    box((5.48,.40,1.17),(.035,.39,.26),'frame')
    box((5.439,.40,1.17),(.007,.35,.22),'screen')
    box((5.48,.40,.91),(.028,.045,.10),'metal')
    box((5.42,.40,.84),(.17,.25,.015),'metal')
    for j in range(5):
        box((5.429,.17+j*.08,1.14+j*.016),(.004,.023,.07+j*.01),'screen_line')
    for i in range(3):
        box((4.97,-.86,.85+i*.035),(.23,.31,.016),('terracotta','linen','accent')[i])
    geom('cylinder',(4.70,1.04,.94),(.08,.11),'ceramic')
    for y in (-.95,1.1):
        box((3.95,y,.46),(.32,.34,.065),'fabric')
        box((3.66,y,.77),(.065,.34,.28),'fabric')
        for x in (3.72,4.18):
            for yy in (y-.24,y+.24):
                box((x,yy,.22),(.022,.022,.22),'metal')
    # Bookcase with non-repeating colored spines and a few open compartments.
    for z in (.16,.70,1.24,1.78,2.32):
        box((6.45,-4.3,z),(.43,1.0,.035),'oak')
    for y in (-5.30,-3.30):
        box((6.45,y,1.24),(.43,.035,1.13),'oak')
    for level in range(4):
        y = -5.17
        for i in range(9 if level%2 else 12):
            w = rng.uniform(.035,.065)
            h = rng.uniform(.25,.42)
            material = rng.choice(['linen','terracotta','accent','book_blue','ochre'])
            box((6.23,y, .20+level*.54+h/2),(.22,w,h/2),material)
            box((5.999,y,.26+level*.54),(.004,w*.7,.008),'paper')
            y += 2*w+.025
    # Lounge area and low coffee table leave the central sensor loop unobstructed.
    box((2.5,-4.85,.34),(1.55,.53,.25),'fabric')
    box((2.5,-5.28,.80),(1.55,.13,.43),'fabric')
    for x in (1.02,3.98):
        box((x,-4.80,.65),(.15,.59,.30),'fabric')
    for x in (1.55,2.5,3.45):
        box((x,-4.75,.62),(.43,.45,.08),'fabric')
    box((2.5,-3.5,.40),(.95,.45,.045),'oak')
    for x in (1.75,3.25):
        for y in (-3.8,-3.2):
            box((x,y,.18),(.025,.025,.18),'metal')
    box((2.30,-3.48,.47),(.23,.28,.024),'linen')
    # A subtle woven rug under the work area gives scale without checker aliases.
    box((4.8,.1,.008),(1.7,2.05,.008),'rug')
    for y in np.arange(-1.90,2.10,.17):
        box((4.8,y,.017),(1.60,.008,.001),'rug_line')
    # Pots, tapered stems, branches, and clusters of individually oriented leaves.
    for px,py in ((5.90,3.08),(5.55,-2.65),(.30,-5.40),(-3.8,5.35)):
        geom('cylinder',(px,py,.22),(.25,.22),'terracotta')
        geom('cylinder',(px,py,.445),(.22,.015),'soil')
        for i in range(12):
            angle = i*2.399
            z = .65+(i%5)*.17
            dx,dy = .32*np.cos(angle),.32*np.sin(angle)
            objects.append(f'<geom type="capsule" fromto="{px} {py} .45 {px+dx} {py+dy} {z}" size=".014" material="leaf_dark"/>')
            geom('ellipsoid',(px+dx,py+dy,z),(.11,.055,.25),'leaf' if i%2 else 'leaf_dark',euler=f'.3 .7 {angle}')
    # Warm ceiling fixtures, pendant rods, and the corresponding real light sources.
    for y in (-2.5,2.5):
        box((3.4,y,3.30),(1.4,.16,.045),'metal')
        box((3.4,y,3.248),(1.34,.12,.008),'lamp')
    for x in (-3.0,1.0):
        box((x,0,3.38),(.035,5.5,.025),'metal')
    textures = '\n'.join(f'<texture name="{name}" type="2d" file="assets/{filename}.png"/>' for name,filename in
                         [('oak_tex','oak'),('fabric_tex','fabric'),('window_tex','window'),*[(f'art{i}_tex',f'art_{i}') for i in range(3)]])
    return f'''<mujoco model="mm3dgs_studio_v2">
      <compiler angle="radian"/>
      <option timestep="{1/physics_hz}" gravity="0 0 -9.81"/>
      <visual>
        <global offwidth="{width}" offheight="{height}"/>
        <map znear=".01" zfar="50"/>
        <quality shadowsize="2048" offsamples="4"/>
        <headlight ambient=".20 .21 .23" diffuse=".30 .30 .30" specular=".04 .04 .04"/>
      </visual>
      <asset>
        {textures}
        <material name="plain" rgba=".65 .65 .65 1" specular="0"/>
        <material name="oak" texture="oak_tex" texrepeat="2 2" texuniform="true" specular=".12" shininess=".15"/>
        <material name="fabric" texture="fabric_tex" texrepeat="3 3" texuniform="true" specular=".02"/>
        <material name="plaster" rgba=".79 .77 .70 1" specular=".02"/>
        <material name="ceiling" rgba=".89 .88 .82 1" specular="0"/>
        <material name="trim" rgba=".88 .87 .81 1" specular=".10"/>
        <material name="accent" rgba=".15 .28 .29 1" specular=".06"/>
        <material name="frame" rgba=".09 .105 .10 1" specular=".14"/>
        <material name="metal" rgba=".16 .18 .18 1" specular=".45" shininess=".45"/>
        <material name="ceramic" rgba=".84 .80 .68 1" specular=".35" shininess=".3"/>
        <material name="linen" rgba=".70 .65 .53 1" specular=".02"/>
        <material name="terracotta" rgba=".62 .29 .18 1" specular=".04"/>
        <material name="ochre" rgba=".72 .50 .21 1" specular=".02"/>
        <material name="book_blue" rgba=".24 .36 .47 1" specular=".02"/>
        <material name="paper" rgba=".88 .83 .70 1" specular="0"/>
        <material name="rug" rgba=".59 .56 .46 1" specular="0"/>
        <material name="rug_line" rgba=".48 .46 .39 1" specular="0"/>
        <material name="leaf" rgba=".18 .35 .18 1" specular=".18"/>
        <material name="leaf_dark" rgba=".10 .24 .13 1" specular=".12"/>
        <material name="soil" rgba=".13 .10 .07 1" specular="0"/>
        <material name="screen" rgba=".13 .23 .29 1" emission=".15"/>
        <material name="screen_line" rgba=".50 .73 .72 1" emission=".2"/>
        <material name="lamp" rgba="1 .89 .66 1" emission=".85"/>
        <material name="window" texture="window_tex" emission=".18" specular=".05"/>
        {''.join(f'<material name="art{i}" texture="art{i}_tex" specular="0"/>' for i in range(3))}
      </asset>
      <default><geom friction=".8 .1 .1"/></default>
      <worldbody>
        <light pos="-2 2 3.2" dir=".4 -.3 -1" directional="true" diffuse=".72 .68 .60" ambient=".32 .32 .33" specular=".15 .15 .15" castshadow="true"/>
        <light pos="5 -3 3.1" dir="-.3 .3 -1" diffuse=".35 .31 .24" attenuation="1 .04 .01" castshadow="false"/>
        <light pos="-4 0 2.0" dir="1 0 .15" directional="true" diffuse=".42 .43 .40" castshadow="false"/>
        <light pos="0 0 1.0" dir="0 0 1" directional="true" diffuse=".40 .40 .38" castshadow="false"/>
        <light pos="1 5 2.8" dir=".1 -1 -.4" diffuse=".32 .40 .46" castshadow="false"/>
        {''.join(objects)}
        <body name="rig" pos="0 0 1.2"><freejoint/>
          <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <site name="imu" pos="0 0 0" size=".01"/>
          <camera name="rgbd" pos=".08 0 .04" quat="{camera_quat}" fovy="{fovy}"/>
        </body>
      </worldbody>
      <sensor><accelerometer name="accel" site="imu"/><gyro name="gyro" site="imu"/></sensor>
    </mujoco>'''
