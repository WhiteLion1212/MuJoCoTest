"""Physical and information-content checks, explicitly not SLAM results."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from .environment import Config, SLAMEnvironment
from .trajectory import state


def outage_sanity(config=None):
    c=config or Config()
    dt=1/c.imu_hz
    times=np.arange(round(c.duration*c.imu_hz)+1)*dt
    imu=[]
    with SLAMEnvironment(c,render=False) as env:
        for t in times:
            env.tick=round(t*c.physics_hz)
            env._set_state(t)
            if t:
                env.ba+=env.rng.normal(0,c.accel_bias_rw*np.sqrt(dt),3)
                env.bg+=env.rng.normal(0,c.gyro_bias_rw*np.sqrt(dt),3)
            imu.append((env.data.sensor('accel').data.copy()+env.ba+env.rng.normal(0,c.accel_noise_density/np.sqrt(dt),3),
                        env.data.sensor('gyro').data.copy()+env.bg+env.rng.normal(0,c.gyro_noise_density/np.sqrt(dt),3)))
    imu=np.array(imu)
    # Known initial orientation during stationary warm-up, not planted bias values.
    ba=imu[times<1.5,0].mean(axis=0)-np.array([0,0,9.81])
    bg=imu[times<1.5,1].mean(axis=0)
    # Oracle boundary pose/velocity isolate dead-reckoning through visual outage.
    # This deliberately is NOT a full VIO estimator or a tight-coupling comparison.
    start=round(c.blackout_start*c.imu_hz); end=round(c.blackout_end*c.imu_hz)
    p,r,v,_,_,_=state(times[start],c.duration)
    p0=p.copy(); v0=v.copy(); errors=[]; cv_errors=[]
    for i in range(start,end):
        f=.5*(imu[i,0]+imu[i+1,0])-ba
        w=.5*(imu[i,1]+imu[i+1,1])-bg
        mid=r*Rotation.from_rotvec(w*dt/2)
        a=mid.apply(f)+np.array([0,0,-9.81])
        p=p+v*dt+.5*a*dt*dt; v=v+a*dt
        r=r*Rotation.from_rotvec(w*dt)
        truth=state(times[i+1],c.duration)[0]
        errors.append(np.linalg.norm(p-truth))
        cv_errors.append(np.linalg.norm(p0+v0*(times[i+1]-times[start])-truth))
    return {'scope':'Oracle boundary pose/velocity; biases estimated from stationary warm-up; sensor sanity only, not SLAM or proof of tight coupling',
            'imu_endpoint_error_m':float(errors[-1]),'constant_velocity_endpoint_error_m':float(cv_errors[-1]),
            'imu_outage_rmse_m':float(np.sqrt(np.mean(np.square(errors)))),
            'constant_velocity_outage_rmse_m':float(np.sqrt(np.mean(np.square(cv_errors))))}


def preview(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    timestamps=[4.,9.,13.,18.]
    canvas=Image.new('RGB',(4*320,3*270),(24,27,34)); draw=ImageDraw.Draw(canvas)
    scores={}
    orb=cv2.ORB_create(nfeatures=1000)
    for row,scenario in enumerate(('rich','low_texture','blackout')):
        scores[scenario]=[]
        with SLAMEnvironment(Config(scenario=scenario,width=320,height=240)) as env:
            for col,t in enumerate(timestamps):
                env._set_state(t)
                frame=env._camera(t)
                rgb=frame['rgb']; count=len(orb.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None))
                scores[scenario].append({'time_s':t,'orb_keypoints':count,'valid_depth_fraction':float(np.mean(frame['depth']>0))})
                canvas.paste(Image.fromarray(rgb),(col*320,row*270))
                draw.text((col*320+8,row*270+244),f'{scenario} | t={t:g}s | ORB={count}',fill='white')
    canvas.save(out/'scenarios.png')
    report={'image_diagnostics':scores,'outage_sanity':outage_sanity(),
            'warning':'ORB count is an image diagnostic, not a measure of dense SLAM observability or tracking accuracy.'}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
