"""Metric SE(3) alignment; no scale fitting; fixed alignment before outage."""
import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(groundtruth, estimate, outage=(8., 12.), tolerance=.002):
    gt = np.atleast_2d(np.loadtxt(groundtruth))
    try: est = np.atleast_2d(np.loadtxt(estimate))
    except ValueError: est = np.empty((0,8))
    if est.size == 0: return {'coverage':0., 'matched_frames':0}
    for name, arr in [('groundtruth',gt),('estimate',est)]:
        if arr.shape[1] != 8 or not np.isfinite(arr).all() or np.any(np.diff(arr[:,0])<=0):
            raise ValueError(f'{name} must be finite, sorted, unique TUM rows')
        if not np.allclose(np.linalg.norm(arr[:,4:8], axis=1), 1, atol=1e-4):
            raise ValueError(f'{name} requires unit xyzw quaternions')
    # Associate each estimate once, then reject duplicates in the GT association.
    idx = np.searchsorted(gt[:,0],est[:,0]).clip(0,len(gt)-1)
    prev = np.maximum(idx-1,0)
    idx = np.where(abs(gt[prev,0]-est[:,0])<abs(gt[idx,0]-est[:,0]),prev,idx)
    good = abs(gt[idx,0]-est[:,0]) <= tolerance
    est, idx = est[good], idx[good]
    if len(np.unique(idx)) != len(idx): raise ValueError('Multiple estimates associate to one truth frame')
    if not len(idx): return {'coverage':0., 'matched_frames':0}
    ref = gt[idx]
    # First-pose SE(3) alignment fixes position AND yaw even on stationary warm-up.
    Rg = Rotation.from_quat(ref[0,4:8]).as_matrix()
    Re = Rotation.from_quat(est[0,4:8]).as_matrix()
    R = Rg@Re.T; offset=ref[0,1:4]-R@est[0,1:4]
    p = est[:,1:4]@R.T+offset
    rotations = Rotation.from_matrix(R)*Rotation.from_quat(est[:,4:8])
    errors=np.linalg.norm(p-ref[:,1:4],axis=1)
    angles=(Rotation.from_quat(ref[:,4:8]).inv()*rotations).magnitude()*180/np.pi
    report={'alignment':'first matched pose SE(3); no scale fit', 'coverage':len(idx)/len(gt),'matched_frames':len(idx),
            'ate_rmse_m':float(np.sqrt(np.mean(errors**2))), 'rotation_rmse_deg':float(np.sqrt(np.mean(angles**2)))}
    for label,mask in [('before_outage',ref[:,0]<outage[0]),('during_outage',(ref[:,0]>=outage[0])&(ref[:,0]<outage[1])),('after_outage',ref[:,0]>=outage[1])]:
        expected = np.count_nonzero((gt[:,0]<outage[0]) if label=='before_outage' else ((gt[:,0]>=outage[0])&(gt[:,0]<outage[1])) if label=='during_outage' else (gt[:,0]>=outage[1]))
        report[label]={'coverage':float(mask.sum()/expected) if expected else None,
                       'ate_rmse_m':float(np.sqrt(np.mean(errors[mask]**2))) if mask.any() else None,
                       'max_error_m':float(errors[mask].max()) if mask.any() else None}
    # 1-second relative pose errors, only on pairs with actual matching timestamps.
    deltap=[]; deltar=[]
    for i,t in enumerate(ref[:,0]):
        j=min(np.searchsorted(ref[:,0],t+1),len(ref)-1)
        if j>0 and abs(ref[j-1,0]-t-1)<abs(ref[j,0]-t-1): j-=1
        if j<=i or abs(ref[j,0]-t-1)>tolerance: continue
        rg=Rotation.from_quat(ref[i,4:8]); re=rotations[i]
        dg=rg.inv().apply(ref[j,1:4]-ref[i,1:4]); de=re.inv().apply(p[j]-p[i])
        deltap.append(np.linalg.norm(de-dg))
        deltar.append(((rg.inv()*Rotation.from_quat(ref[j,4:8])).inv()*(re.inv()*rotations[j])).magnitude()*180/np.pi)
    report['rpe_1s_translation_rmse_m']=float(np.sqrt(np.mean(np.square(deltap)))) if deltap else None
    report['rpe_1s_rotation_rmse_deg']=float(np.sqrt(np.mean(np.square(deltar)))) if deltar else None
    report['rpe_1s_pairs']=len(deltap)
    return report
