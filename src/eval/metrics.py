import numpy as np

def align_trajectory(est, gt):
    est_centered = est - np.mean(est, axis=0)
    gt_centered = gt - np.mean(gt, axis=0)
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = np.mean(gt, axis=0) - R @ np.mean(est, axis=0)
    return (R @ est.T).T + t

def ATE(est, gt):
    aligned = align_trajectory(est, gt)
    return np.sqrt(np.mean(np.linalg.norm(aligned - gt, axis=1)**2))

def RPE(est, gt, delta=1):
    errors = []
    for i in range(len(est)-delta):
        d_est = est[i+delta] - est[i]
        d_gt = gt[i+delta] - gt[i]
        errors.append(np.linalg.norm(d_est - d_gt))
    return np.mean(errors)

def heading_error(est_psi, gt_psi):
    diff = np.abs(est_psi - gt_psi)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.mean(diff)
