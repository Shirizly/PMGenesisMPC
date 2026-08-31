"""Restate the foresight results in physical units instead of occupancy-per-pixel.

The rms numbers the fit reports are dimensionless occupancy per pixel, which is
uninterpretable on its own and makes real differences look like rounding. This
converts to: cube-equivalents of mass, millimetres of centre-of-mass, and
percentage of the change that actually occurred.

Calibration for the 40 mm dataset: 128 mm box on a 64x64 grid -> 2.0 mm/px;
50 cubes occupy ~492 px of binary silhouette -> ~9.8 px per cube.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (actions_to_pixels, canonicalise, contact_score,
    fit_operator_nonneg, predict_heuristic, predict_world, swept_region_mask)
from loro_foresight import gaussian_blur

import argparse as _ap
_p=_ap.ArgumentParser(); _p.add_argument('--dataset', default='configs/dataset/genesis_foresight_L040.yaml')
_p.add_argument('--n-particles', type=int, default=50)
_p.add_argument('--res', type=int, default=32); _p.add_argument('--crop', type=float, default=0.5)
_p.add_argument('--blur', type=float, default=1.0); _p.add_argument('--folds', type=int, default=8)
_a=_p.parse_args()
CFG=_a.dataset
R,CR,BLUR,ITERS,B=_a.res,_a.crop,_a.blur,2000,3
d=load_transition_arrays(CFG,'train')
H,W=d.occ_t.shape[-2:]
MM_PER_PX=128.0/W
d.occ_t,d.occ_t1=gaussian_blur(d.occ_t,BLUR),gaussian_blur(d.occ_t1,BLUR)
s,e=actions_to_pixels(d.actions,d.workspace_min,d.workspace_max,(H,W))
dev='cuda' if torch.cuda.is_available() else 'cpu'
o0,o1,s,e=d.occ_t.to(dev),d.occ_t1.to(dev),s.to(dev),e.to(dev)
eid=d.episode_ids.to(dev); runs=eid.unique(); plate=0.04/0.128*W
PX_PER_CUBE=float(o0.sum(dim=(1,2)).mean())/float(_a.n_particles)

rows,base={},{}
runs = runs[:_a.folds] if len(runs) > _a.folds else runs
for r in runs:
    mte=eid==r; mtr=~mte
    ote,o1te,ste,ete=o0[mte],o1[mte],s[mte],e[mte]
    reg=swept_region_mask(ste,ete,(H,W),half_width_px=0.5*plate+2.0,pad_px=0.5*plate)
    Y0=canonicalise(o0[mtr],s[mtr],e[mtr],R,CR).reshape(int(mtr.sum()),-1).T
    Y1=canonicalise(o1[mtr],s[mtr],e[mtr],R,CR).reshape(int(mtr.sum()),-1).T
    A=fit_operator_nonneg(Y0,Y1,max_iter=ITERS,ridge=1.0)
    eye=torch.eye(R*R,device=dev)
    preds={'persistence (do nothing)':ote,
           'identity (warp only)':predict_world(eye,ote,ste,ete,R,(H,W),CR),
           'linear operator':predict_world(A,ote,ste,ete,R,(H,W),CR),
           'heuristic cumulative':predict_heuristic('cumulative',ote.cpu(),ste.cpu(),ete.cpu()).to(dev)}
    rows_gr=torch.arange(H,device=dev).float(); rows_gc=torch.arange(W,device=dev).float()
    gr,gc=torch.meshgrid(rows_gr,rows_gc,indexing='ij')
    def com(x):
        m=(x*reg).sum(dim=(1,2)).clamp_min(1e-6)
        return torch.stack([((x*reg)*gr).sum(dim=(1,2))/m,((x*reg)*gc).sum(dim=(1,2))/m],dim=-1)
    def mass(x): return (x*reg).sum(dim=(1,2))
    n=ote.shape[0]; npx=reg.reshape(n,-1).sum(1).clamp_min(1)
    truth_com,truth_mass=com(o1te),mass(o1te)
    # what actually happened
    base.setdefault('change_l2',[]).append(((o1te-ote)*reg).reshape(n,-1).norm(dim=1).mean())
    base.setdefault('change_rms',[]).append((((o1te-ote)*reg).reshape(n,-1).pow(2).sum(1)/npx).sqrt().mean())
    base.setdefault('com_shift_mm',[]).append(((com(o1te)-com(ote)).norm(dim=-1)*MM_PER_PX).mean())
    base.setdefault('mass_change_cubes',[]).append(((mass(o1te)-mass(ote)).abs()/PX_PER_CUBE).mean())
    base.setdefault('mass_in_region_cubes',[]).append((mass(ote)/PX_PER_CUBE).mean())
    for nm,p in preds.items():
        err=((p-o1te)*reg).reshape(n,-1)
        rows.setdefault(nm,{}).setdefault('rms',[]).append((err.pow(2).sum(1)/npx).sqrt().mean())
        rows[nm].setdefault('l2',[]).append(err.norm(dim=1).mean())
        rows[nm].setdefault('com_mm',[]).append(((com(p)-truth_com).norm(dim=-1)*MM_PER_PX).mean())
        rows[nm].setdefault('mass_cubes',[]).append(((mass(p)-truth_mass).abs()/PX_PER_CUBE).mean())

print(f"calibration: {MM_PER_PX:.2f} mm/px, {PX_PER_CUBE:.1f} px per cube "
      f"(50 cubes, binary silhouette)\n")
print(f"WHAT ONE PUSH ACTUALLY DOES, inside the swept band (mean over {len(runs)} folds):")
print(f"  material present in the band  : {float(torch.stack(base['mass_in_region_cubes']).mean()):6.1f} cube-equivalents")
print(f"  net mass change in the band   : {float(torch.stack(base['mass_change_cubes']).mean()):6.1f} cube-equivalents")
print(f"  centre-of-mass shift          : {float(torch.stack(base['com_shift_mm']).mean()):6.2f} mm  (push is 40 mm)")
print(f"  per-pixel rms change          : {float(torch.stack(base['change_rms']).mean()):6.4f} occupancy units")
ch=float(torch.stack(base['change_rms']).mean())
cs=float(torch.stack(base['com_shift_mm']).mean())
print(f"\nERRORS (mean over {len(runs)} folds), and as a share of that actual change:")
hdr=f"{'model':26s} {'rms':>8s} {'rms as % of':>12s} {'COM err':>9s} {'COM err as':>12s} {'mass err':>10s}"
print(hdr); print(f"{'':26s} {'':>8s} {'the change':>12s} {'(mm)':>9s} {'% of shift':>12s} {'(cubes)':>10s}")
print("-"*len(hdr))
for nm,v in rows.items():
    rms=float(torch.stack(v['rms']).mean()); cm=float(torch.stack(v['com_mm']).mean())
    ms=float(torch.stack(v['mass_cubes']).mean())
    print(f"{nm:26s} {rms:8.4f} {100*rms/ch:11.1f}% {cm:9.2f} {100*cm/cs:11.1f}% {ms:10.2f}")
print(f"\nreference: predicting the change PERFECTLY would give rms 0, COM err 0.")
print(f"'rms as % of the change' = error relative to how much actually moved;")
print(f"100% means the model's error is as large as the entire change, i.e. it")
print(f"is no better than doing nothing.")
