"""Does the operator explain a real share of the change on pushes that ENGAGE
the pile? Stratified by contact score, in physical units."""
from __future__ import annotations
import torch
from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (actions_to_pixels, canonicalise, contact_score,
    fit_operator_nonneg, predict_heuristic, predict_world, swept_region_mask)
from loro_foresight import gaussian_blur

d=load_transition_arrays('configs/dataset/genesis_foresight_L040.yaml','train')
H,W=d.occ_t.shape[-2:]; MM=128.0/W; R,CR=32,0.5
d.occ_t,d.occ_t1=gaussian_blur(d.occ_t,1.0),gaussian_blur(d.occ_t1,1.0)
s,e=actions_to_pixels(d.actions,d.workspace_min,d.workspace_max,(H,W))
dev='cuda' if torch.cuda.is_available() else 'cpu'
o0,o1,s,e=d.occ_t.to(dev),d.occ_t1.to(dev),s.to(dev),e.to(dev)
eid=d.episode_ids.to(dev); runs=eid.unique(); plate=0.04/0.128*W
PXC=float(o0.sum(dim=(1,2)).mean())/50.0
total_pile=float(o0.sum(dim=(1,2)).mean())
cs_all=contact_score(o0,s,e,(H,W),plate)
print(f"pile in blade path: mean {100*float(cs_all.mean())/total_pile:.1f}% of the pile")
print(f"  pushes engaging <5% of the pile : {100*float((cs_all<0.05*total_pile).float().mean()):.1f}%")
print(f"  pushes engaging >25% of the pile: {100*float((cs_all>0.25*total_pile).float().mean()):.1f}%")
QS=[0.0,0.5,0.8,0.95,1.0]; LBL=['bottom 50%','50-80%','80-95%','top 5%']
acc={}
for r in runs:
    mte=eid==r; mtr=~mte
    ote,o1te,ste,ete=o0[mte],o1[mte],s[mte],e[mte]
    reg=swept_region_mask(ste,ete,(H,W),half_width_px=0.5*plate+2.0,pad_px=0.5*plate)
    Y0=canonicalise(o0[mtr],s[mtr],e[mtr],R,CR).reshape(int(mtr.sum()),-1).T
    Y1=canonicalise(o1[mtr],s[mtr],e[mtr],R,CR).reshape(int(mtr.sum()),-1).T
    A=fit_operator_nonneg(Y0,Y1,max_iter=2000,ridge=1.0)
    lin=predict_world(A,ote,ste,ete,R,(H,W),CR)
    cum=predict_heuristic('cumulative',ote.cpu(),ste.cpu(),ete.cpu()).to(dev)
    cte=contact_score(ote,ste,ete,(H,W),plate)
    ed=torch.quantile(contact_score(o0[mtr],s[mtr],e[mtr],(H,W),plate),
                      torch.tensor(QS[1:-1],device=dev))
    b=torch.bucketize(cte,ed)
    gr,gc=torch.meshgrid(torch.arange(H,device=dev).float(),torch.arange(W,device=dev).float(),indexing='ij')
    def com(x,rg):
        m=(x*rg).sum(dim=(1,2)).clamp_min(1e-6)
        return torch.stack([((x*rg)*gr).sum(dim=(1,2))/m,((x*rg)*gc).sum(dim=(1,2))/m],dim=-1)
    for bi in range(4):
        m=b==bi
        if int(m.sum())<3: continue
        rg=reg[m]; a0,a1=ote[m],o1te[m]; n=int(m.sum())
        npx=rg.reshape(n,-1).sum(1).clamp_min(1)
        chg=(((a1-a0)*rg).reshape(n,-1).pow(2).sum(1)/npx).sqrt()
        cshift=(com(a1,rg)-com(a0,rg)).norm(dim=-1)*MM
        for nm,p in [('persistence',a0),('linear',lin[m]),('cumulative',cum[m])]:
            err=(((p-a1)*rg).reshape(n,-1).pow(2).sum(1)/npx).sqrt()
            cerr=(com(p,rg)-com(a1,rg)).norm(dim=-1)*MM
            # Ratio of MEANS, not mean of ratios: per-sample ratios blow up
            # where a push changed almost nothing (a zero-change sample makes
            # err/chg either 0/0 or divide by ~0). Verified by the fact that
            # persistence, whose error IS the change, must read exactly 100%.
            acc.setdefault((bi,nm),{'e':[],'c':[]})
            acc[(bi,nm)]['e'].append(err.mean())
            acc[(bi,nm)]['c'].append(cerr.mean())
        acc.setdefault((bi,'_chg'),{'e':[],'c':[]})
        acc[(bi,'_chg')]['e'].append(chg.mean()); acc[(bi,'_chg')]['c'].append(cshift.mean())
print(f"\n{'contact bin':>13s} {'real change':>12s} {'COM shift':>10s} | "
      f"{'error as % of the change':>26s}")
print(f"{'':>13s} {'(rms occ)':>12s} {'(mm)':>10s} | {'persist':>9s} {'linear':>8s} {'cumul':>7s}")
print("-"*76)
for bi in range(4):
    if (bi,'_chg') not in acc: continue
    ch=float(torch.stack(acc[(bi,'_chg')]['e']).mean()); cm=float(torch.stack(acc[(bi,'_chg')]['c']).mean())
    row=f"{LBL[bi]:>13s} {ch:12.4f} {cm:10.2f} | "
    for nm in ('persistence','linear','cumulative'):
        row+=f"{100*float(torch.stack(acc[(bi,nm)]['e']).mean())/ch:8.1f}%"
    print(row)
print("\nSame, on centre of mass (error as % of the true COM shift):")
for bi in range(4):
    if (bi,'_chg') not in acc: continue
    row=f"{LBL[bi]:>13s} {'':22s} | "
    for nm in ('persistence','linear','cumulative'):
        row+=f"{100*float(torch.stack(acc[(bi,nm)]['c']).mean())/cm:8.1f}%"
    print(row)
