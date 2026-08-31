"""What KIND of nonlinearity does the push map need?

variance_decomposition.py showed displacement is 84% predictable from
grid-visible features but only 58% linearly. This asks whether that 26-point gap
is a simple, cheap nonlinearity (saturation at the push length, a product with
the contact mass) or something genuinely complex -- i.e. whether a
quadratic/bilinear operator could recover it, or only a deep model can.
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from variance_decomposition import load, features, r2, MM

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
s0, s1, quat, ps, pe, run, nf = load()
s0, s1, quat, ps, pe = (t.to(dev) for t in (s0, s1, quat, ps, pe))
Xo, Xp, Xq, inband = features(s0, quat, ps, pe)
disp = (s1 - s0)[..., :2].norm(dim=-1)
cnt = inband.float().sum(1).clamp_min(1)
y = ((disp * inband.float()).sum(1) / cnt * MM).cpu().numpy()
runs = run.numpy()
gkf = GroupKFold(n_splits=5)

def score(X, model='linear'):
    out = []
    for tr, te in gkf.split(X, y, groups=runs):
        sc = X[tr].std(0); sc[sc == 0] = 1; mu = X[tr].mean(0)
        A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
        m = (RidgeCV(alphas=np.logspace(-3, 4, 20)) if model == 'linear'
             else HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                                random_state=0))
        out.append(r2(y[te], m.fit(A, y[tr]).predict(B)))
    return float(np.mean(out))

print(f"n={len(y)}, target = mean displacement in band (mm), sd={y.std():.2f}")
print(f"\n{'model / feature construction':46s} {'R2':>7s}")
print('-' * 55)
print(f"{'LINEAR on raw OCC features':46s} {score(Xo):7.3f}")
print(f"{'BOOSTED on raw OCC features (the target)':46s} {score(Xo, 'gb'):7.3f}")

# 1. saturation: displacement cannot exceed the push length -> try a monotone
#    transform of the target instead of the features
ylog = np.log1p(np.clip(y, 0, None))
def score_t(X, yt, inv):
    out = []
    for tr, te in gkf.split(X, y, groups=runs):
        sc = X[tr].std(0); sc[sc == 0] = 1; mu = X[tr].mean(0)
        A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
        p = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(A, yt[tr]).predict(B)
        out.append(r2(y[te], inv(p)))
    return float(np.mean(out))
print(f"{'  + linear on log1p(target) [saturation]':46s} "
      f"{score_t(Xo, ylog, lambda p: np.expm1(p)):7.3f}")

# 2. quadratic expansion: squares + pairwise products of a compact subset
#    (the whole 26-dim quadratic is 350 terms; use the most informative 8 dims)
imp = np.abs(RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(
    (Xo - Xo.mean(0)) / np.where(Xo.std(0) == 0, 1, Xo.std(0)), y).coef_)
top = np.argsort(imp)[::-1][:8]
Z = Xo[:, top]
quad = np.hstack([Xo, Z ** 2] + [(Z[:, i:i+1] * Z[:, j:j+1])
                                 for i in range(8) for j in range(i + 1, 8)])
print(f"{'  + quadratic in top-8 OCC dims':46s} {score(quad):7.3f}")

# 3. explicit physical nonlinearity: the profile is a *cumulative* quantity --
#    the snow-plough picture says displacement depends on mass swept SO FAR
prof = Xo[:, :18]
cum = np.cumsum(prof, axis=1)
print(f"{'  + cumulative mass profile (snow-plough)':46s} "
      f"{score(np.hstack([Xo, cum])):7.3f}")
print(f"{'  + cumulative AND quadratic':46s} "
      f"{score(np.hstack([quad, cum])):7.3f}")

# 4. random Fourier features: generic smooth nonlinearity, no physics
rng = np.random.default_rng(0)
Xs = (Xo - Xo.mean(0)) / np.where(Xo.std(0) == 0, 1, Xo.std(0))
for D in (200, 1000):
    Wf = rng.normal(scale=0.35, size=(Xs.shape[1], D)); b = rng.uniform(0, 2 * np.pi, D)
    rff = np.hstack([Xo, np.cos(Xs @ Wf + b)])
    print(f"{f'  + {D} random Fourier features':46s} {score(rff):7.3f}")
