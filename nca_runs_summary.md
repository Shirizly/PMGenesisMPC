

2
NCA hidden: 16
NCA steps:  4
Resolution scale: 1.0
Log dir: runs_cubes/nca_mse_mass2_2
47/70 [3:18:51<1:33:21, 243.55s/it, Train Loss=0.0131, Val Loss=0.0129, IoU=0.868, Best=47, No Improv=0]Epoch 48: train loss=0.013065, train MSE=0.012740, val loss=0.012871, val MSE=0.012559, val IoU=0.8680, val Dice=0.9256, val copy MSE=0.017172, val changed MSE=0.624946, val changed copy MSE=1.000000

3
NCA hidden: 16
NCA steps:  4
Resolution scale: 0.25
Log dir: runs_cubes/nca_mse_mass2_3
=== Convergence summary ===
  Epochs run:        70 / 70
  Resumed from:      0
  Best val loss:     0.019715  (epoch 68)
  Suggested budget:  78 epochs  (best epoch + early-stop patience)
Test Loss: 0.020504, Test BCE: 1.237234, Test DiceLoss: 0.132421, Test SharpnessLoss: 0.020476, Test TVLoss: 0.258697, Test MassLoss: 0.003625, Test AddLoss: 0.011626, Test RemoveLoss: 0.008153, Test MSE: 0.019779, Test IoU: 0.8674, Test Dice: 0.9256, Zero MSE: 0.167978, Copy MSE: 0.025118, Changed Pixel Frac: 0.025118, Changed MSE: 0.694778, Changed Zero MSE: 0.474469, Changed Copy MSE: 1.000000

4
NCA hidden: 16
NCA steps:  4
Resolution scale: 0.5
Log dir: runs_cubes/nca_mse_mass2_4
=== Convergence summary ===
  Epochs run:        70 / 70
  Resumed from:      0
  Best val loss:     0.015900  (epoch 70)
  Suggested budget:  80 epochs  (best epoch + early-stop patience)
Test Loss: 0.016610, Test BCE: 1.040803, Test DiceLoss: 0.140843, Test SharpnessLoss: 0.016479, Test TVLoss: 0.129398, Test MassLoss: 0.002752, Test AddLoss: 0.009949, Test RemoveLoss: 0.006110, Test MSE: 0.016060, Test IoU: 0.8640, Test Dice: 0.9234, Zero MSE: 0.122921, Copy MSE: 0.020322, Changed Pixel Frac: 0.020322, Changed MSE: 0.704290, Changed Zero MSE: 0.492871, Changed Copy MSE: 1.000000

5
python train_NCAvk_genesis.py --resolution-scale 0.25
NCA hidden: 32
NCA steps:  8
Resolution scale: 0.25
Log dir: runs_cubes/nca_mse_mass2_5
=== Convergence summary ===
  Epochs run:        37 / 70
  Resumed from:      0
  Best val loss:     0.019607  (epoch 27)
  Suggested budget:  37 epochs  (best epoch + early-stop patience)
Test Loss: 0.020448, Test BCE: 1.236772, Test DiceLoss: 0.128995, Test SharpnessLoss: 0.019656, Test TVLoss: 0.264003, Test MassLoss: 0.004480, Test AddLoss: 0.011320, Test RemoveLoss: 0.008232, Test MSE: 0.019552, Test IoU: 0.8648, Test Dice: 0.9240, Zero MSE: 0.167978, Copy MSE: 0.025118, Changed Pixel Frac: 0.025118, Changed MSE: 0.649055, Changed Zero MSE: 0.474469, Changed Copy MSE: 1.000000