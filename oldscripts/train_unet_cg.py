###################################################################
# train_unet_cg.py
#
# Trains (or fine-tunes) a UNetFiLM transition model on real
# camera-based pile-manipulation data.
#
# Data format: RealData/dataset.py  (RealPileSweepData)
# Structurally identical to train_unet_genesis.py; key differences:
#   • Uses RealPileSweepData instead of PileSweepData
#   • data_root is a configurable path (not hardcoded to Genesis/data/)
#   • pretrained_path loads weights before training (fine-tuning)
###################################################################

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from RealData.dataset import RealPileSweepData
from GranularDynamics2.myClasses.UNetModels_conditioned import UNetConditioned, UNetFiLM
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
import os

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-4


def chose_loss(loss: str):
    if loss == "mse":
        return torch.nn.MSELoss()
    else:
        from GranularDynamics2.utils import output_dice_loss
        return output_dice_loss


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    continue_training = False
    pretrained_path   = None          # e.g. "runs/unetfilm/unet.pth" to fine-tune
    data_root         = "."           # root directory for real data
    data_folders      = ["real_data"] # subdirectories under data_root
    log_dir           = "runs/unetfilm_cg"
    data_aug          = True
    default_physics   = [0.0, 0.0, 0.0]   # placeholder when physics are unknown

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset: Dataset = RealPileSweepData(
        data_root, data_folders, split="train", default_physics=default_physics
    )
    val_dataset:   Dataset = RealPileSweepData(
        data_root, data_folders, split="val",   default_physics=default_physics
    )
    test_dataset:  Dataset = RealPileSweepData(
        data_root, data_folders, split="test",  default_physics=default_physics
    )

    # +++ INSPECT DATA +++
    # for i in range(len(train_dataset)):
    #     inputs, label = train_dataset[i]
    #     inputt, _ = inputs
    #     train_dataset.plot_input_and_output(inputt.cpu(), label.cpu())

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # model = UNetConditioned(physics_dim=3).to(DEVICE)
    model = UNetFiLM(physics_dim=3).to(DEVICE)

    if continue_training and pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print(f"Loaded pre-trained weights from {pretrained_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    writer    = SummaryWriter(log_dir=log_dir)
    scaler    = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    batch_size   = BATCH_SIZE // 8 if data_aug else BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    avg_val_losses = []
    with trange(EPOCHS, desc="Training Epochs") as tbar:

        for epoch in tbar:
            model.train()
            total_loss = 0.0
            val_loss   = 0.0
            train_size = 0
            val_size   = 0

            for inputs_, outputs in train_loader:
                inputs, physics = inputs_

                inputs  = inputs.to(DEVICE)
                physics = physics.to(DEVICE)
                outputs = outputs.to(DEVICE)

                if data_aug:
                    inputs_rot  = [torch.rot90(inputs,  k, dims=(-2, -1)) for k in range(4)]
                    inputs_mir  = [torch.flip(r, dims=[-1]) for r in inputs_rot]
                    inputs      = torch.cat(inputs_rot + inputs_mir, dim=0)

                    outputs_rot = [torch.rot90(outputs, k, dims=(-2, -1)) for k in range(4)]
                    outputs_mir = [torch.flip(r, dims=[-1]) for r in outputs_rot]
                    outputs     = torch.cat(outputs_rot + outputs_mir, dim=0)

                    physics = physics.repeat(8, 1)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    pred_next = model(inputs, physics)
                loss = criterion(pred_next.squeeze(1).float(), outputs)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * inputs.size(0)
                train_size += inputs.size(0)

            print("Total loss", total_loss, "train size", train_size)

            # --------------------------------------------------------------
            # Validation loop
            # --------------------------------------------------------------
            model.eval()
            with torch.no_grad():
                for inputs_, outputs in val_loader:
                    inputs, physics = inputs_

                    inputs  = inputs.to(DEVICE)
                    physics = physics.to(DEVICE)
                    outputs = outputs.to(DEVICE)

                    pred_next = model(inputs, physics)
                    val_loss += criterion(pred_next.squeeze(1), outputs.squeeze(1)).item() * inputs.size(0)
                    val_size += inputs.size(0)

            avg_val_loss = val_loss / val_size
            avg_val_losses.append(avg_val_loss)
            print("Avg val loss", avg_val_loss, "val size", val_size)

            if len(avg_val_losses) > 5 and avg_val_loss > max(avg_val_losses[-5:]):
                print("Early stopping due to no improvement in validation loss.")
                break

            avg_train_loss = total_loss / train_size
            writer.add_scalar("Loss/Val",   avg_val_loss,   epoch)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(log_dir, f"unet_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)

            scheduler.step()

    writer.close()
    save_path = os.path.join(log_dir, "unet.pth")
    torch.save(model.state_dict(), save_path)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    criterion  = torch.nn.MSELoss()
    test_loss  = 0.0
    model.eval()
    with torch.no_grad():
        for inputs_, outputs in test_loader:
            inputs, physics = inputs_

            inputs  = inputs.to(DEVICE)
            physics = physics.to(DEVICE)
            outputs = outputs.to(DEVICE)

            pred_next   = model(inputs, physics)
            test_inputs = pred_next.squeeze(1)
            test_loss  += criterion(test_inputs, outputs).item() * inputs.size(0)

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss}")
