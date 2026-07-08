import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import numpy as np
import random

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ToolUser.config as config
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# ===================== #
#  Choose use           #
# ===================== #
Train = True
Test = True
Visualize = True
model_type = "multiexit"
loss_type = "mse"
type = model_type + "_" + loss_type
dataset_type = "medium_tool_limited"


# ===================== #
#  CONFIG PLACEHOLDER   #
# ===================== #
H, W = 128, 128  # image resolution (from paper, adjust if needed)
STATE_NORM = False  # whether to normalize states to [0,1]
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
asset_path = 'assets'
tool_pts = np.loadtxt(os.path.join(asset_path,'object_outline_normalized_large.txt'), dtype=np.float32)
lambda_low  = 0.01
lambda_mid  = 0.1
lambda_high = 1.0
LAMBDA = [lambda_low, lambda_mid, lambda_high]


# -----------------------------
# Some utility functions for training
# -----------------------------
def multiscale_loss(pred, target, criterion=nn.MSELoss()):
    """
    Multi-scale reconstruction loss.
    Args:
        pred, target: (B, 1, H, W)
    """
    losses = []
    for scale in [1, 2, 4]:  # original, 1/2, 1/4 resolution
        if scale > 1:
            pred_s = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_s = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        else:
            pred_s, target_s = pred, target
        losses.append(criterion(pred_s, target_s))
    return sum(losses) / len(losses)



def data_augmentation(inputs, outputs):
    """
    Data augmentation: flips along height and width (using the mirror symmetry of the setup)
    """
    # Original
    aug_inputs = [inputs]
    aug_outputs = [outputs]

    # Horizontal flip (dim=3: width)
    aug_inputs.append(torch.flip(inputs, dims=[3]))
    aug_outputs.append(torch.flip(outputs, dims=[2]))  # careful: outputs squeezed (B,1,H,W) → flip H/W accordingly

    # Vertical flip (dim=2: height)
    aug_inputs.append(torch.flip(inputs, dims=[2]))
    aug_outputs.append(torch.flip(outputs, dims=[1]))

    # Both flips
    aug_inputs.append(torch.flip(inputs, dims=[2,3]))
    aug_outputs.append(torch.flip(outputs, dims=[1,2]))

    # Concatenate all augmented versions
    inputs = torch.cat(aug_inputs, dim=0)
    outputs = torch.cat(aug_outputs, dim=0)      
    return inputs, outputs     

def multi_file_loader(buffer_file_list, file_idx, max_samples=50000):
    """
    Load multiple buffer files into one big buffer until reaching max_samples or end of list.
    Args:
        buffer_file_list: list of file paths
        start_idx: index to start loading from
        max_samples: maximum number of samples to load
    """
    big_buffer = []
    loaded_samples = 0

    while file_idx < len(buffer_file_list) and loaded_samples < max_samples:
        buffer_file = buffer_file_list[file_idx]
        buf = torch.load(buffer_file, map_location="cpu", weights_only=False)
        big_buffer.extend(buf)  # assumes buf is a list-like of samples
        loaded_samples += len(buf)
        file_idx += 1
    return big_buffer, file_idx


if loss_type == "mse":
    criterion = nn.MSELoss()
elif loss_type == "dice":
    from utils import output_dice_loss
    criterion = output_dice_loss
elif loss_type == "mixed":
    from utils import output_loss
    criterion = output_loss

def multi_exit_loss(preds, gt, LAMBDA=[0.2, 0.3, 0.5]):
    # Losses = [model_loss(preds[key], gt.unsqueeze(1)) for key in preds]
    Losses = [criterion(preds['low'], gt.unsqueeze(1)),
                criterion(preds['mid'], gt.unsqueeze(1)),
                criterion(preds['high'], gt.unsqueeze(1))]
    normed_Losses = Losses #[norm(f"exit_{i}", loss) for i, loss in enumerate(Losses)]
    return sum(x * y for x, y in zip(LAMBDA,normed_Losses))


from myClasses.MultiExitUnet import UNetMultiExit

# ===================== #

#    TRAINING LOOP      
# ===================== #
import torch.autograd.profiler as profiler

def train_nfd(model, buffer_file_list, epochs=EPOCHS, lr=LR, device=DEVICE, log_dir="runs/unet_train",data_aug=False, max_samples_per_load=20000):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = multiscale_loss
    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.amp.GradScaler()
    # Add learning rate scheduler (annealing)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    avg_val_losses = []
    file_idx = 0
    n_files = len(buffer_file_list)
    # norm = LossNormalizer(alpha=0.01)
    with trange(epochs, desc="Training Epochs") as tbar:
        for epoch in tbar:
            model.train()
            total_loss = 0.0
            val_loss = 0.0
            train_size = 0
            val_size = 0
            file_idx = 0
            while file_idx < n_files:
                # --- Load multiple files into one big buffer ---
                buf, file_idx = multi_file_loader(buffer_file_list, file_idx, max_samples=max_samples_per_load)
                # Convert to tensors for GPU transfer
                states = torch.stack([s for _,s,_,a,_,_,s_next,_,_ in buf])
                actions = torch.stack([a for _,s,_,a,_,_,s_next,_,_ in buf])
                outputs = torch.stack([s_next for _,s,_,a,_,_,s_next,_,_ in buf])
                inputs = torch.cat([states.unsqueeze(1), actions], dim=1)  # (N, 3, H, W)
                dataset = torch.utils.data.TensorDataset(inputs, outputs)
                # instead of random_split(...)
                n_total = len(dataset)
                n_train = int(0.8 * n_total)
                n_val = int(0.1 * n_total)
                train_data = torch.utils.data.Subset(dataset, range(0, n_train))
                val_data   = torch.utils.data.Subset(dataset, range(n_train, n_train+ n_val))
                # train_data, val_data = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
                if data_aug:
                    current_batch_size =  BATCH_SIZE // 4
                val_loader = DataLoader(val_data, batch_size=current_batch_size, shuffle=False, num_workers=4, pin_memory=True)
                train_loader = DataLoader(train_data, batch_size=current_batch_size, shuffle=True, num_workers=4, pin_memory=True)
                # with profiler.profile(record_shapes=True) as prof:
                for inputs, outputs in train_loader:
                    if data_aug:
                        inputs,outputs = data_augmentation(inputs,outputs)
                    inputs, outputs = inputs.to(device), outputs.to(device)
                
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device, dtype=torch.float16):
                        pred_next = model(inputs)
                        loss = multi_exit_loss(pred_next, outputs,LAMBDA)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    total_loss += loss.item() * inputs.size(0)
                    train_size += inputs.size(0)
                # Validation
                model.eval()
                
                with torch.no_grad():
                    for inputs, outputs in val_loader:
                        inputs, outputs = inputs.to(device), outputs.to(device)
                        pred_next = model(inputs)
                        val_loss += multi_exit_loss(pred_next, outputs,LAMBDA).item() * inputs.size(0)
                        val_size += inputs.size(0)
            avg_val_loss = val_loss / val_size
            avg_val_losses.append(avg_val_loss)
            if len(avg_val_losses) > 5 and avg_val_loss > max(avg_val_losses[-5:]):
                print("Early stopping due to no improvement in validation loss.")
                break
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            avg_train_loss = total_loss / train_size
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.save_checkpoint(epoch)
            scheduler.step()
            pass
    writer.close()                
    


def newest_exp_num(log_dir, prefix="unet_exp_"):
    existing = [d for d in os.listdir(log_dir) if d.startswith(prefix)]
    if not existing:
        return 0
    nums = [int(d.replace(prefix, "").split("_")[0]) for d in existing if d.replace(prefix, "").split("_")[0].isdigit()]
    return max(nums) + 1 if nums else 0

def last_epoch_num(log_dir, prefix="unet_epoch_"):
    existing = [d.replace(".pth", "") for d in os.listdir(log_dir) if d.startswith(prefix)] 
    if not existing:
        return -1
    nums = [int(d.replace(prefix, "").split("_")[0]) for d in existing if d.replace(prefix, "").split("_")[0].isdigit()]
    return max(nums) if nums else -1

# ===================== #
#   MAIN EXECUTION       #
# ===================== #
if __name__ == "__main__":
    model = None
    exp_num = None
    in_channels = 2  # state + action channels
    ## prepare model structures for training/testing
    # 

    # "bottleneck_kwargs": {"num_heads": 4, "num_layers": 2, "dim_feedforward": 512}
    
    # dataset directory
    # first experiment:
    buffer_dir_list = []
    buffer_dir_list.append('datasets/simulation_data/medium_tool_limited_10_40disks_500ep')

    buffer_file_list_limited = []
    for buffer_dir in buffer_dir_list:
        for i in range(0,30,1):
            disk_num = i+10
            buffer_file_list_limited.append(f"{buffer_dir}/buffer_sweepfield_{disk_num}.pt")
    # buffer_file_list.append(f"{buffer_dir}/buffer_field_40.pt")
    # second experiment:
    buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_2000ep')
    buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_1000ep')
    # buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_5000ep')
    buffer_file_list_all = buffer_file_list_limited.copy()
    for buffer_dir in buffer_dir_list:
        for i in range(20,30):
            disk_num = i+10
            buffer_file_list_all.append(f"{buffer_dir}/buffer_sweepfield_{disk_num}.pt")
    if dataset_type == "medium_tool_limited":
        buffer_file_lists = [buffer_file_list_limited]
    else:
        buffer_file_lists = [buffer_file_list_all]


    if Train:
        log_dir_list = [f"datasets/weights/unet{type}_exp_{0}"]
        for i,buffer_file_list in enumerate(buffer_file_lists):
            print(f"Using {len(buffer_file_list)} buffer files for training/validation/testing.")
            model = UNetMultiExit(base_ch=8, path=log_dir_list[0])
            model.to(DEVICE)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {num_params}")
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num_trainable}")

                # add later newer model loading
                
            exp_num = newest_exp_num(log_dir = "datasets/weights", prefix=f"unet{type}_exp_")
            train_nfd(model, buffer_file_list,log_dir=f"datasets/weights/unet{type}_exp_{exp_num}",data_aug=True)

    if Test:
        if model is None:
            model = UNetMultiExit(base_ch=16)
            if exp_num is None:
                exp_num = newest_exp_num(log_dir = "datasets/weights", prefix=f"unet{type}_exp_")-1
            last = last_epoch_num(f"datasets/weights/unet{type}_exp_{exp_num}",prefix="unet_epoch_")
            model.load_state_dict(torch.load(os.path.join(f"datasets/weights/unet{type}_exp_{exp_num}",f'unet_epoch_{last}.pth'), map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
        model.eval()
        # norm = LossNormalizer(alpha=0.01)
        MEtotal_loss = 0.0
        L1total_loss = 0.0
        test_size = 0
        buffer_file_list = buffer_file_list_all
        n_files = len(buffer_file_list)
        file_idx = 0
        while file_idx < n_files:
            # --- Load multiple files into one big buffer ---
            buf, file_idx = multi_file_loader(buffer_file_list, file_idx, max_samples=20000)
            # Convert to tensors for GPU transfer
            states = torch.stack([s for _,s,_,a,_,_,s_next,_,_ in buf])
            actions = torch.stack([a for _,s,_,a,_,_,s_next,_,_ in buf])
            outputs = torch.stack([s_next for _,s,_,a,_,_,s_next,_,_ in buf])
            inputs = torch.cat([states.unsqueeze(1), actions], dim=1)  # (N, 3, H, W)
            dataset = torch.utils.data.TensorDataset(inputs, outputs)
            test_data   = torch.utils.data.Subset(dataset, range(0, int(0.1 * len(dataset))))
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            with torch.no_grad():
                for inputs, outputs in test_loader:
                    inputs, outputs = inputs.to(DEVICE), outputs.to(DEVICE)
                    pred_next = model(inputs)
                    MEloss = multi_exit_loss(pred_next, outputs,LAMBDA)
                    MEtotal_loss += MEloss.item() * inputs.size(0)
                    l1loss = F.l1_loss(pred_next['high'], outputs)
                    L1total_loss += l1loss.item() * inputs.size(0)
                    test_size += inputs.size(0)
        avg_ME_loss = MEtotal_loss / test_size
        avg_L1_loss = L1total_loss / test_size
        print(f"Test Multi-Exit Loss: {avg_ME_loss:.6f}, L1 Loss: {avg_L1_loss:.6f}")

    if Visualize:
        import matplotlib.pyplot as plt
        import random
        if not model:
            model = UNetMultiExit(base_ch=16)
            if exp_num is None:
                exp_num = newest_exp_num(log_dir = "datasets/weights", prefix=f"unet{type}_exp_")-1
            last = last_epoch_num(f"datasets/weights/unet{type}_exp_{exp_num}",prefix="unet_epoch_")
            model.load_state_dict(torch.load(os.path.join(f"datasets/weights/unet{type}_exp_{exp_num}",f'unet_epoch_{last}.pth'), map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
        model.eval()
        buffer_file_list = buffer_file_list_all
        # pick a random buffer file
        buffer_file = buffer_file_list[-1]
        print(f"Visualizing from buffer file: {buffer_file}")
        buf = torch.load(buffer_file, map_location="cpu", weights_only=False)
        states = torch.stack([s for _,s,_,a,_,_,s_next,_,_ in buf])
        actions = torch.stack([a for _,s,_,a,_,_,s_next,_,_ in buf])
        outputs = torch.stack([s_next for _,s,_,a,_,_,s_next,_,_ in buf])
        inputs = torch.cat([states.unsqueeze(1), actions], dim=1)  # (N, 3, H, W)
        dataset = torch.utils.data.TensorDataset(inputs, outputs)
        sample_idx = random.randint(0, len(dataset)-1)
        input_sample, output_sample = dataset[sample_idx]
        input_sample = input_sample.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_next = model(input_sample)
        pred_next_high = pred_next['high'].squeeze().cpu().numpy()
        gt_next = output_sample.squeeze().numpy()
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(input_sample[0,0].cpu().numpy(), cmap='gray')
        axs[0].set_title("Input State")
        axs[1].imshow(gt_next, cmap='gray')
        axs[1].set_title("Ground Truth Next State")
        axs[2].imshow(pred_next_high, cmap='gray')
        axs[2].set_title("Predicted Next State (High Exit)")
        plt.show()