import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import numpy as np
import random

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ToolUser.config as config
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# ===================== #
#  CONFIG PLACEHOLDER   #
# ===================== #
H, W = 128, 128  # image resolution (from paper, adjust if needed)
STATE_NORM = False  # whether to normalize states to [0,1]
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 200
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
tool_pts = np.loadtxt('object_outline_normalized_large.txt', dtype=np.float32)


# ===================== #
#    DATASET WRAPPER    #
# ===================== #
# class GranularDynamicsDataset(Dataset):
#     """
#     Dataset for (state, action, next_state) triplets.
#     Loads samples on-the-fly from a list of buffer files.
#     Each buffer file is expected to be a dict with keys: 'states', 'actions', 'next_states'.
#     """
#     def __init__(self, buffer_file_list):
#         self.buffer_file_list = buffer_file_list
#         self.file_sample_counts = []
#         self.cumulative_counts = []
#         self.total_samples = 0

#         # Precompute sample counts for each file
#         for file in self.buffer_file_list:
#             # buf = torch.load(file, map_location='cpu',weights_only=False)
#             # n = len(buf)
#             n = 5000
#             self.file_sample_counts.append(n)
#             self.total_samples += n
#             self.cumulative_counts.append(self.total_samples)

#     def __len__(self):
#         return self.total_samples

#     def _find_file_and_index(self, idx):
#         # Find which file this idx belongs to
#         for i, cum_count in enumerate(self.cumulative_counts):
#             if idx < cum_count:
#                 file_idx = i
#                 local_idx = idx if i == 0 else idx - self.cumulative_counts[i-1]
#                 return file_idx, local_idx
#         raise IndexError("Index out of range")
#     def __getitem__(self, idx):
#         file_idx, local_idx = self._find_file_and_index(idx)
#         buf = torch.load(self.buffer_file_list[file_idx], map_location='cpu',weights_only=False)
#         sample = buf[local_idx]
#         _,s,_, a,_,_, s_next,_,_ = sample  # each of shape (1, H, W) or (2, H, W) for action
#         # if STATE_NORM: # add normalization if needed
            
#         return s, a, s_next

# class LazyGranularDynamicsDataset(Dataset):
#     """
#     Lazy-loading dataset with per-buffer train/val/test splitting.
#     """
#     def __init__(self, buffer_file_list, split="train", train_frac=0.8, val_frac=0.1):
#         assert split in ["train", "val", "test"]
#         self.buffer_file_list = buffer_file_list
#         self.split = split

#         # Precompute ranges per buffer
#         self.buffer_ranges = []
#         self.samples_per_buffer = []
#         for file in buffer_file_list:
#             # buf = torch.load(file, map_location="cpu",weights_only=False)
#             n = 5000#len(buf)
#             self.samples_per_buffer.append(n)

#             train_end = int(n * train_frac)
#             val_end = train_end + int(n * val_frac)

#             if split == "train":
#                 self.buffer_ranges.append((0, train_end))
#             elif split == "val":
#                 self.buffer_ranges.append((train_end, val_end))
#             else:
#                 self.buffer_ranges.append((val_end, n))

#         self.total_samples = sum(end - start for start, end in self.buffer_ranges)

#         # simple cache for last loaded buffer
#         self._cached_buffer_idx = None
#         self._cached_buffer = None

#     def __len__(self):
#         return sum(end - start for start, end in self.buffer_ranges)

#     def _find_file_and_local_index(self, idx):
#         # idx relative to the split
#         cum_count = 0
#         for buf_idx, (start, end) in enumerate(self.buffer_ranges):
#             n = end - start
#             if idx < cum_count + n:
#                 local_idx = start + (idx - cum_count)
#                 return buf_idx, local_idx
#             cum_count += n
#         raise IndexError("Index out of range")

#     def __getitem__(self, idx):
#         buf_idx, local_idx = self._find_file_and_local_index(idx)
#         if self._cached_buffer_idx != buf_idx:
#             self._cached_buffer = torch.load(self.buffer_file_list[buf_idx], map_location="cpu",weights_only=False)
#             self._cached_buffer_idx = buf_idx

#         sample = self._cached_buffer[local_idx]
#         _, s, _, a, _, _, s_next, _, _ = sample
#         return s, a, s_next


# from torch.utils.data import Sampler

# class BufferBatchSampler(Sampler):
#     """
#     Yields batches of indices, each batch entirely within a single buffer file.
#     """
#     def __init__(self, dataset: LazyGranularDynamicsDataset, batch_size: int, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         # Compute index ranges for each buffer
#         self.buffer_ranges = []
#         start = 0
#         for count in dataset.file_sample_counts:
#             self.buffer_ranges.append((start, start + count))
#             start += count

#     def __iter__(self):
#         batch_indices = []
#         # Optionally shuffle buffer order each epoch
#         buffer_order = np.arange(len(self.buffer_ranges))
#         if self.shuffle:
#             np.random.shuffle(buffer_order)

#         for buf_idx in buffer_order:
#             start, end = self.buffer_ranges[buf_idx]
#             indices = np.arange(start, end)
#             if self.shuffle:
#                 np.random.shuffle(indices)

#             # Yield consecutive chunks as batches
#             for i in range(0, len(indices), self.batch_size):
#                 batch = indices[i:i+self.batch_size]
#                 yield batch

#     def __len__(self):
#         # Approximate number of batches
#         return sum((count + self.batch_size - 1) // self.batch_size for count in self.dataset.file_sample_counts)


# def create_dataloaders(buffer_file_list, batch_size=BATCH_SIZE):
#     """
#     Splits the dataset entries (not files) into train/val/test and creates dataloaders.
#     """
#     # # Create a single dataset over all buffer files
#     # full_dataset = LazyGranularDynamicsDataset(buffer_file_list)
#     # total_len = len(full_dataset)
#     # train_len = int(total_len * TRAIN_SPLIT)
#     # val_len = int(total_len * VAL_SPLIT)
#     # test_len = total_len - train_len - val_len

#     # # Randomly split the dataset entries
#     # train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
#     # batch_sampler = BufferBatchSampler(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     # train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
#     # val_loader = DataLoader(val_ds,  batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
#     # test_loader = DataLoader(test_ds, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

#     train_ds = LazyGranularDynamicsDataset(buffer_file_list, split="train")
#     val_ds = LazyGranularDynamicsDataset(buffer_file_list, split="val")
#     test_ds = LazyGranularDynamicsDataset(buffer_file_list, split="test")

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, val_loader, test_loader



# def create_dataloaders_data(data, batch_size=BATCH_SIZE):
    # dataset = GranularDynamicsDataset(data)
    # total_len = len(dataset)
    # train_len = int(total_len * TRAIN_SPLIT)
    # val_len = int(total_len * VAL_SPLIT)
    # test_len = total_len - train_len - val_len

    # train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    # return train_loader, val_loader, test_loader


# -----------------------------
# Multi-Scale Loss
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


from myClasses.UNetModels import UNetOriginal, UNetDeep, UNetDelta, UNetLarge, UNetSmall, UNetMedium

# ===================== #

#    TRAINING LOOP      
# ===================== #
import torch.autograd.profiler as profiler
def train_nfd(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE, log_dir="runs/unet_train", ):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.amp.GradScaler()

    with trange(epochs, desc="Training Epochs") as tbar:
        for epoch in tbar:
            model.train()
            total_loss = 0.0
            i=0
            with profiler.profile(record_shapes=True) as prof:
                for states, actions, next_states in train_loader:
                    i=i+1
                    print(i)
                    states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
                    inputs = torch.cat([states.unsqueeze(1), actions], dim=1)  # (B, 3, H, W)
                    with torch.amp.autocast():
                        pred_next = model(inputs)
                        loss = criterion(pred_next.squeeze(1), next_states)  # (B, 1, H, W) -> (B, H, W)
                    

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    total_loss += loss.item() * states.size(0)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                print(prof.key_averages().table(sort_by="cuda_time_total"))
            avg_train_loss = total_loss / len(train_loader.dataset)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for states, actions, next_states in val_loader:
                    states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
                    inputs = torch.cat([states, actions], dim=1)
                    pred_next = model(inputs)
                    val_loss += criterion(pred_next, next_states).item() * states.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)

            tbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(log_dir, f"unet_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
    writer.close()

def data_augmentation(inputs, outputs):
    # Data augmentation: flips along height and width (using the mirror symmetry of the setup)
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

def train_nfd_perbuffer(model, buffer_file_list, epochs=EPOCHS, lr=LR, device=DEVICE, log_dir="runs/unet_train2",data_aug=False, max_samples_per_load=10000):
    
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
    with trange(epochs, desc="Training Epochs") as tbar:
        for epoch in tbar:

            model.train()
            
            total_loss = 0.0
            val_loss = 0.0
            train_size = 0
            val_size = 0
            # i=0
            file_idx = 0

            while file_idx < n_files:
                # --- Load multiple files into one big buffer ---
                big_buffer = []
                loaded_samples = 0

                while file_idx < n_files and loaded_samples < max_samples_per_load:
                    buffer_file = buffer_file_list[file_idx]
                    buf = torch.load(buffer_file, map_location="cpu", weights_only=False)
                    big_buffer.extend(buf)  # assumes buf is a list-like of samples
                    loaded_samples += len(buf)
                    file_idx += 1
                buf = big_buffer
                
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
                        loss = criterion(pred_next.squeeze(1), outputs)  # (B, 1, H, W) -> (B, H, W)
                    
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
                        val_loss += criterion(pred_next.squeeze(1), outputs.squeeze(1)).item() * inputs.size(0)
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
                save_path = os.path.join(log_dir, f"unet_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
            scheduler.step()
    writer.close()                
    save_path = os.path.join(log_dir, "unet.pth")
    torch.save(model.state_dict(), save_path)


# ===================== #
#   ROLLOUT FUNCTION    #
# ===================== #
def rollout(model, init_state, actions_seq, device=DEVICE):
    """
    # Option 2: Use buffer_file_list for lazy loading
    train_loader, val_loader, test_loader = create_dataloaders(buffer_file_list)
    train_nfd(model, train_loader, val_loader)
    """
    model.eval()
    current_state = init_state.unsqueeze(0).to(device)  # (1,1,H,W)
    predicted_states = [current_state.detach().cpu()]
    with torch.no_grad():
        for action in actions_seq:
            action = action.unsqueeze(0).to(device)  # (1,2,H,W)
            inp = torch.cat([current_state, action], dim=1)  # (1,3,H,W)
            next_state = model(inp)
            predicted_states.append(next_state.detach().cpu())
            current_state = next_state
    return predicted_states

# ===================== #
#   MAIN EXECUTION       #
# ===================== #
if __name__ == "__main__":
    Train = True
    continue_train = False
    Test = False
    evaluate = True
    in_channels = 2  # state + action channels
    type_list = ['mixed']  # 'small', 'medium', 'large', 'original', 'deep', 'deeper', 'delta'
    mixed_blocks_list = [0,1,2,3,4]  # which blocks to use mixed activations in the UNetMixed model
    
    
    ## DATASET DIRECTORY
    buffer_dir_list = []
    # buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_2000ep')
    # buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_1000ep')
    buffer_dir_list.append('datasets/simulation_data/medium_tool_limited_10_40disks_500ep')

    buffer_file_list = []
    for buffer_dir in buffer_dir_list:
        for i in range(30):
            disk_num = i+10
            buffer_file_list.append(f"{buffer_dir}/buffer_sweepfield_{disk_num}.pt")
    # buffer_file_list.append(f"{buffer_dir}/buffer_field_40.pt")
    buffer_dir_list = []
    buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_2000ep')
    buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_1000ep')
    buffer_dir_list.append('datasets/simulation_data/medium_tool_10_40disks_5000ep')

    ## FILES IN DIRS
    buffer_file_list2 = []
    for buffer_dir in buffer_dir_list:
        for i in range(30):
            disk_num = i+10
            buffer_file_list2.append(f"{buffer_dir}/buffer_sweepfield_{disk_num}.pt")
    buffer_file_lists = [buffer_file_list2]

    epoch_count = EPOCHS*2 if continue_train else EPOCHS
    log_dir = [f"datasets/weights/unet{type}deeper_{epoch_count}ep_{LR}lr_{BATCH_SIZE}bs_aug_all"]
    
    for i, buffer_file_list in enumerate(buffer_file_lists):
        print(f"Using {len(buffer_file_list)} buffer files for training/validation/testing.")

        #### TRAINING ####

        if Train:
            for i,type in enumerate(type_list):
                if type == 'small':
                    model = UNetSmall()
                if type == 'large':
                    model = UNetLarge()
                if type == 'original':
                    model = UNetOriginal(in_channels=2)
                if type == 'medium':
                    model = UNetMedium()
                if type == 'deep':
                    model = UNetDeep(in_channels=3, out_channels=1, features=[4,8,16,32,64])
                if type == 'deeper':
                    model = UNetDeep(in_channels=3, out_channels=1, features=[8,16,32,64,128])
                if type == 'delta':
                    model = UNetDelta(in_ch=2, out_ch=1, features=[8,16,32,64])
                if type == 'NCAUNet':
                    from myClasses.NCAModels import NCAPlusUNet
                    model = NCAPlusUNet(in_ch=3, out_ch=1, nca_steps=1, unet_features=[8,16,32])
                if type == 'mixed':
                    from myClasses.UNetModels import UNetMixed
                    model = UNetMixed(in_ch=2,out_ch=1, features=[16,32,64,128,256],mixed_blocks=mixed_blocks_list,act_list=['relu','silu','gelu','mish'])
                if type == 'strided':
                    from myClasses.UNetModels import UNetStrided
                    model = UNetStrided(in_channels=2, out_channels=1, base_channels=8, kernel_size=5, activation=nn.ReLU)        
                
                ## HOW MANY PARAMETERS MODEL HAS
                num_params = sum(p.numel() for p in model.parameters())
                print(f"Number of parameters: {num_params}")
                num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable parameters: {num_trainable}")

                ## CONTINUE TRAINING EXISTING MODEL
                if continue_train:
                    log_dir=f"datasets/weights/unet{type}_train_{EPOCHS}ep_{LR}lr"
                    model_path = os.path.join(log_dir,"unet.pth")
                    if os.path.exists(model_path):
                        print(f"Continuing training from {model_path}")
                        model.load_state_dict(torch.load(model_path, map_location=DEVICE,weights_only=True))
                
                
                train_nfd_perbuffer(model, buffer_file_list,log_dir=log_dir[i],data_aug=True)
        
        #### TESTING ####

        if Test:
            # Load a trained model
            type = type_list[0]
            if type == 'small':
                model = UNetSmall()
            if type == 'large':
                BATCH_SIZE = 128
                model = UNetLarge()
            if type == 'original':
                model = UNetOriginal(in_channels=in_channels)
            if type == 'medium':
                model = UNetMedium()
            if type == 'deep':
                model = UNetDeep(in_channels=in_channels, out_channels=1, features=[4,8,16,32,64])
            if type == 'deeper':
                model = UNetDeep(in_channels=in_channels, out_channels=1, features=[8,16,32,64,128])
            if type == 'delta':
                model = UNetDelta(in_ch=in_channels, out_ch=1, features=[8,16,32,64])
            if type == 'mixed':
                from myClasses.UNetModels import UNetMixed
                model = UNetMixed(in_ch=2,out_ch=1, features=[8,16,32,64,128],mixed_blocks=mixed_blocks_list[0],act_list=['relu','silu','gelu','mish'])
            if type == 'NCAUNet':
                from myClasses.NCAModels import NCAPlusUNet
                model = NCAPlusUNet(in_ch=in_channels, out_ch=1, nca_steps=1, unet_features=[8,16,32])
            epoch_count = EPOCHS*2 if continue_train else EPOCHS
            
            
            log_dir=f"datasets/weights/unet{type}deeper_{epoch_count}ep_{LR}lr_{BATCH_SIZE}bs_aug_limited"
            model_path = os.path.join(log_dir,"unet.pth")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE,weights_only=True))
            model.to(DEVICE)
            model.eval()

            # load test data
            test_loader = None
            if test_loader is None:
                buf = torch.load(buffer_file_list[len(buffer_file_list)-1], map_location="cpu",weights_only=False)  # load full buffer
                # Convert to tensors for GPU transfer
                states = torch.stack([s for _,s,_,a,_,_,s_next,_,_ in buf])
                actions = torch.stack([a for _,s,_,a,_,_,s_next,_,_ in buf])
                outputs = torch.stack([s_next for _,s,_,a,_,_,s_next,_,_ in buf])
                dataset = torch.utils.data.TensorDataset(states,actions, outputs)
                # instead of random_split(...)
                n_total = len(dataset)
                n_train = int(0.8 * n_total)
                n_val = int(0.1 * n_total)
                test_data   = torch.utils.data.Subset(dataset, range(n_train+n_val, n_total))
                test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
           
            # Evaluate on test set
            if evaluate:
                criterion = nn.MSELoss()
                test_loss = 0.0
                with torch.no_grad():
                    for states,actions, outputs in test_loader:
                        inputs = torch.cat([states.unsqueeze(1), actions], dim=1)  # (B, 3, H, W)
                        inputs, outputs = inputs.to(DEVICE), outputs.to(DEVICE)
                        pred_next = model(inputs)
                        test_inputs = pred_next.squeeze(1)
                        # test_inputs = inputs[:,0:1,:,:].squeeze(1)
                        test_loss += criterion(test_inputs, outputs).item() * inputs.size(0)#
                avg_test_loss = test_loss / len(test_loader.dataset)
                print(f"Test Loss: {avg_test_loss}")


            from ToolUser.utils import visualize_physical_state, visualize_transition,visualize_transition_field
            import matplotlib.pyplot as plt
            # Visualize some predictions
            data_iter = iter(test_loader)
            states, actions, next_states = next(data_iter)
            inputs = torch.cat([states.unsqueeze(1), actions], dim=1)
            inputs, outputs = inputs.to(DEVICE), outputs.to(DEVICE)

            with torch.no_grad():
                pred_next = model(inputs)
            states = states.cpu().numpy()
            actions = actions.cpu().numpy()
            next_states = next_states.cpu().numpy()
            pred_next = pred_next.cpu().numpy().squeeze(1) # (B,1,H,W)
            for i in range(3):
                s = states[i]
                a = actions[i]
                s_next = next_states[i]
                s_pred = pred_next[i]
                ax = visualize_transition_field(s,s_next+a[0])
                ax2 = visualize_transition_field(s_next,s_pred)
                plt.show()
                plt.close()
