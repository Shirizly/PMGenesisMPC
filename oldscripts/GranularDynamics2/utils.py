import torch
import torch.nn.functional as F

class LossNormalizer:
    def __init__(self, alpha=0.01, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.running = {}

    def __call__(self, name, value):
        v = value.detach()
        if name not in self.running:
            self.running[name] = v
        self.running[name] = (1 - self.alpha) * self.running[name] + self.alpha * v
        return value / (self.running[name] + self.eps)

def occupancy_mask(field, thresh=0.05):
    return (field > thresh).float()

def soft_occupancy(field, thresh=0.05, sharpness=6.0):
    return torch.sigmoid((field - thresh) * sharpness)

def dice_loss(pred_mask, gt_mask, eps=1e-6):
    inter = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return 1 - (2*inter + eps) / (union + eps)

def boundary_map(mask):
    # absolute gradient magnitude; cheap Sobel operator
    gx = F.conv2d(mask, torch.tensor([[[[-1,0,1],
                                        [-2,0,2],
                                        [-1,0,1]]]],dtype=torch.float32).to(mask.device), padding=1)
    gy = F.conv2d(mask, torch.tensor([[[[-1,-2,-1],
                                        [ 0, 0, 0],
                                        [ 1, 2, 1]]]],dtype=torch.float32).to(mask.device), padding=1)
    return torch.sqrt(gx**2 + gy**2)  # boundary strength

def boundary_loss(pred_mask, gt_mask):
    pred_b = boundary_map(pred_mask)
    gt_b   = boundary_map(gt_mask)
    return F.l1_loss(pred_b, gt_b)

def masked_density_loss(pred, pred_mask, gt, gt_mask):
    return F.mse_loss(pred * pred_mask, gt * gt_mask)
    # return F.mse_loss(pred,gt)

def output_loss(pred, gt, w_dice=1., w_boundary=1., w_density=1.):
    pred_mask = soft_occupancy(pred)
    gt_mask   = soft_occupancy(gt)

    l_dice = dice_loss(pred_mask, gt_mask)
    l_bound = boundary_loss(pred_mask, gt_mask)
    l_mse = masked_density_loss(pred, pred_mask, gt, gt_mask)
    # print(f"Dice Loss: {l_dice.item():.4f}, Boundary Loss: {l_bound.item():.4f}, Density Loss: {l_mse.item():.4f}")
    return (
        w_dice * l_dice +
        w_boundary * l_bound +
        w_density * l_mse
    )



def output_dice_loss(pred, gt):
    pred_mask = soft_occupancy(pred)
    gt_mask   = soft_occupancy(gt)
    
    l_dice = dice_loss(pred_mask, gt_mask)
    # print(f"Dice Loss: {l_dice.item():.4f}, Boundary Loss: {l_bound.item():.4f}, Density Loss: {l_mse.item():.4f}")
    return l_dice

def output_boundary_loss(pred, gt):
    pred_mask = soft_occupancy(pred)
    gt_mask   = soft_occupancy(gt)
    
    loss = boundary_loss(pred_mask, gt_mask)
    # print(f"Dice Loss: {l_dice.item():.4f}, Boundary Loss: {l_bound.item():.4f}, Density Loss: {l_mse.item():.4f}")
    return loss

from ToolUser.utils import visualize_physical_state, visualize_transition,visualize_transition_field, visualize_state_action_transition, visualize_transition_field_with_prediction
import matplotlib.pyplot as plt

def visualize_loss(pred,gt):
    fig, axs = plt.subplots(2,3, figsize=(15,10))
    visualize_transition_field_with_prediction(axs[0,0], pred, title="Prediction")
    visualize_transition_field(axs[0,1], gt, title="Ground Truth")
    visualize_transition_field_with_prediction(axs[0,2], pred-gt, title="Error (Prediction - GT)")
    
    pred_mask = soft_occupancy(pred)
    gt_mask   = soft_occupancy(gt)
    
    visualize_transition_field_with_prediction(axs[1,0], pred_mask, title="Predicted Occupancy Mask")
    visualize_transition_field_with_prediction(axs[1,1], gt_mask, title="Ground Truth Occupancy Mask")
    visualize_transition_field_with_prediction(axs[1,2], torch.abs(pred_mask - gt_mask), title="Occupancy Mask Error")
    
    plt.tight_layout()
    plt.show()

