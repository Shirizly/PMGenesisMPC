import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

def gaussian_blur_torch(rho, sigma=1.0):
    size = int(2 * round(3*sigma) + 1)
    coords = torch.arange(size) - size//2
    grid = coords**2
    kernel_1d = torch.exp(-grid / (2*sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d[:,None] * kernel_1d[None,:]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

    rho_ = rho.unsqueeze(0).unsqueeze(0)
    rho_blur = F.conv2d(rho_, kernel_2d, padding=size//2)
    return rho_blur[0,0]

def differentiable_push(rho, p0, p1, width=5, spread_sigma=2.0):
    """
    [DEPRECATED — use differentiable_push_splat instead]
    Pull-based differentiable push using grid_sample.
    Does not correctly handle mass accumulation (many-to-one mapping).
    """
    H, W = rho.shape
    device = rho.device
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device),
                                    torch.arange(W, device=device),
                                    indexing='ij')
    px = torch.stack([x_grid.float(), y_grid.float()], dim=-1)
    d = p1 - p0
    L = torch.norm(d)
    if L.item() == 0:
        return rho.clone()
    d_unit = d / L
    d_perp = torch.tensor([-d_unit[1], d_unit[0]], device=device)
    rel0 = px - p0[None,None,:]
    rel1 = px - p1[None,None,:]
    x_from_p0 = torch.sum(rel0 * d_unit[None,None,:], dim=-1)
    x_from_p1 = torch.sum(rel1 * d_unit[None,None,:], dim=-1)
    y_local = torch.sum(rel1 * d_perp[None,None,:], dim=-1)
    sigma = 1.0
    lateral_mask = torch.sigmoid((width - torch.abs(y_local))/sigma)
    swept_mask = torch.sigmoid(x_from_p0/sigma) * torch.sigmoid((-x_from_p1 - sigma)/sigma) * lateral_mask
    accumulation_length = L
    dest_start_offset = sigma
    dest_range = accumulation_length - dest_start_offset
    swept_range = L - dest_start_offset
    dest_fraction = torch.clamp((x_from_p1 - dest_start_offset) / dest_range, 0.0, 1.0)
    source_x_from_p1 = -L + dest_fraction * swept_range
    offset_x = (source_x_from_p1 - x_from_p1) * d_unit[0]
    offset_y = (source_x_from_p1 - x_from_p1) * d_unit[1]
    sharp_sigma = 0.3
    dest_mask_sharp = torch.sigmoid((x_from_p1 - dest_start_offset)/sharp_sigma) * \
                      torch.sigmoid((accumulation_length - x_from_p1)/sharp_sigma) * \
                      torch.sigmoid((width - torch.abs(y_local))/sharp_sigma)
    x_source = dest_mask_sharp * (px[...,0] + offset_x) + (1 - dest_mask_sharp) * px[...,0]
    y_source = dest_mask_sharp * (px[...,1] + offset_y) + (1 - dest_mask_sharp) * px[...,1]
    rho_ = rho.unsqueeze(0).unsqueeze(0)
    x_norm = 2.0 * x_source / (W-1) - 1.0
    y_norm = 2.0 * y_source / (H-1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    rho_new = F.grid_sample(rho_, grid, mode='bilinear', padding_mode='border', align_corners=True)
    rho_new = rho_new[0,0]
    rho_new = rho_new * (1 - swept_mask)
    rho_new = torch.clamp(rho_new, 0.0, 1.0)
    return rho_new


def differentiable_push_splat(rho, p0, p1, width=5, sigma=1.0):
    """
    Differentiable approximation of push_baseline_discrete using soft masks
    and bilinear splatting.  Differentiable w.r.t. p0 and p1.

    Algorithm (mirrors the discrete version step-by-step):
      1. Compute tool-local coordinates (along-motion x_loc, lateral y_loc).
      2. Build a soft swept mask via sigmoids (replaces boolean thresholds).
      3. Extract mass = rho * swept;  clear swept region.
      4. Compute continuous destination at the front line (p1 + y_loc * d_perp).
      5. Deposit mass via bilinear splatting (index_put_ + accumulate),
         which correctly handles the many-to-one mapping that grid_sample cannot.

    Gradient paths w.r.t. p0, p1:
      • Through the soft mask  (which pixels are swept)
      • Through the bilinear weights  (where mass is deposited)

    Args:
        rho:   (H, W) torch float tensor – occupancy field
        p0:    (2,)   torch tensor [x, y] – tool start
        p1:    (2,)   torch tensor [x, y] – tool end
        width: half-width of tool in pixels (not differentiable)
        sigma: softness of mask edges in pixels (default 1.0;
               smaller ≈ sharper / closer to discrete, but weaker gradients)

    Returns:
        rho_new: (H, W) updated occupancy field
        swept:   (H, W) soft swept mask (useful for visualisation)
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    d = p1 - p0
    L = torch.norm(d)
    if L.item() < 1e-6:
        return rho.clone(), torch.zeros_like(rho)

    d_unit = d / L
    d_perp = torch.stack([-d_unit[1], d_unit[0]])

    # pixel grid  (H, W, 2)  with [...,0]=x, [...,1]=y
    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    # tool-local coordinates
    rel = px - p0                                  # (H, W, 2)
    x_loc = (rel * d_unit).sum(-1)                 # along motion
    y_loc = (rel * d_perp).sum(-1)                 # lateral

    # soft swept mask  (sigmoid approximation of indicator)
    swept = (torch.sigmoid(x_loc / sigma)
           * torch.sigmoid((L - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    # extract mass and clear swept region
    mass = rho * swept
    rho_cleared = rho * (1 - swept)

    # destination: front line at p1, preserving lateral offset
    dest_x = p1[0] + y_loc * d_perp[0]            # (H, W)
    dest_y = p1[1] + y_loc * d_perp[1]

    # ---- bilinear splatting ------------------------------------------------
    # For each pixel with mass, distribute to its 4 nearest integer neighbors
    # weighted by bilinear distance.  Gradients flow through the weights
    # (smooth in dest coords) and through the soft mask.
    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx = dest_x - x0f                              # fractional parts ∈ [0,1)
    fy = dest_y - y0f

    deposited = torch.zeros(H, W, device=device, dtype=dtype)

    for dy in (0, 1):
        for dx in (0, 1):
            wx = fx if dx else (1 - fx)
            wy = fy if dy else (1 - fy)
            xi = x0f.long() + dx
            yi = y0f.long() + dy

            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass

            # accumulate into a fresh tensor (keeps autograd graph clean)
            part = torch.zeros(H, W, device=device, dtype=dtype)
            if valid.any():
                part.index_put_((yi[valid], xi[valid]),
                                w[valid], accumulate=True)
            deposited = deposited + part

    rho_new = rho_cleared + deposited
    return rho_new, swept


def differentiable_push_splat_batch(rho, p0, p1, width=5, sigma=1.0):
    """
    Fully-batched, GPU-vectorised version of ``differentiable_push_splat``.

    Eliminates the Python ``for b in range(B)`` loop so that all B candidates
    are processed in a single set of tensor operations — one GPU kernel per
    arithmetic op instead of B serial kernel launches.  Gradient paths are
    identical to the scalar version:
      • swept mask  (sigmoid gates) → p0, p1
      • bilinear weights (fx, fy)   → dest_x, dest_y → p0, p1

    Args:
        rho: (B, H, W) torch float tensor – batch of occupancy fields.
        p0:  (B, 2)    torch tensor [x, y] – per-sample tool start.
        p1:  (B, 2)    torch tensor [x, y] – per-sample tool end.
        width, sigma: same semantics as ``differentiable_push_splat``.

    Returns:
        rho_new: (B, H, W) updated occupancy fields.
        swept:   (B, H, W) soft swept masks.
    """
    B, H, W = rho.shape
    device = rho.device
    dtype  = rho.dtype

    d      = p1 - p0                             # (B, 2)
    L      = torch.norm(d, dim=-1)               # (B,)
    zero   = L < 1e-6                            # (B,) – degenerate (zero-length) pushes
    L_safe = L.clamp(min=1e-6)

    d_unit = d / L_safe.unsqueeze(-1)            # (B, 2)
    d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)  # (B, 2)

    # shared pixel grid – built once, broadcast over B
    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)         # (H, W, 2)

    # (B, H, W, 2): displacement of every pixel from each push start
    rel   = px.unsqueeze(0) - p0.view(B, 1, 1, 2)
    x_loc = (rel * d_unit.view(B, 1, 1, 2)).sum(-1)               # (B, H, W) along-motion
    y_loc = (rel * d_perp.view(B, 1, 1, 2)).sum(-1)               # (B, H, W) lateral

    swept = (torch.sigmoid( x_loc / sigma)
           * torch.sigmoid((L_safe.view(B, 1, 1) - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))         # (B, H, W)

    mass        = rho * swept
    rho_cleared = rho * (1.0 - swept)

    # destination: deposit at p1, preserving lateral offset
    dest_x = p1[:, 0].view(B, 1, 1) + y_loc * d_perp[:, 0].view(B, 1, 1)
    dest_y = p1[:, 1].view(B, 1, 1) + y_loc * d_perp[:, 1].view(B, 1, 1)

    x0f = torch.floor(dest_x)                   # (B, H, W)
    y0f = torch.floor(dest_y)
    fx  = dest_x - x0f                          # fractional parts ∈ [0, 1)
    fy  = dest_y - y0f

    # Bilinear scatter into a flat (B, H*W) buffer.
    # Gradients flow through the bilinear weights (wx * wy) and through mass.
    # Integer indices carry no gradient — same as the scalar version's index_put_.
    deposited = torch.zeros(B, H, W, device=device, dtype=dtype)
    for dy_int in (0, 1):
        for dx_int in (0, 1):
            wx = fx       if dx_int else (1.0 - fx)   # (B, H, W)
            wy = fy       if dy_int else (1.0 - fy)
            xi = (x0f + dx_int).long()               # (B, H, W)  dest col
            yi = (y0f + dy_int).long()               # (B, H, W)  dest row
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass * valid.to(dtype)     # zero out out-of-bounds
            # clamp indices so scatter_add_ never writes out of bounds
            xi_c = xi.clamp(0, W - 1)
            yi_c = yi.clamp(0, H - 1)
            flat_idx = (yi_c * W + xi_c).view(B, H * W)   # (B, HW)
            w_flat   = w.view(B, H * W)
            # fresh buffer per corner keeps autograd graph clean
            part = torch.zeros(B, H * W, device=device, dtype=dtype)
            part.scatter_add_(1, flat_idx, w_flat)
            deposited = deposited + part.view(B, H, W)

    rho_new = rho_cleared + deposited

    # restore zero-length pushes (identity)
    if zero.any():
        rho_new = torch.where(zero.view(B, 1, 1), rho,           rho_new)
        swept   = torch.where(zero.view(B, 1, 1), torch.zeros_like(swept), swept)

    return rho_new, swept


def differentiable_push_spread(rho, p0, p1, width=5, sigma=1.0):
    """
    Differentiable push with linear proportional spread (Approach B).

    Like ``differentiable_push_splat`` but instead of depositing all swept
    mass at a single line at p1, spreads the pile forward proportionally:
    material originally near p0 is deposited farthest past p1, material
    near p1 stays close to p1.  The pile length equals total_mass / (2*width),
    which is exact for uniform-density input.

    Algorithm:
      1–3. Same as differentiable_push_splat (mask, extract, clear).
      4.   Compute pile_depth = total_mass / (2 * width).
      5.   Deposit position: p1 + pile_depth * (L - x_loc)/L * d_unit
                                + y_loc * d_perp
      6.   Bilinear splat.

    Gradient paths w.r.t. p0, p1:
      • Through the soft mask  (which pixels are swept)
      • Through the bilinear weights  (where mass is deposited)
      • Through pile_depth  (total swept mass → mask → p0, p1)

    Args:
        rho:   (H, W) torch float tensor – occupancy field
        p0:    (2,)   torch tensor [x, y] – tool start
        p1:    (2,)   torch tensor [x, y] – tool end
        width: half-width of tool in pixels
        sigma: softness of mask edges in pixels

    Returns:
        rho_new: (H, W) updated occupancy field
        swept:   (H, W) soft swept mask
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    d = p1 - p0
    L = torch.norm(d)
    if L.item() < 1e-6:
        return rho.clone(), torch.zeros_like(rho)

    d_unit = d / L
    d_perp = torch.stack([-d_unit[1], d_unit[0]])

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    rel = px - p0
    x_loc = (rel * d_unit).sum(-1)
    y_loc = (rel * d_perp).sum(-1)

    swept = (torch.sigmoid(x_loc / sigma)
           * torch.sigmoid((L - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass = rho * swept
    rho_cleared = rho * (1 - swept)

    # Pile depth: how far the pile extends past p1.
    # For uniform density this exactly fills the pile to occupancy = 1.
    total_mass = mass.sum()
    pile_depth = total_mass / (2 * width + 1e-8)

    # Linear spread: material near p0 (x_loc≈0) goes farthest,
    # material near p1 (x_loc≈L) stays at p1.
    # Clamp the fraction to [0, 1] so pixels outside the swept region
    # (which carry ~0 mass anyway via the soft mask) don't extrapolate.
    spread_frac = ((L - x_loc) / L).clamp(0.0, 1.0)
    forward_offset = pile_depth * spread_frac

    dest_x = p1[0] + forward_offset * d_unit[0] + y_loc * d_perp[0]
    dest_y = p1[1] + forward_offset * d_unit[1] + y_loc * d_perp[1]

    # Bilinear splatting
    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx = dest_x - x0f
    fy = dest_y - y0f

    deposited = torch.zeros(H, W, device=device, dtype=dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            wx = fx if dx else (1 - fx)
            wy = fy if dy else (1 - fy)
            xi = x0f.long() + dx
            yi = y0f.long() + dy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass
            part = torch.zeros(H, W, device=device, dtype=dtype)
            if valid.any():
                part.index_put_((yi[valid], xi[valid]),
                                w[valid], accumulate=True)
            deposited = deposited + part

    rho_new = rho_cleared + deposited
    return rho_new, swept


def differentiable_push_spread_batch(rho, p0, p1, width=5, sigma=1.0):
    """
    Fully-batched version of ``differentiable_push_spread``.

    Args:
        rho: (B, H, W) torch float tensor
        p0:  (B, 2)    torch tensor [x, y] – tool start
        p1:  (B, 2)    torch tensor [x, y] – tool end
        width, sigma: same as scalar version

    Returns:
        rho_new: (B, H, W) updated occupancy fields
        swept:   (B, H, W) soft swept masks
    """
    B, H, W = rho.shape
    device = rho.device
    dtype  = rho.dtype

    d      = p1 - p0
    L      = torch.norm(d, dim=-1)
    zero   = L < 1e-6
    L_safe = L.clamp(min=1e-6)

    d_unit = d / L_safe.unsqueeze(-1)
    d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    rel   = px.unsqueeze(0) - p0.view(B, 1, 1, 2)
    x_loc = (rel * d_unit.view(B, 1, 1, 2)).sum(-1)
    y_loc = (rel * d_perp.view(B, 1, 1, 2)).sum(-1)

    swept = (torch.sigmoid( x_loc / sigma)
           * torch.sigmoid((L_safe.view(B, 1, 1) - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass        = rho * swept
    rho_cleared = rho * (1.0 - swept)

    # Per-sample pile depth
    total_mass = mass.reshape(B, -1).sum(-1)                      # (B,)
    pile_depth = total_mass / (2 * width + 1e-8)                  # (B,)

    spread_frac = ((L_safe.view(B, 1, 1) - x_loc)
                   / L_safe.view(B, 1, 1)).clamp(0.0, 1.0)       # (B, H, W)
    forward_offset = pile_depth.view(B, 1, 1) * spread_frac      # (B, H, W)

    dest_x = (p1[:, 0].view(B, 1, 1)
              + forward_offset * d_unit[:, 0].view(B, 1, 1)
              + y_loc * d_perp[:, 0].view(B, 1, 1))
    dest_y = (p1[:, 1].view(B, 1, 1)
              + forward_offset * d_unit[:, 1].view(B, 1, 1)
              + y_loc * d_perp[:, 1].view(B, 1, 1))

    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx  = dest_x - x0f
    fy  = dest_y - y0f

    deposited = torch.zeros(B, H, W, device=device, dtype=dtype)
    for dy_int in (0, 1):
        for dx_int in (0, 1):
            wx = fx       if dx_int else (1.0 - fx)
            wy = fy       if dy_int else (1.0 - fy)
            xi = (x0f + dx_int).long()
            yi = (y0f + dy_int).long()
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass * valid.to(dtype)
            xi_c = xi.clamp(0, W - 1)
            yi_c = yi.clamp(0, H - 1)
            flat_idx = (yi_c * W + xi_c).view(B, H * W)
            w_flat   = w.view(B, H * W)
            part = torch.zeros(B, H * W, device=device, dtype=dtype)
            part.scatter_add_(1, flat_idx, w_flat)
            deposited = deposited + part.view(B, H, W)

    rho_new = rho_cleared + deposited

    if zero.any():
        rho_new = torch.where(zero.view(B, 1, 1), rho,           rho_new)
        swept   = torch.where(zero.view(B, 1, 1), torch.zeros_like(swept), swept)

    return rho_new, swept


def differentiable_push_cumulative(rho, p0, p1, width=5, sigma=1.0):
    """
    Differentiable push with cumulative-mass forward spread (Approach A).

    Like ``differentiable_push_splat`` but deposits each swept pixel at a
    position that accounts for the mass ahead of it along the push direction.
    This implements the snow-plow formula and is exact for arbitrary density
    distributions (not just uniform).

    Algorithm:
      1–3. Same as differentiable_push_splat (mask, extract, clear).
      4.   For each pixel, compute the cumulative mass ahead (between it and
           p1) by shifting the mass field backward in K discrete steps and
           accumulating via grid_sample.
      5.   Deposit at p1 + cum_ahead * d_unit + y_loc * d_perp.
      6.   Bilinear splat.

    Gradient paths w.r.t. p0, p1:
      • Through the soft mask  (which pixels are swept)
      • Through bilinear weights  (where mass is deposited)
      • Through cum_ahead  (grid_sample chain for each scan step)

    Args:
        rho:   (H, W) torch float tensor – occupancy field
        p0:    (2,)   torch tensor [x, y] – tool start
        p1:    (2,)   torch tensor [x, y] – tool end
        width: half-width of tool in pixels
        sigma: softness of mask edges in pixels

    Returns:
        rho_new: (H, W) updated occupancy field
        swept:   (H, W) soft swept mask
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    d = p1 - p0
    L = torch.norm(d)
    if L.item() < 1e-6:
        return rho.clone(), torch.zeros_like(rho)

    d_unit = d / L
    d_perp = torch.stack([-d_unit[1], d_unit[0]])

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    rel = px - p0
    x_loc = (rel * d_unit).sum(-1)
    y_loc = (rel * d_perp).sum(-1)

    swept = (torch.sigmoid(x_loc / sigma)
           * torch.sigmoid((L - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass = rho * swept                     # (H, W) mass to be pushed
    rho_cleared = rho * (1 - swept)

    # --- cumulative mass ahead via sequential shift-and-sum -----------------
    # K = number of scan steps (≈ push length in pixels, capped)
    K = max(1, int(L.item() + 0.5))
    K = min(K, max(H, W))

    # Precompute a grid_sample grid that shifts content backward along d_unit
    # by 1 pixel.  "Shift backward" = at each position, sample from
    # (pos + d_unit), which gives the value one step ahead.
    x_src = x_g + d_unit[0]
    y_src = y_g + d_unit[1]
    x_norm = 2.0 * x_src / (W - 1) - 1.0
    y_norm = 2.0 * y_src / (H - 1) - 1.0
    shift_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1,H,W,2)

    # shifted_mass[k] = mass field shifted backward by k steps, i.e.
    # shifted_mass[k](pos) = mass(pos + k * d_unit).
    # cum_ahead(pos) = sum_{k=1..K} shifted_mass[k](pos)
    #                = total mass between pos and pos + K*d_unit  (approx.)
    cum_ahead = torch.zeros(H, W, device=device, dtype=dtype)
    current = mass.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    for _ in range(K):
        current = F.grid_sample(
            current, shift_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True)
        cum_ahead = cum_ahead + current[0, 0]

    # Deposit position: p1 + cum_ahead * d_unit + y_loc * d_perp
    dest_x = p1[0] + cum_ahead * d_unit[0] + y_loc * d_perp[0]
    dest_y = p1[1] + cum_ahead * d_unit[1] + y_loc * d_perp[1]

    # Bilinear splatting
    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx = dest_x - x0f
    fy = dest_y - y0f

    deposited = torch.zeros(H, W, device=device, dtype=dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            wx = fx if dx else (1 - fx)
            wy = fy if dy else (1 - fy)
            xi = x0f.long() + dx
            yi = y0f.long() + dy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass
            part = torch.zeros(H, W, device=device, dtype=dtype)
            if valid.any():
                part.index_put_((yi[valid], xi[valid]),
                                w[valid], accumulate=True)
            deposited = deposited + part

    rho_new = rho_cleared + deposited
    return rho_new, swept


def differentiable_push_cumulative_batch(rho, p0, p1, width=5, sigma=1.0):
    """
    Fully-batched version of ``differentiable_push_cumulative``.

    Uses per-sample scan steps (K derived from push length). Falls back to
    the maximum K across the batch for the shared loop, with shorter pushes
    naturally zeroing out via grid_sample padding.

    Args:
        rho: (B, H, W) torch float tensor
        p0:  (B, 2)    torch tensor [x, y] – tool start
        p1:  (B, 2)    torch tensor [x, y] – tool end
        width, sigma: same as scalar version

    Returns:
        rho_new: (B, H, W) updated occupancy fields
        swept:   (B, H, W) soft swept masks
    """
    B, H, W = rho.shape
    device = rho.device
    dtype  = rho.dtype

    d      = p1 - p0
    L      = torch.norm(d, dim=-1)
    zero   = L < 1e-6
    L_safe = L.clamp(min=1e-6)

    d_unit = d / L_safe.unsqueeze(-1)
    d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    rel   = px.unsqueeze(0) - p0.view(B, 1, 1, 2)
    x_loc = (rel * d_unit.view(B, 1, 1, 2)).sum(-1)
    y_loc = (rel * d_perp.view(B, 1, 1, 2)).sum(-1)

    swept = (torch.sigmoid( x_loc / sigma)
           * torch.sigmoid((L_safe.view(B, 1, 1) - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass        = rho * swept
    rho_cleared = rho * (1.0 - swept)

    # --- cumulative mass ahead (batched) ---
    # Use the max push length across the batch for the loop bound.
    K = max(1, int(L.max().item() + 0.5))
    K = min(K, max(H, W))

    # Per-sample shift grids: shift backward along each sample's d_unit by 1 px.
    # grid_sample expects (B, H, W, 2) with values in [-1, 1].
    x_src = x_g.unsqueeze(0) + d_unit[:, 0].view(B, 1, 1)  # (B, H, W)
    y_src = y_g.unsqueeze(0) + d_unit[:, 1].view(B, 1, 1)
    x_norm = 2.0 * x_src / (W - 1) - 1.0
    y_norm = 2.0 * y_src / (H - 1) - 1.0
    shift_grid = torch.stack([x_norm, y_norm], dim=-1)       # (B, H, W, 2)

    cum_ahead = torch.zeros(B, H, W, device=device, dtype=dtype)
    current = mass.unsqueeze(1)                               # (B, 1, H, W)
    for _ in range(K):
        current = F.grid_sample(
            current, shift_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True)
        cum_ahead = cum_ahead + current[:, 0]

    # Deposit positions
    dest_x = (p1[:, 0].view(B, 1, 1)
              + cum_ahead * d_unit[:, 0].view(B, 1, 1)
              + y_loc * d_perp[:, 0].view(B, 1, 1))
    dest_y = (p1[:, 1].view(B, 1, 1)
              + cum_ahead * d_unit[:, 1].view(B, 1, 1)
              + y_loc * d_perp[:, 1].view(B, 1, 1))

    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx  = dest_x - x0f
    fy  = dest_y - y0f

    deposited = torch.zeros(B, H, W, device=device, dtype=dtype)
    for dy_int in (0, 1):
        for dx_int in (0, 1):
            wx = fx       if dx_int else (1.0 - fx)
            wy = fy       if dy_int else (1.0 - fy)
            xi = (x0f + dx_int).long()
            yi = (y0f + dy_int).long()
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass * valid.to(dtype)
            xi_c = xi.clamp(0, W - 1)
            yi_c = yi.clamp(0, H - 1)
            flat_idx = (yi_c * W + xi_c).view(B, H * W)
            w_flat   = w.view(B, H * W)
            part = torch.zeros(B, H * W, device=device, dtype=dtype)
            part.scatter_add_(1, flat_idx, w_flat)
            deposited = deposited + part.view(B, H, W)

    rho_new = rho_cleared + deposited

    if zero.any():
        rho_new = torch.where(zero.view(B, 1, 1), rho,           rho_new)
        swept   = torch.where(zero.view(B, 1, 1), torch.zeros_like(swept), swept)

    return rho_new, swept


def differentiable_push_spread2(rho, p0, p1, width=5, sigma=1.0, blur_sigma=0.0):
    """
    Differentiable push with destination-aware spread and optional blur.

    Extends ``differentiable_push_spread`` to handle occupied destination
    regions.  Two mechanisms reduce stacking:

    1.  **Destination extension** – before depositing, measure how much mass
        already exists in the landing zone (the rectangle past p1 where the
        pile would land).  The pile is extended forward by
        ``extra_depth = existing_mass / (2 * width)`` so swept material
        leapfrogs the pre-existing occupancy.

    2.  **Isotropic blur** (optional, ``blur_sigma > 0``) – after splatting,
        apply a small Gaussian blur *only* to the deposited mass before adding
        it back to the cleared field.  This simulates granular scatter and
        smooths residual peaks that the first-order extension misses
        (e.g. non-uniform lateral density at the destination).

    Gradient paths (all through standard ops — sigmoids, grid arithmetic,
    conv2d):
      • Swept mask        → p0, p1
      • Bilinear weights  → dest → p0, p1
      • pile_depth        → total swept mass → mask → p0, p1
      • extra_depth       → landing mask (soft sigmoid on pile_depth) → p0, p1

    Args:
        rho:        (H, W) torch float tensor – occupancy field
        p0:         (2,)   torch tensor [x, y] – tool start
        p1:         (2,)   torch tensor [x, y] – tool end
        width:      half-width of tool in pixels
        sigma:      softness of mask edges in pixels
        blur_sigma: if > 0, Gaussian σ (pixels) for post-deposit blur

    Returns:
        rho_new: (H, W) updated occupancy field
        swept:   (H, W) soft swept mask
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    d = p1 - p0
    L = torch.norm(d)
    if L.item() < 1e-6:
        return rho.clone(), torch.zeros_like(rho)

    d_unit = d / L
    d_perp = torch.stack([-d_unit[1], d_unit[0]])

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    # Tool-local coordinates relative to p0
    rel = px - p0
    x_loc = (rel * d_unit).sum(-1)
    y_loc = (rel * d_perp).sum(-1)

    swept = (torch.sigmoid(x_loc / sigma)
           * torch.sigmoid((L - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass = rho * swept
    rho_cleared = rho * (1 - swept)

    # Pile depth from swept mass
    total_mass = mass.sum()
    pile_depth = total_mass / (2 * width + 1e-8)

    # --- Destination extension: measure existing mass in landing zone --------
    # Landing zone: soft rectangle from p1 to p1 + pile_depth * d_unit,
    # width = 2 * width, centred on the push line.
    rel_p1 = px - p1
    x_from_p1 = (rel_p1 * d_unit).sum(-1)
    y_from_p1 = (rel_p1 * d_perp).sum(-1)

    landing = (torch.sigmoid(x_from_p1 / sigma)
             * torch.sigmoid((pile_depth - x_from_p1) / sigma)
             * torch.sigmoid((width - y_from_p1.abs()) / sigma))

    existing_in_zone = (rho_cleared * landing).sum()
    extra_depth = existing_in_zone / (2 * width + 1e-8)

    total_depth = pile_depth + extra_depth

    # Linear spread with extended distance
    spread_frac = ((L - x_loc) / L).clamp(0.0, 1.0)
    forward_offset = total_depth * spread_frac

    dest_x = p1[0] + forward_offset * d_unit[0] + y_loc * d_perp[0]
    dest_y = p1[1] + forward_offset * d_unit[1] + y_loc * d_perp[1]

    # Bilinear splatting
    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx = dest_x - x0f
    fy = dest_y - y0f

    deposited = torch.zeros(H, W, device=device, dtype=dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            wx = fx if dx else (1 - fx)
            wy = fy if dy else (1 - fy)
            xi = x0f.long() + dx
            yi = y0f.long() + dy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass
            part = torch.zeros(H, W, device=device, dtype=dtype)
            if valid.any():
                part.index_put_((yi[valid], xi[valid]),
                                w[valid], accumulate=True)
            deposited = deposited + part

    # Optional excess-only Gaussian blur: only blur the stacked portion
    # (occupancy > 1) so that regions at or below 1 stay sharp.
    rho_combined = rho_cleared + deposited
    if blur_sigma > 0:
        ks = max(3, int(2 * round(3 * blur_sigma) + 1))
        c = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        g1d = torch.exp(-c ** 2 / (2 * blur_sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d[:, None] * g1d[None, :]).reshape(1, 1, ks, ks)
        excess = F.relu(rho_combined - 1.0)
        excess_blurred = F.conv2d(
            excess.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2
        )[0, 0]
        rho_combined = (rho_combined - excess) + excess_blurred

    rho_new = rho_combined
    return rho_new, swept


def differentiable_push_spread2_batch(rho, p0, p1, width=5, sigma=1.0,
                                      blur_sigma=0.0):
    """
    Fully-batched version of ``differentiable_push_spread2``.

    Args:
        rho: (B, H, W) torch float tensor
        p0:  (B, 2)    torch tensor [x, y] – tool start
        p1:  (B, 2)    torch tensor [x, y] – tool end
        width, sigma, blur_sigma: same as scalar version

    Returns:
        rho_new: (B, H, W) updated occupancy fields
        swept:   (B, H, W) soft swept masks
    """
    B, H, W = rho.shape
    device = rho.device
    dtype  = rho.dtype

    d      = p1 - p0
    L      = torch.norm(d, dim=-1)
    zero   = L < 1e-6
    L_safe = L.clamp(min=1e-6)

    d_unit = d / L_safe.unsqueeze(-1)
    d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)

    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    px = torch.stack([x_g, y_g], dim=-1)

    # Tool-local coordinates relative to p0
    rel   = px.unsqueeze(0) - p0.view(B, 1, 1, 2)
    x_loc = (rel * d_unit.view(B, 1, 1, 2)).sum(-1)
    y_loc = (rel * d_perp.view(B, 1, 1, 2)).sum(-1)

    swept = (torch.sigmoid( x_loc / sigma)
           * torch.sigmoid((L_safe.view(B, 1, 1) - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    mass        = rho * swept
    rho_cleared = rho * (1.0 - swept)

    # Pile depth from swept mass
    total_mass = mass.reshape(B, -1).sum(-1)                      # (B,)
    pile_depth = total_mass / (2 * width + 1e-8)                  # (B,)

    # --- Destination extension -----------------------------------------------
    rel_p1  = px.unsqueeze(0) - p1.view(B, 1, 1, 2)
    x_fp1   = (rel_p1 * d_unit.view(B, 1, 1, 2)).sum(-1)         # (B,H,W)
    y_fp1   = (rel_p1 * d_perp.view(B, 1, 1, 2)).sum(-1)

    landing = (torch.sigmoid(x_fp1 / sigma)
             * torch.sigmoid((pile_depth.view(B, 1, 1) - x_fp1) / sigma)
             * torch.sigmoid((width - y_fp1.abs()) / sigma))

    existing = (rho_cleared * landing).reshape(B, -1).sum(-1)     # (B,)
    extra_depth = existing / (2 * width + 1e-8)                   # (B,)

    total_depth = pile_depth + extra_depth                        # (B,)

    # Linear spread with extended distance
    spread_frac = ((L_safe.view(B, 1, 1) - x_loc)
                   / L_safe.view(B, 1, 1)).clamp(0.0, 1.0)
    forward_offset = total_depth.view(B, 1, 1) * spread_frac

    dest_x = (p1[:, 0].view(B, 1, 1)
              + forward_offset * d_unit[:, 0].view(B, 1, 1)
              + y_loc * d_perp[:, 0].view(B, 1, 1))
    dest_y = (p1[:, 1].view(B, 1, 1)
              + forward_offset * d_unit[:, 1].view(B, 1, 1)
              + y_loc * d_perp[:, 1].view(B, 1, 1))

    x0f = torch.floor(dest_x)
    y0f = torch.floor(dest_y)
    fx  = dest_x - x0f
    fy  = dest_y - y0f

    deposited = torch.zeros(B, H, W, device=device, dtype=dtype)
    for dy_int in (0, 1):
        for dx_int in (0, 1):
            wx = fx       if dx_int else (1.0 - fx)
            wy = fy       if dy_int else (1.0 - fy)
            xi = (x0f + dx_int).long()
            yi = (y0f + dy_int).long()
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = wx * wy * mass * valid.to(dtype)
            xi_c = xi.clamp(0, W - 1)
            yi_c = yi.clamp(0, H - 1)
            flat_idx = (yi_c * W + xi_c).reshape(B, H * W)
            w_flat   = w.reshape(B, H * W)
            part = torch.zeros(B, H * W, device=device, dtype=dtype)
            part.scatter_add_(1, flat_idx, w_flat)
            deposited = deposited + part.view(B, H, W)

    # Optional excess-only Gaussian blur: only blur the stacked portion
    # (occupancy > 1) so that regions at or below 1 stay sharp.
    rho_combined = rho_cleared + deposited
    if blur_sigma > 0:
        ks = max(3, int(2 * round(3 * blur_sigma) + 1))
        c = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        g1d = torch.exp(-c ** 2 / (2 * blur_sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d[:, None] * g1d[None, :]).reshape(1, 1, ks, ks)
        excess = F.relu(rho_combined - 1.0)
        excess_blurred = F.conv2d(
            excess.unsqueeze(1), kernel, padding=ks // 2
        )[:, 0]
        rho_combined = (rho_combined - excess) + excess_blurred

    rho_new = rho_combined

    if zero.any():
        rho_new = torch.where(zero.view(B, 1, 1), rho,           rho_new)
        swept   = torch.where(zero.view(B, 1, 1), torch.zeros_like(swept), swept)

    return rho_new, swept


def differentiable_redistribute(rho, d_unit, max_iters=10):
    """
    Differentiable redistribution of overfull pixels along the push direction.

    Iteratively peels off excess mass (rho > 1) and shifts it forward by one
    pixel in the motion direction using grid_sample (bilinear).  Analogous to
    redistribute_overfull but fully differentiable w.r.t. rho, p0, p1.

    Args:
        rho:       (H, W) torch tensor
        d_unit:    (2,) unit vector [dx, dy] of the push direction
        max_iters: maximum spread distance in pixels

    Returns:
        rho_out: (H, W) redistributed field
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    # precompute sampling grid that shifts content forward by d_unit
    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')
    # to move content in +d_unit direction, pull from (pos - d_unit)
    x_norm = 2.0 * (x_g - d_unit[0]) / (W - 1) - 1.0
    y_norm = 2.0 * (y_g - d_unit[1]) / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)   # (1,H,W,2)

    rho_out = rho
    for _ in range(max_iters):
        excess = F.relu(rho_out - 1.0)
        if excess.sum().item() < 1e-6:
            break
        rho_out = rho_out - excess
        shifted = F.grid_sample(
            excess.unsqueeze(0).unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )[0, 0]
        rho_out = rho_out + shifted

    return rho_out


def fluid_push(rho, p0, p1, width=5, sigma=1.0,
               n_steps=20, decay=0.95, media_sharpness=5.0,
               blur_sigma=1.0, correct_divergence=False):
    """
    Velocity-field-based differentiable push model for granular media.

    Instead of moving mass explicitly, models the tool as a source of a
    velocity field that propagates through occupied regions and advects the
    density field, treating media as a fluid.

    Physics:
      1. The sweeping tool imposes velocity *d = p1 − p0* inside the swept
         region (rigid-body boundary condition).
      2. At each propagation step the velocity field is blurred (small
         isotropic Gaussian) then gated by a soft media-presence mask and
         multiplied by a decay factor.  This means velocity propagates
         through occupied regions but vanishes in empty space.
      3. Media is advected by the final velocity field via semi-Lagrangian
         back-tracing (``grid_sample``), then the swept region is cleared
         (rigid tool leaves no media behind).

    Differentiable w.r.t. p0 and p1 through:
      • The soft swept mask (which positions receive tool velocity)
      • The tool velocity *d* itself (magnitude and direction of the field)

    Args:
        rho:             (H, W) torch float tensor – occupancy field.
        p0:              (2,) torch tensor [x, y] – tool start.
        p1:              (2,) torch tensor [x, y] – tool end.
        width:           half-width of tool in pixels.
        sigma:           edge softness of the swept mask (pixels).
        n_steps:         propagation iterations (controls influence radius).
        decay:           per-step velocity attenuation (0–1).
        media_sharpness: steepness of the soft media-presence gate.
        blur_sigma:      Gaussian σ for the propagation blur kernel.
        correct_divergence: if True, scale the advected field to compensate
                            for the Jacobian of the velocity field, making
                            the advection step nearly mass-conserving.

    Returns:
        rho_new:  (H, W)    updated density field.
        v_field:  (H, W, 2) final velocity field [vx, vy].
        swept:    (H, W)    soft swept mask.
    """
    H, W = rho.shape
    device = rho.device
    dtype = rho.dtype

    d = p1 - p0
    L = torch.norm(d)
    if L.item() < 1e-6:
        return (rho.clone(),
                torch.zeros(H, W, 2, device=device, dtype=dtype),
                torch.zeros(H, W, device=device, dtype=dtype))

    d_unit = d / L
    d_perp = torch.stack([-d_unit[1], d_unit[0]])

    # pixel grid (H, W) – reused for coordinates and advection
    y_g, x_g = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij')

    # tool-local coordinates
    rel_x = x_g - p0[0]
    rel_y = y_g - p0[1]
    x_loc = rel_x * d_unit[0] + rel_y * d_unit[1]
    y_loc = rel_x * d_perp[0] + rel_y * d_perp[1]

    # soft swept mask
    swept = (torch.sigmoid(x_loc / sigma)
           * torch.sigmoid((L - x_loc) / sigma)
           * torch.sigmoid((width - y_loc.abs()) / sigma))

    # media-presence gate: high where rho is present, ~0 where empty
    media_gate = torch.sigmoid(media_sharpness * (rho - 0.1))

    # --- build velocity field ------------------------------------------------
    # Initial field: full tool displacement inside swept region
    v = swept.unsqueeze(-1) * d                      # (H, W, 2)

    # Gaussian blur kernel (depthwise: same kernel for vx and vy)
    ks = max(3, int(2 * round(2 * blur_sigma) + 1))
    if ks % 2 == 0:
        ks += 1
    c = torch.arange(ks, device=device, dtype=dtype) - ks // 2
    g1d = torch.exp(-c ** 2 / (2 * blur_sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = (g1d[:, None] * g1d[None, :]).reshape(1, 1, ks, ks)
    kernel_dw = kernel.expand(2, -1, -1, -1)          # (2, 1, ks, ks)
    pad = ks // 2

    gate = media_gate.unsqueeze(-1) * decay            # precompute gate·decay

    for _ in range(n_steps):
        # blur both velocity components in one depthwise conv
        vp = v.permute(2, 0, 1).unsqueeze(0)          # (1, 2, H, W)
        vb = F.conv2d(vp, kernel_dw, padding=pad, groups=2)
        v_blurred = vb[0].permute(1, 2, 0)            # (H, W, 2)

        # propagate: gated by media presence and decayed
        v_prop = v_blurred * gate

        # boundary condition: swept region keeps tool velocity
        sw = swept.unsqueeze(-1)
        v = sw * d + (1.0 - sw) * v_prop

    # --- divergence correction (optional) ------------------------------------
    # The Jacobian determinant of the backward map x → x − v(x) tells us
    # the local area change.  Multiplying by |J| compensates the stretching /
    # compression so the advection step conserves mass up to bilinear
    # interpolation error.
    if correct_divergence:
        # central finite differences for ∂vx/∂x and ∂vy/∂y
        dvx_dx = (torch.roll(v[..., 0], -1, 1) - torch.roll(v[..., 0], 1, 1)) / 2.0
        dvy_dy = (torch.roll(v[..., 1], -1, 0) - torch.roll(v[..., 1], 1, 0)) / 2.0
        # For the backward map  φ(x) = x − v(x),  J = I − ∇v,
        # det(J) ≈ 1 − (∂vx/∂x + ∂vy/∂y) to first order.
        det_J = 1.0 - (dvx_dx + dvy_dy)
        # Clamp to avoid negative / extreme corrections
        det_J = torch.clamp(det_J, 0.1, 10.0)

    # --- semi-Lagrangian advection -------------------------------------------
    source_x = x_g - v[..., 0]
    source_y = y_g - v[..., 1]
    x_norm = 2.0 * source_x / (W - 1) - 1.0
    y_norm = 2.0 * source_y / (H - 1) - 1.0
    adv_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    rho_advected = F.grid_sample(
        rho.unsqueeze(0).unsqueeze(0), adv_grid,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )[0, 0]

    if correct_divergence:
        rho_advected = rho_advected * det_J

    # clear swept region (rigid tool leaves empty space behind)
    rho_new = rho_advected * (1.0 - swept)

    return rho_new, v, swept


def velocity_field_divergence(v):
    """
    Compute the divergence of a 2D velocity field using central differences.

    Args:
        v: (H, W, 2) tensor with [...,0]=vx, [...,1]=vy

    Returns:
        div: (H, W) tensor  ∂vx/∂x + ∂vy/∂y
    """
    dvx_dx = (torch.roll(v[..., 0], -1, 1) - torch.roll(v[..., 0], 1, 1)) / 2.0
    dvy_dy = (torch.roll(v[..., 1], -1, 0) - torch.roll(v[..., 1], 1, 0)) / 2.0
    return dvx_dx + dvy_dy


def diagnose_mass_loss(rho, rho_advected, rho_new, swept, v):
    """
    Print a breakdown of where mass goes in fluid_push.

    Call with the intermediate values:
        rho          – input density
        rho_advected – density after grid_sample (before swept clearing)
        rho_new      – final output
        swept        – soft swept mask
        v            – velocity field

    To obtain rho_advected, temporarily save it inside fluid_push or
    replicate the advection step externally.
    """
    m_in = rho.sum().item()
    m_adv = rho_advected.sum().item()
    m_out = rho_new.sum().item()
    m_swept = (rho_advected * swept).sum().item()

    div = velocity_field_divergence(v)
    div_pos = div.clamp(min=0).sum().item()
    div_neg = div.clamp(max=0).sum().item()

    print("=== Mass loss diagnosis ===")
    print(f"  Input mass:                  {m_in:.4f}")
    print(f"  After advection (pre-clear): {m_adv:.4f}  "
          f"(Δ = {m_adv - m_in:+.4f}, {(m_adv-m_in)/m_in*100:+.2f}%)")
    print(f"  Cleared by swept mask:       {m_swept:.4f}")
    print(f"  Output mass:                 {m_out:.4f}  "
          f"(Δ = {m_out - m_in:+.4f}, {(m_out-m_in)/m_in*100:+.2f}%)")
    print(f"  Velocity divergence:         "
          f"pos={div_pos:.4f}  neg={div_neg:.4f}  net={div_pos+div_neg:.4f}")
    print(f"  |div| weighted by rho:       "
          f"{(div.abs() * rho).sum().item():.4f}")


def sweep_mask(p0, p1, H, W, width=5):
    """Hard boolean swept mask (for visualisation / comparison)."""
    d = p1 - p0
    L = torch.norm(d)
    if L.item() == 0:
        return torch.zeros((H, W), dtype=torch.bool)
    d_unit = d / L
    d_perp = torch.tensor([-d_unit[1], d_unit[0]])

    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    px = torch.stack([x_grid.float(), y_grid.float()], dim=-1)

    rel = px - p0[None, None, :]
    x_local = torch.sum(rel * d_unit[None, None, :], dim=-1)
    y_local = torch.sum(rel * d_perp[None, None, :], dim=-1)

    return (x_local >= 0) & (x_local <= L) & (torch.abs(y_local) <= width)

if __name__ == "__main__":
    from mass_push import push_baseline_discrete, redistribute_overfull

    tool_width = 10
    H, W = 50, 50
    scale_factor = 5.0

    # --- build occupancy field ---
    rho_np = np.zeros((H, W))
    particle_size = 1
    # random patches
    np.random.seed(0)
    for _ in range(150):
        x0, y0 = np.random.randint(20, 80, size=2)
        if any(rho_np[y0-particle_size:y0+particle_size, x0-particle_size:x0+particle_size].flatten() > 0):
            continue  # avoid overlap
        rho_np[y0-particle_size:y0+particle_size, x0-particle_size:x0+particle_size] += 1.0


    rho = torch.tensor(rho_np, dtype=torch.float32)
    rho = gaussian_blur_torch(rho, sigma=0.5)

    # --- action (tool start and end) ---
    p0_base = torch.tensor([1.0, 2.0]) * scale_factor
    p1_base = torch.tensor([6.0, 7.5]) * scale_factor

    # ===== 1. Discrete baseline =====
    p0_np, p1_np = p0_base.numpy(), p1_base.numpy()
    rho_discrete, M_discrete = push_baseline_discrete(
        rho.detach().numpy(), p0_np, p1_np, width=tool_width)
    rho_discrete = redistribute_overfull(rho_discrete, p0_np, p1_np,
                                         max_lateral=0, max_forward=10)

    # ===== 2. Splatting push =====
    p0_s = p0_base.clone().requires_grad_(True)
    p1_s = p1_base.clone().requires_grad_(True)
    t0 = time.time()
    rho_splat, swept_splat = differentiable_push_splat(rho, p0_s, p1_s,
                                                        width=tool_width, sigma=1.0)
    d_u = (p1_s - p0_s) / torch.norm(p1_s - p0_s)
    rho_splat_r = differentiable_redistribute(rho_splat, d_u, max_iters=10)
    t_splat = time.time() - t0
    loss_s = rho_splat_r[30:50, 30:50].sum()
    loss_s.backward()
    print(f"Splat  | time {t_splat:.4f}s | mass {rho_splat_r.sum().item():.2f}"
          f" | grad p0 {p0_s.grad.tolist()} | grad p1 {p1_s.grad.tolist()}")

    # ===== 3. Fluid push =====
    p0_f = p0_base.clone().requires_grad_(True)
    p1_f = p1_base.clone().requires_grad_(True)
    t0 = time.time()
    rho_fluid, v_field, swept_fluid = fluid_push(
        rho, p0_f, p1_f, width=tool_width, sigma=1.0,
        n_steps=20, decay=0.95, media_sharpness=5.0, blur_sigma=1.0)
    t_fluid = time.time() - t0
    loss_f = rho_fluid[30:50, 30:50].sum()
    loss_f.backward()
    print(f"Fluid  | time {t_fluid:.4f}s | mass {rho_fluid.sum().item():.2f}"
          f" | grad p0 {p0_f.grad.tolist()} | grad p1 {p1_f.grad.tolist()}")
    print(f"Discrete mass: {rho_discrete.sum():.2f}")

    # ===== 4. Mass-loss diagnosis =====
    # Re-run without grad to get rho_advected (before swept clearing)
    with torch.no_grad():
        _, v_diag, swept_diag = fluid_push(
            rho, p0_base, p1_base, width=tool_width, sigma=1.0,
            n_steps=20, decay=0.95, media_sharpness=5.0, blur_sigma=1.0)
        # replicate the advection step to get the intermediate rho_advected
        y_g, x_g = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32), indexing='ij')
        sx = x_g - v_diag[..., 0]
        sy = y_g - v_diag[..., 1]
        xn = 2.0 * sx / (W - 1) - 1.0
        yn = 2.0 * sy / (H - 1) - 1.0
        ag = torch.stack([xn, yn], dim=-1).unsqueeze(0)
        rho_adv = F.grid_sample(
            rho.unsqueeze(0).unsqueeze(0), ag,
            mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
    diagnose_mass_loss(rho, rho_adv, rho_fluid.detach(), swept_diag, v_diag)

    # ===== 5. Fluid push with divergence correction =====
    p0_fc = p0_base.clone().requires_grad_(True)
    p1_fc = p1_base.clone().requires_grad_(True)
    rho_fluid_c, v_fc, swept_fc = fluid_push(
        rho, p0_fc, p1_fc, width=tool_width, sigma=1.0,
        n_steps=20, decay=0.95, media_sharpness=5.0, blur_sigma=1.0,
        correct_divergence=True)
    loss_fc = rho_fluid_c[30:50, 30:50].sum()
    loss_fc.backward()
    print(f"\nFluid (corrected) | mass {rho_fluid_c.sum().item():.2f}"
          f" | grad p0 {p0_fc.grad.tolist()} | grad p1 {p1_fc.grad.tolist()}")

    # ===== visualise =====
    M_hard = sweep_mask(p0_base, p1_base, H, W, width=tool_width)
    v_mag = torch.norm(v_field, dim=-1).detach().numpy()

    fig, axs = plt.subplots(3, 4, figsize=(16, 11))

    # Row 0: Discrete baseline
    axs[0][0].imshow(rho.detach().numpy());     axs[0][0].set_title("Initial")
    axs[0][1].imshow(M_hard.numpy().astype(float)); axs[0][1].set_title("Swept (hard)")
    axs[0][2].imshow(rho_discrete);              axs[0][2].set_title("Discrete result")
    axs[0][3].axis('off')

    # Row 1: Splatting model
    axs[1][0].imshow(swept_splat.detach().numpy()); axs[1][0].set_title("Swept (soft)")
    axs[1][1].imshow(rho_splat.detach().numpy());   axs[1][1].set_title("Splat push")
    axs[1][2].imshow(rho_splat_r.detach().numpy()); axs[1][2].set_title("Splat + redist")
    diff_s = rho_splat_r.detach().numpy() - rho_discrete
    axs[1][3].imshow(diff_s, cmap='RdBu', vmin=-1, vmax=1)
    axs[1][3].set_title("Splat − Discrete")

    # Row 2: Fluid model
    axs[2][0].imshow(swept_fluid.detach().numpy()); axs[2][0].set_title("Swept (soft)")
    axs[2][1].imshow(v_mag);                         axs[2][1].set_title("Velocity |v|")
    axs[2][2].imshow(rho_fluid.detach().numpy());    axs[2][2].set_title("Fluid result")
    diff_f = rho_fluid.detach().numpy() - rho_discrete
    axs[2][3].imshow(diff_f, cmap='RdBu', vmin=-1, vmax=1)
    axs[2][3].set_title("Fluid − Discrete")

    for row in axs:
        for ax in row:
            ax.axis('off')
    fig.text(0.02, 0.83, "Discrete", fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.50, "Splatting", fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.17, "Fluid", fontsize=12, fontweight='bold', rotation=90, va='center')
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()