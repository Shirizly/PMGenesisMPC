import matplotlib.pyplot as plt

device = "cpu"

import torch


def make_grid(H, W, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # (H,W,2)


def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def soft_halfplane(edge_a, edge_b, points, temperature=0.02):
    """
    edge from a -> b
    inside is left side
    """

    edge = edge_b - edge_a
    rel = points - edge_a

    signed = cross2d(edge, rel)

    return torch.sigmoid(signed / temperature)


def differentiable_parallelogram(
    points,
    center,
    width,
    height,
    skew,
    angle=0.0,
    temperature=0.02,
):
    """
    points: (...,2)

    returns occupancy in [0,1]
    """

    c = torch.cos(angle)
    s = torch.sin(angle)

    R = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c]),
    ])

    # local corners
    p0 = torch.tensor([-width, -height])
    p1 = torch.tensor([ width, -height])
    p2 = torch.tensor([ width + skew, height])
    p3 = torch.tensor([-width + skew, height])

    corners = torch.stack([p0, p1, p2, p3]).to(points.device)

    # rotate
    corners = corners @ R.T

    # translate
    corners = corners + center

    # edges
    occ = 1.0

    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]

        occ = occ * soft_halfplane(
            a,
            b,
            points,
            temperature,
        )

    return occ

grid = make_grid(256, 256, device)

occ = differentiable_parallelogram(
    grid,
    center=torch.tensor([0.0, 0.0]),
    width=0.1,
    height=0.4,
    skew=0.2,
    angle=torch.tensor(0.),
    temperature=0.01,
)

plt.imshow(occ.detach().numpy(), cmap="gray")
plt.show()