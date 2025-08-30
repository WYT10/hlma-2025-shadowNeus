import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Define Ground Truth Object (Circle)
# ------------------------------
radius = 0.3
center = torch.tensor([0.0, 0.5])

def true_sdf(points):
    """Signed distance to circle."""
    return torch.linalg.norm(points - center, dim=-1) - radius

# ------------------------------
# 2. Light and Shadow Projection
# ------------------------------
light_pos = torch.tensor([0.0, 1.5])  # point light above
ground_y = 0.0

def project_shadow(points):
    """
    For points on ground (x,0), cast rays to light,
    check intersection with object boundary.
    """
    x = points[:, 0]
    ground_points = torch.stack([x, torch.zeros_like(x)], dim=-1)
    # Vector from light to ground
    dirs = ground_points - light_pos
    # Normalize
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

    # Sample along ray
    ts = torch.linspace(0, 2.0, 200)
    mask = []
    for gx, d in zip(ground_points, dirs):
        pts = light_pos + ts[:, None] * d
        inside = true_sdf(pts).min() < 0
        mask.append(float(inside))
    return torch.tensor(mask)

# Ground 1D samples
x_samples = torch.linspace(-1, 1, 200)[:, None]
shadow_obs = project_shadow(torch.cat([x_samples, torch.zeros_like(x_samples)], dim=-1))

# ------------------------------
# 3. Neural SDF Model
# ------------------------------
class SDFNet(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = []
        in_dim, out_dim = 2, 1
        dims = [in_dim] + [hidden]*depth + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

sdf_net = SDFNet()
optimizer = optim.Adam(sdf_net.parameters(), lr=1e-3)

# ------------------------------
# 4. Loss: Shadow + Eikonal
# ------------------------------
def eikonal_loss(points):
    points.requires_grad_(True)
    f = sdf_net(points)
    grads = torch.autograd.grad(
        f, points, torch.ones_like(f), create_graph=True
    )[0]
    return ((grads.norm(dim=-1) - 1)**2).mean()

def shadow_loss():
    # Predict shadow from learned sdf
    pred = project_shadow(torch.cat([x_samples, torch.zeros_like(x_samples)], dim=-1))
    return ((pred - shadow_obs)**2).mean()

# ------------------------------
# 5. Training Loop
# ------------------------------
for step in range(500):
    optimizer.zero_grad()
    pts = torch.rand(500, 2) * 2 - 1  # sample in [-1,1]^2
    loss_eik = eikonal_loss(pts)
    loss_shadow = shadow_loss()
    loss = loss_eik + 0.1 * loss_shadow
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}")

# ------------------------------
# 6. Visualization
# ------------------------------
grid_x, grid_y = torch.meshgrid(
    torch.linspace(-1, 1, 100),
    torch.linspace(-1, 1, 100),
    indexing="xy"
)
grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
with torch.no_grad():
    sdf_pred = sdf_net(grid).reshape(100,100)

plt.contourf(grid_x, grid_y, sdf_pred.numpy(), levels=50, cmap="RdBu")
plt.colorbar(label="Predicted SDF")
plt.contour(grid_x, grid_y, sdf_pred.numpy(), levels=[0], colors='k')  # learned shape boundary
plt.title("Learned 2D Shape from 1D Shadows")
plt.show()
