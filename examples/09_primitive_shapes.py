"""
Example 09: Primitive Shapes & Dynamic Composition

Demonstrates:
1. Training separate PGA-INR models for basic 3D primitives (Sphere, Box, Cylinder).
2. Composing these primitives into complex scenes using CSG operations.
3. Animating the composed objects moving independently in the scene.

This example highlights the "Object-Oriented" nature of PGA-INR:
Each object is a separate neural field that can be transformed and combined at runtime.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from pga_inr.models import PGA_INR_SDF
from pga_inr.models.inr import compose_scenes
from pga_inr.losses import GeometricConsistencyLoss
from pga_inr.training import create_optimizer
from pga_inr.utils.visualization import (
    plot_sdf_slice,
    save_animation_gif
)
from pga_inr.utils.quaternion import quaternion_from_axis_angle

# =============================================================================
# 1. Analytical SDF Functions (Ground Truth for Training)
# =============================================================================

def sd_sphere(p, radius):
    """Signed distance to a sphere."""
    return p.norm(dim=-1) - radius

def sd_box(p, b):
    """
    Signed distance to a box.
    b: half-extents (x, y, z)
    """
    q = torch.abs(p) - b
    length = torch.norm(torch.relu(q), dim=-1)
    inside_dist = torch.min(torch.max(q, dim=-1)[0], torch.tensor(0.0).to(p.device))
    return length + inside_dist

def sd_cylinder(p, h, r):
    """
    Signed distance to a vertical cylinder.
    h: half-height
    r: radius
    """
    d_xz = torch.norm(p[:, [0, 2]], dim=-1) - r
    d_y = torch.abs(p[:, 1]) - h
    d = torch.stack([d_xz, d_y], dim=-1)
    outside_dist = torch.norm(torch.relu(d), dim=-1)
    inside_dist = torch.min(torch.max(d, dim=-1)[0], torch.tensor(0.0).to(p.device))
    return outside_dist + inside_dist

# =============================================================================
# 2. Data Generation
# =============================================================================

def generate_primitive_data(primitive_type, num_samples=10000, device='cpu'):
    """Generate training data for a specific primitive."""
    
    # Points in [-1, 1] box
    points = torch.rand(num_samples, 3, device=device) * 2 - 1
    
    if primitive_type == 'sphere':
        sdf = sd_sphere(points, radius=0.5)
    elif primitive_type == 'box':
        sdf = sd_box(points, b=torch.tensor([0.4, 0.4, 0.4], device=device))
    elif primitive_type == 'cylinder':
        sdf = sd_cylinder(points, h=0.5, r=0.3)
    else:
        raise ValueError(f"Unknown primitive: {primitive_type}")
        
    # Compute normals (gradient of SDF)
    points.requires_grad_(True)
    # Re-compute SDF with graph for gradients
    if primitive_type == 'sphere':
        y = sd_sphere(points, radius=0.5)
    elif primitive_type == 'box':
        y = sd_box(points, b=torch.tensor([0.4, 0.4, 0.4], device=device))
    elif primitive_type == 'cylinder':
        y = sd_cylinder(points, h=0.5, r=0.3)
        
    normals = torch.autograd.grad(
        outputs=y,
        inputs=points,
        grad_outputs=torch.ones_like(y),
        create_graph=False
    )[0]
    
    # Detach everything
    return {
        'points': points.detach(),
        'sdf': sdf.unsqueeze(-1).detach(),
        'normals': normals.detach()
    }

# =============================================================================
# 3. Training Helper
# =============================================================================

def train_primitive_model(name, primitive_type, device, steps=500):
    """Train a quick PGA-INR model for a primitive."""
    print(f"Training {name} ({primitive_type})...")
    
    model = PGA_INR_SDF(
        hidden_features=128,
        hidden_layers=3,
        omega_0=30.0,
        geometric_init=True
    ).to(device)
    
    optimizer = create_optimizer(model, lr=1e-4)
    loss_fn = GeometricConsistencyLoss(lambda_eikonal=0.1)
    
    data = generate_primitive_data(primitive_type, num_samples=5000, device=device)
    
    for i in range(steps):
        optimizer.zero_grad()
        
        # Resample data every 100 steps for better coverage
        if i % 100 == 0:
            data = generate_primitive_data(primitive_type, num_samples=2000, device=device)
        
        points = data['points'].requires_grad_(True)
        gt_sdf = data['sdf']
        gt_normals = data['normals']
        
        outputs = model(points)
        
        loss, _ = loss_fn(outputs, gt_sdf, gt_normals)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")
            
    return model

# =============================================================================
# 4. Demos
# =============================================================================

def demo_composition(models, device):
    """Show static composition of the trained primitives."""
    print("\n" + "="*60)
    print("Static Scene Composition")
    print("="*60)
    
    sphere, box, cylinder = models['sphere'], models['box'], models['cylinder']
    
    # Define poses
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    
    # 1. Union: Sphere + Box
    pose_sphere = (torch.tensor([-0.4, 0.0, 0.0], device=device), identity_quat)
    pose_box = (torch.tensor([0.4, 0.0, 0.0], device=device), identity_quat)
    
    def union_scene(p):
        return compose_scenes(p, [sphere, box], [pose_sphere, pose_box], 'union')
        
    # 2. Subtraction: Box - Cylinder (Hole)
    pose_box_center = (torch.tensor([0.0, 0.0, 0.0], device=device), identity_quat)
    pose_cyl_center = (torch.tensor([0.0, 0.0, 0.0], device=device), identity_quat)
    
    def subtraction_scene(p):
        return compose_scenes(p, [box, cylinder], [pose_box_center, pose_cyl_center], 'subtraction')
        
    # 3. Smooth Union: Three objects melting together
    pose_s = (torch.tensor([0.0, 0.4, 0.0], device=device), identity_quat)
    pose_b = (torch.tensor([-0.3, -0.3, 0.0], device=device), identity_quat)
    pose_c = (torch.tensor([0.3, -0.3, 0.0], device=device), identity_quat)
    
    def smooth_scene(p):
        return compose_scenes(p, [sphere, box, cylinder], [pose_s, pose_b, pose_c], 'smooth_union', blend_k=8.0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_sdf_slice(union_scene, resolution=100, ax=axes[0], title="Union: Sphere + Box", device=device)
    plot_sdf_slice(subtraction_scene, resolution=100, ax=axes[1], title="Subtract: Box - Cyl", device=device)
    plot_sdf_slice(smooth_scene, resolution=100, ax=axes[2], title="Smooth Union (Melting)", device=device)
    
    plt.tight_layout()
    plt.savefig("09_composition_static.png")
    print("Saved 09_composition_static.png")
    # plt.show() # Uncomment to show interactive window

def demo_dynamic_animation(models, device):
    """Animate objects moving and interacting."""
    print("\n" + "="*60)
    print("Dynamic Animation Generation")
    print("="*60)
    
    sphere, box, cylinder = models['sphere'], models['box'], models['cylinder']
    
    frames = []
    num_frames = 40
    
    print(f"Generating {num_frames} frames...")
    
    for i in range(num_frames):
        t = i / num_frames * 2 * np.pi  # 0 to 2pi
        
        # --- Define Trajectories ---
        
        # Sphere: Orbits in a circle
        sphere_pos = torch.tensor([0.6 * np.cos(t), 0.6 * np.sin(t), 0.0], device=device, dtype=torch.float32)
        sphere_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        
        # Box: Rotates in place at center
        box_pos = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        # Rotate around Z axis
        axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        angle = torch.tensor([[t]], device=device, dtype=torch.float32) # Full rotation
        box_quat = quaternion_from_axis_angle(axis, angle).squeeze()
        
        # Cylinder: Moves up and down (linear)
        cyl_pos = torch.tensor([0.0, 0.6 * np.sin(t * 2), 0.0], device=device, dtype=torch.float32) # Faster bounce
        # Rotate cylinder 90 deg to be horizontal
        axis_rot = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        angle_rot = torch.tensor([[np.pi/2]], device=device, dtype=torch.float32)
        cyl_quat = quaternion_from_axis_angle(axis_rot, angle_rot).squeeze()
        
        # --- Compose Scene ---
        # We will use Smooth Union to show them "interacting" like liquid when they touch
        
        def frame_scene(p):
            return compose_scenes(
                p, 
                [sphere, box, cylinder], 
                [(sphere_pos, sphere_quat), (box_pos, box_quat), (cyl_pos, cyl_quat)], 
                operation='smooth_union',
                blend_k=5.0
            )
        
        # Render frame (SDF slice)
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_sdf_slice(frame_scene, resolution=128, ax=ax, bounds=(-1.2, 1.2), show_zero_contour=True, device=device)
        ax.set_title(f"Frame {i}/{num_frames}")
        
        # Convert plot to image
        fig.canvas.draw()
        
        # Use buffer_rgba() which is supported in modern matplotlib
        # Returns (H, W, 4) uint8 array
        image = np.asarray(fig.canvas.buffer_rgba())
        
        # Drop Alpha channel to get RGB
        image = image[:, :, :3]
        
        frames.append(image)
        plt.close(fig)
        
        if i % 10 == 0:
            print(f"  Rendered frame {i}")
            
    # Save GIF
    save_path = "09_dynamic_scene.gif"
    try:
        save_animation_gif(frames, save_path, fps=15)
        print(f"Saved animation to {save_path}")
    except ImportError:
        print("Could not save GIF (imageio not installed).")

def main():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Train Models
    models = {}
    models['sphere'] = train_primitive_model("Sphere", "sphere", device)
    models['box'] = train_primitive_model("Box", "box", device)
    models['cylinder'] = train_primitive_model("Cylinder", "cylinder", device)
    
    # 2. Show Static Composition
    demo_composition(models, device)
    
    # 3. Show Dynamic Animation
    demo_dynamic_animation(models, device)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
