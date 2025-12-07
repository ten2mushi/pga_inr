import os
import copy
from typing import Callable, Dict, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pga_inr.core import OUTPUT_SDF
from pga_inr.models import PGA_INR_SDF
from pga_inr.losses import GeometricConsistencyLoss
from pga_inr.training import create_optimizer, CurriculumScheduler, LossWeightScheduler
from pga_inr.data import UniformSampler
from pga_inr.ops import SDFMorpher, smooth_min, smooth_max  # CSG operations
from pga_inr.utils.visualization import (
    plot_sdf_slice,
    plot_sdf_3d_isosurface,
    save_animation_gif,
)

@dataclass
class Config:
    """Training and visualization configuration."""
    # Training
    hidden_features: int = 128
    hidden_layers: int = 3
    omega_0: float = 15.0
    training_steps: int = 1000
    learning_rate: float = 1e-4
    num_samples: int = 8000
    num_surface_samples: int = 4000
    lambda_eikonal: float = 0.2

    # Visualization
    grid_resolution: int = 64
    animation_frames: int = 180
    animation_fps: int = 24
    bounds: Tuple[float, float] = (-1.5, 1.5)

    # Animation dynamics
    rotation_speed: float = 1.0
    translation_amplitude: float = 0.3
    scale_amplitude: float = 0.15

    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output
    output_dir: str = 'output'


CONFIG = Config()

def sd_box(p: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed distance to axis-aligned box."""
    q = torch.abs(p) - b
    return torch.norm(torch.relu(q), dim=-1) + torch.min(torch.max(q, dim=-1)[0], torch.zeros(1, device=p.device))


def sd_sphere(p: torch.Tensor, r: float = 0.5) -> torch.Tensor:
    """Signed distance to sphere."""
    return torch.norm(p, dim=-1) - r


def sd_torus(p: torch.Tensor, R: float = 0.5, r: float = 0.2) -> torch.Tensor:
    """Signed distance to torus."""
    q = torch.stack([torch.norm(p[..., :2], dim=-1) - R, p[..., 2]], dim=-1)
    return torch.norm(q, dim=-1) - r


PRIMITIVES = {
    'box': lambda p: sd_box(p, torch.tensor([0.4, 0.4, 0.4], device=p.device)),
    'sphere': lambda p: sd_sphere(p, r=0.5),
    'torus': lambda p: sd_torus(p, R=0.4, r=0.15),
}

def generate_sdf_data_with_library(
    sdf_fn: Callable,
    num_samples: int,
    device: torch.device,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    num_surface_samples: int = 0,
) -> Dict[str, torch.Tensor]:
    """Generate training data using library samplers."""
    # Use library's UniformSampler
    sampler = UniformSampler(bounds=bounds, device=device)
    uniform_points = sampler.sample(num_samples, batch_size=1).squeeze(0)

    all_points = [uniform_points]

    # Surface importance sampling (still custom - library sampler needs SDF function)
    if num_surface_samples > 0:
        candidate_count = num_surface_samples * 4
        candidates = sampler.sample(candidate_count, batch_size=1).squeeze(0)

        with torch.no_grad():
            candidate_sdf = sdf_fn(candidates).abs()

        # Keep points near surface
        surface_band = 0.15
        near_surface_mask = candidate_sdf < surface_band
        near_surface_points = candidates[near_surface_mask]

        if len(near_surface_points) >= num_surface_samples:
            indices = torch.randperm(len(near_surface_points))[:num_surface_samples]
            near_surface_points = near_surface_points[indices]

        if len(near_surface_points) > 0:
            all_points.append(near_surface_points)

    points = torch.cat(all_points, dim=0)
    points.requires_grad_(True)

    sdf = sdf_fn(points)
    normals = torch.autograd.grad(
        outputs=sdf,
        inputs=points,
        grad_outputs=torch.ones_like(sdf),
        create_graph=False,
    )[0]

    return {
        'points': points.detach(),
        'sdf': sdf.unsqueeze(-1).detach(),
        'normals': normals.detach(),
    }


def train_model_with_curriculum(
    sdf_fn: Callable,
    config: Config,
    name: str = 'primitive',
    use_curriculum: bool = True,
    verbose: bool = True,
) -> PGA_INR_SDF:
    """Train using library's CurriculumScheduler."""
    device = torch.device(config.device)

    model = PGA_INR_SDF(
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        omega_0=config.omega_0,
        geometric_init=False,  # Must be False for weight manipulation
    ).to(device)

    optimizer = create_optimizer(model, lr=config.learning_rate)
    loss_fn = GeometricConsistencyLoss(lambda_eikonal=config.lambda_eikonal)

    # Use library's curriculum scheduler
    if use_curriculum:
        curriculum = CurriculumScheduler(
            total_steps=config.training_steps,
            num_stages=3,
            warmup_steps=config.training_steps // 20,
        )
        loss_scheduler = LossWeightScheduler(
            sdf_weight=1.0,
            eikonal_weight=config.lambda_eikonal,
            normal_weight=0.1,
            total_steps=config.training_steps,
            schedule_type='cosine',
        )
    else:
        curriculum = None
        loss_scheduler = None

    # Initial data
    data = generate_sdf_data_with_library(
        sdf_fn, config.num_samples, device,
        bounds=config.bounds,
        num_surface_samples=config.num_surface_samples
    )

    iterator = tqdm(range(config.training_steps), desc=f'Training {name}', disable=not verbose)

    for step in iterator:
        optimizer.zero_grad()

        # Get curriculum parameters
        if curriculum:
            params = curriculum.get_stage_params(step)
            surface_ratio = params.get('surface_ratio', 0.5)
        else:
            surface_ratio = 0.5

        # Resample periodically with curriculum-adjusted ratios
        if step % 200 == 0 and step > 0:
            adjusted_surface_samples = int(config.num_surface_samples * (0.5 + surface_ratio))
            data = generate_sdf_data_with_library(
                sdf_fn, config.num_samples // 2, device,
                bounds=config.bounds,
                num_surface_samples=adjusted_surface_samples
            )

        points = data['points'].requires_grad_(True)
        outputs = model(points)

        # Get scheduled loss weights
        if loss_scheduler:
            weights = loss_scheduler.get_weights(step)
            # Temporarily adjust loss_fn weights
            loss_fn.lambda_eikonal = weights['eikonal']

        loss, metrics = loss_fn(outputs, data['sdf'], data['normals'])
        loss.backward()
        optimizer.step()

        if verbose:
            stage_str = f"S{curriculum.get_current_stage(step)}" if curriculum else ""
            iterator.set_postfix(loss=f'{loss.item():.5f}', stage=stage_str)

    return model

class WeightTransform:
    """Lightweight weight-space transformations."""

    @staticmethod
    def get_first_layer(model: PGA_INR_SDF) -> nn.Linear:
        return model.backbone.layers[0].linear

    @staticmethod
    def apply(
        model: PGA_INR_SDF,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        scale: float = None,
    ) -> None:
        """Apply transformation: f_new(x) = f_orig((x - t) @ R / s)"""
        layer = WeightTransform.get_first_layer(model)
        W_orig = layer.weight.data.clone()
        b = layer.bias.data
        W = W_orig.clone()

        if rotation is not None:
            W = W @ rotation.T
        if scale is not None:
            W = W / scale
        if translation is not None:
            R = rotation if rotation is not None else torch.eye(3, device=W.device)
            s = scale if scale is not None else 1.0
            t_transformed = (translation @ R) / s
            b = b - t_transformed @ W_orig.T

        layer.weight.data = W
        layer.bias.data = b

    @staticmethod
    def rotation_y(angle: float, device: torch.device) -> torch.Tensor:
        c, s = np.cos(angle), np.sin(angle)
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=torch.float32)

    @staticmethod
    def rotation_z(angle: float, device: torch.device) -> torch.Tensor:
        c, s = np.cos(angle), np.sin(angle)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device, dtype=torch.float32)


def render_triview(
    sdf_fn: Callable,
    config: Config,
    title: str = '',
    gt_fn: Callable = None,
) -> np.ndarray:
    """Render three orthogonal views."""
    device = torch.device(config.device)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    views = [('z', 0.0, 'Front (XY)'), ('x', 0.0, 'Side (YZ)'), ('y', 0.0, 'Top (XZ)')]

    for ax, (axis, val, name) in zip(axes, views):
        plot_sdf_slice(
            sdf_fn, resolution=config.grid_resolution,
            slice_axis=axis, slice_value=val,
            bounds=config.bounds, device=device, ax=ax,
            title=name, show_zero_contour=True,
        )

        if gt_fn:
            X, Y, gt = _compute_grid(gt_fn, config, device, axis, val)
            ax.contour(X, Y, gt, levels=[0], colors='red', linestyles='dashed', linewidths=2)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def _compute_grid(sdf_fn, config, device, axis, val):
    """Compute SDF on 2D grid."""
    x = torch.linspace(config.bounds[0], config.bounds[1], config.grid_resolution, device=device)
    X, Y = torch.meshgrid(x, x, indexing='xy')

    if axis == 'z':
        pts = torch.stack([X.flatten(), Y.flatten(), torch.full_like(X.flatten(), val)], dim=-1)
    elif axis == 'y':
        pts = torch.stack([X.flatten(), torch.full_like(X.flatten(), val), Y.flatten()], dim=-1)
    else:
        pts = torch.stack([torch.full_like(X.flatten(), val), X.flatten(), Y.flatten()], dim=-1)

    with torch.no_grad():
        sdf = sdf_fn(pts)
        if isinstance(sdf, dict):
            sdf = sdf['sdf']
        sdf = sdf.squeeze()

    return X.cpu().numpy(), Y.cpu().numpy(), sdf.reshape(config.grid_resolution, config.grid_resolution).cpu().numpy()


def create_transform_animation(
    model: PGA_INR_SDF,
    config: Config,
    name: str,
    gt_sdf: Callable = None,
) -> List[np.ndarray]:
    """Create transformation animation."""
    device = torch.device(config.device)
    frames = []

    for i in tqdm(range(config.animation_frames), desc=f'Rendering {name}'):
        transformed = copy.deepcopy(model)
        t = i / config.animation_frames * 2 * np.pi * config.rotation_speed

        rotation = WeightTransform.rotation_y(t, device)
        translation = torch.tensor([
            config.translation_amplitude * np.cos(t),
            config.translation_amplitude * 0.25 * np.sin(2 * t),
            config.translation_amplitude * np.sin(t),
        ], device=device, dtype=torch.float32)
        scale = 1.0 + config.scale_amplitude * np.sin(2 * t)

        WeightTransform.apply(transformed, rotation=rotation, translation=translation, scale=scale)

        # Transformed ground truth
        gt = None
        if gt_sdf:
            def gt(p, R=rotation, tr=translation, s=scale, orig=gt_sdf):
                return orig(((p - tr) @ R) / s)

        title = f'{name.capitalize()} Transform | Rot: {np.degrees(t):.0f}° | Scale: {scale:.2f}'
        frames.append(render_triview(lambda p: transformed(p), config, title, gt))

    return frames


def create_morph_animation_with_library(
    model_a: PGA_INR_SDF,
    model_b: PGA_INR_SDF,
    name_a: str,
    name_b: str,
    config: Config,
    sdf_a: Callable = None,
    sdf_b: Callable = None,
) -> List[np.ndarray]:
    """Create morphing animation using library's SDFMorpher."""
    frames = []

    # Use library's SDFMorpher - the correct approach for SIREN networks
    morpher = SDFMorpher(model_a, model_b, blend_mode='linear')

    n_half = config.animation_frames // 2

    for i in tqdm(range(config.animation_frames), desc=f'Morphing {name_a} <-> {name_b}'):
        # Ping-pong
        alpha = i / n_half if i < n_half else 1 - (i - n_half) / n_half

        def morphed(pts, a=alpha):
            return morpher(pts, a)

        # GT interpolation
        if sdf_a and sdf_b:
            def gt(pts, a=alpha):
                return (1 - a) * sdf_a(pts) + a * sdf_b(pts)
        else:
            gt = sdf_a if alpha < 0.5 else sdf_b

        title = f'SDF Morphing: {name_a} <-> {name_b} | alpha = {alpha:.2f}'
        frames.append(render_triview(morphed, config, title, gt))

    return frames


def create_csg_demo_animation(
    model_a: PGA_INR_SDF,
    model_b: PGA_INR_SDF,
    config: Config,
) -> List[np.ndarray]:
    """Demonstrate CSG operations using library's smooth_min/smooth_max."""
    device = torch.device(config.device)
    frames = []

    operations = [
        ('Union', lambda a, b: smooth_min(a, b, k=0.1)),
        ('Intersection', lambda a, b: smooth_max(a, b, k=0.1)),
        ('Subtraction', lambda a, b: smooth_max(a, -b, k=0.1)),
    ]

    for op_name, op_fn in operations:
        for i in tqdm(range(config.animation_frames // 3), desc=f'CSG {op_name}'):
            t = i / (config.animation_frames // 3) * 2 * np.pi

            # Animate model_b's position
            transformed_b = copy.deepcopy(model_b)
            offset = torch.tensor([0.3 * np.cos(t), 0.0, 0.3 * np.sin(t)], device=device, dtype=torch.float32)
            WeightTransform.apply(transformed_b, translation=offset)

            def csg_sdf(pts, ma=model_a, mb=transformed_b, fn=op_fn):
                with torch.no_grad():
                    sdf_a = ma(pts)['sdf']
                    sdf_b = mb(pts)['sdf']
                    return {'sdf': fn(sdf_a, sdf_b)}

            title = f'CSG {op_name} (smooth k=0.1)'
            frames.append(render_triview(csg_sdf, config, title))

    return frames

def main():
    """Run the library-native weight space manipulation demo."""

    print("=" * 70)
    print("Weight Space Manipulation (v2) - Using Library APIs")
    print("=" * 70)
    print(f"\nDevice: {CONFIG.device}")
    print(f"Output: {CONFIG.output_dir}\n")

    os.makedirs(CONFIG.output_dir, exist_ok=True)
    device = torch.device(CONFIG.device)

    print("\n[1/4] Training with CurriculumScheduler...")

    trained = {}
    for name in ['box', 'sphere', 'torus']:
        print(f"\n  Training {name} with curriculum...")
        trained[name] = train_model_with_curriculum(
            PRIMITIVES[name], CONFIG, name=name,
            use_curriculum=True, verbose=True
        )

    print("\n[2/4] Creating transformation animation...")

    frames = create_transform_animation(
        trained['box'], CONFIG, 'box', PRIMITIVES['box']
    )
    save_animation_gif(frames, os.path.join(CONFIG.output_dir, '02_transform_box.gif'), fps=CONFIG.animation_fps)
    print(f"  Saved: 02_transform_box.gif")

    print("\n[3/4] Creating morphing animation (using SDFMorpher)...")

    morph_frames = create_morph_animation_with_library(
        trained['box'], trained['sphere'],
        'box', 'sphere', CONFIG,
        PRIMITIVES['box'], PRIMITIVES['sphere']
    )
    save_animation_gif(morph_frames, os.path.join(CONFIG.output_dir, '02_morph_box_sphere.gif'), fps=CONFIG.animation_fps)
    print(f"  Saved: 02_morph_box_sphere.gif")

    print("\n[4/4] Creating CSG demo (using smooth_min/smooth_max)...")

    csg_frames = create_csg_demo_animation(trained['box'], trained['sphere'], CONFIG)
    save_animation_gif(csg_frames, os.path.join(CONFIG.output_dir, '02_csg_operations.gif'), fps=CONFIG.animation_fps)
    print(f"  Saved: 02_csg_operations.gif")

if __name__ == "__main__":
    main()
