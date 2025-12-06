"""
Example 10: FBX Character Loading and Articulated Neural Fields

Demonstrates the complete pipeline for loading rigged characters from FBX:
1. Loading mesh, skeleton, skinning weights, and animations from FBX files
2. Creating KinematicChain from FBX skeleton
3. Applying LBS/DQS skinning with animations
4. Training an SDF on the canonical (T-pose) mesh
5. Creating ArticulatedNeuralField with skeleton-driven deformation
6. Visualizing animation sequences with multi-view comparison

Input files:
- input/3d_meshes/x_bot_t_pose.fbx - T-pose mesh with skeleton and skinning
- input/animations/x_bot_walking.fbx - Walking animation
- input/animations/x_bot_breakdance_1990.fbx - Breakdance animation

Output files:
- output/10_skeleton.png - Skeleton hierarchy visualization
- output/10_skinning_weights.png - Skinning weight distribution
- output/10_rest_pose.png - T-pose mesh
- output/10_walking.gif - Walking animation (original mesh)
- output/10_breakdance.gif - Breakdance animation (original mesh)
- output/10_skinning_comparison.png - LBS vs DQS skinning comparison
- output/10_canonical_sdf.png - SDF slices of canonical mesh
- output/10_original_vs_neural.png - Static comparison at a single pose
- output/10_comparison_multiview.gif - All animations with 6-panel comparison
    (Original vs Neural from front/side/top views)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# PGA-INR imports
from pga_inr.data import (
    load_rigged_character,
    CanonicalMeshDataset,
)
from pga_inr.data.mesh_utils import sdf_to_mesh, transfer_skinning_weights
from pga_inr.models import PGA_INR_SDF
from pga_inr.losses import GeometricConsistencyLoss
from pga_inr.training import create_optimizer
from pga_inr.utils.visualization import plot_sdf_slice, save_animation_gif
from pga_inr.spacetime.kinematic_chain import LinearBlendSkinning


# =============================================================================
# 1. Configuration
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MESH_PATH = PROJECT_ROOT / "input/3d_meshes/x_bot_t_pose.fbx"
ANIMATION_PATHS = [
    PROJECT_ROOT / "input/animations/x_bot_walking.fbx",
    PROJECT_ROOT / "input/animations/x_bot_breakdance_1990.fbx",
]
OUTPUT_DIR = PROJECT_ROOT / "output"


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# 2. Character Loading
# =============================================================================

def load_character(device):
    """Load the FBX character with skeleton, skinning, and animations."""
    print("=" * 60)
    print("Phase 1: Loading FBX Character")
    print("=" * 60)

    if not MESH_PATH.exists():
        raise FileNotFoundError(
            f"T-pose mesh not found: {MESH_PATH}\n"
            f"Please ensure x_bot_t_pose.fbx is in input/3d_meshes/"
        )

    # Filter to existing animation files
    anim_paths = [p for p in ANIMATION_PATHS if p.exists()]
    if not anim_paths:
        print("  Warning: No animation files found, proceeding without animations")

    # Load character
    character = load_rigged_character(
        mesh_path=str(MESH_PATH),
        animation_paths=[str(p) for p in anim_paths] if anim_paths else None,
        device=device,
        normalize=True,
        max_bones=4,
        sample_fps=30.0
    )

    # Print info
    print(f"\nMesh:")
    print(f"  Vertices: {character['metadata']['num_vertices']}")
    print(f"  Faces: {character['metadata']['num_faces']}")

    print(f"\nSkeleton:")
    print(f"  Joints: {character['metadata']['num_joints']}")
    if character['joint_names']:
        for i, name in enumerate(character['joint_names'][:10]):
            print(f"    {i}: {name}")
        if len(character['joint_names']) > 10:
            print(f"    ... and {len(character['joint_names']) - 10} more")

    print(f"\nSkinning:")
    print(f"  Max bones per vertex: {character['bone_weights'].shape[1]}")

    print(f"\nAnimations: {character['metadata']['num_animations']}")
    if character['animations']:
        for name, anim in character['animations'].items():
            print(f"  {name}: {anim.duration:.2f}s, {anim.fps} fps, "
                  f"{anim.keyframe_rotations.shape[0]} frames")

    return character


# =============================================================================
# 3. Visualization Functions
# =============================================================================

def visualize_skeleton(character, output_dir):
    """Visualize the skeleton hierarchy."""
    print("\n" + "=" * 60)
    print("Visualizing Skeleton")
    print("=" * 60)

    chain = character['kinematic_chain']
    if chain is None:
        print("  No skeleton to visualize")
        return

    # Get rest pose joint positions
    transforms = chain.forward_kinematics()
    joint_positions = {}
    for name in chain.joint_order:
        trans, _ = transforms[name]
        joint_positions[name] = trans.detach().cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints and bones
    for name, pos in joint_positions.items():
        joint = chain.joints[name]

        # Plot joint
        ax.scatter(pos[0], pos[1], pos[2], c='blue', s=50)
        ax.text(pos[0], pos[1], pos[2], f"  {name[:12]}", fontsize=5)

        # Plot bone to parent
        if joint.parent:
            parent_pos = joint_positions[joint.parent.name]
            ax.plot(
                [parent_pos[0], pos[0]],
                [parent_pos[1], pos[1]],
                [parent_pos[2], pos[2]],
                'k-', linewidth=2
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Skeleton Hierarchy (Rest Pose)')

    # Equal aspect ratio
    max_range = 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    save_path = output_dir / "10_skeleton.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def visualize_skinning_weights(character, output_dir):
    """Visualize skinning weight distribution."""
    print("\n" + "=" * 60)
    print("Visualizing Skinning Weights")
    print("=" * 60)

    verts = character['rest_vertices'].cpu().numpy()
    weights = character['bone_weights'].cpu().numpy()
    indices = character['bone_indices'].cpu().numpy()

    fig = plt.figure(figsize=(16, 8))

    # Weight distribution histogram
    ax1 = fig.add_subplot(121)
    max_weights = weights.max(axis=1)
    ax1.hist(max_weights, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Maximum Weight per Vertex')
    ax1.set_ylabel('Count')
    ax1.set_title('Skinning Weight Distribution')
    ax1.grid(True, alpha=0.3)

    # 3D visualization colored by primary bone
    ax2 = fig.add_subplot(122, projection='3d')
    primary_bone = indices[:, 0]

    # Subsample for visualization
    step = max(1, len(verts) // 5000)
    scatter = ax2.scatter(
        verts[::step, 0], verts[::step, 1], verts[::step, 2],
        c=primary_bone[::step], cmap='tab20', s=1, alpha=0.6
    )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Vertices Colored by Primary Bone')
    plt.colorbar(scatter, ax=ax2, label='Bone Index', shrink=0.6)

    plt.tight_layout()
    save_path = output_dir / "10_skinning_weights.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def visualize_rest_pose(character, output_dir):
    """Visualize the T-pose mesh."""
    print("\n" + "=" * 60)
    print("Visualizing Rest Pose (T-Pose)")
    print("=" * 60)

    verts = character['rest_vertices'].cpu().numpy()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for visualization
    step = max(1, len(verts) // 5000)
    ax.scatter(
        verts[::step, 0], verts[::step, 1], verts[::step, 2],
        c=verts[::step, 1], cmap='viridis', s=1, alpha=0.6
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rest Pose (T-Pose) Mesh')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.tight_layout()
    save_path = output_dir / "10_rest_pose.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def visualize_animation(character, anim_name, output_dir, num_frames=30, display_name=None):
    """Generate animation GIF for a specific animation.

    Args:
        character: Character dict from load_rigged_character
        anim_name: Animation name key in character['animations']
        output_dir: Output directory path
        num_frames: Number of frames to render
        display_name: Optional clean name for display/filename (defaults to anim_name)
    """
    display = display_name or anim_name
    print(f"\n  Generating animation: {display}")

    if anim_name not in character['animations']:
        print(f"    Animation '{anim_name}' not found, skipping")
        return

    anim = character['animations'][anim_name]
    chain = character['kinematic_chain']
    lbs = character['lbs']

    if chain is None or lbs is None:
        print("    No skeleton/skinning available")
        return

    # Sample frames evenly
    total_frames = anim.keyframe_rotations.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

    frames = []
    for frame_idx in tqdm(frame_indices, desc=f"    Rendering {display}"):
        rotations = anim.keyframe_rotations[frame_idx]
        root_trans = anim.root_translations[frame_idx]

        # Apply skinning
        transforms = chain.forward_kinematics(rotations, root_trans)
        deformed = lbs(transforms).detach().cpu().numpy()

        # Render frame
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample vertices
        step = max(1, len(deformed) // 3000)
        ax.scatter(
            deformed[::step, 0],
            deformed[::step, 1],
            deformed[::step, 2],
            c=deformed[::step, 1], cmap='viridis', s=1, alpha=0.6
        )

        t = frame_idx / max(total_frames - 1, 1) * anim.duration
        ax.set_title(f'{display}: t = {t:.2f}s')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Render to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(image)
        plt.close(fig)

    # Save GIF with clean filename
    clean_name = display.replace(':', '_').replace(' ', '_').replace('.', '_').lower()
    save_path = output_dir / f"10_{clean_name}.gif"
    try:
        save_animation_gif(frames, str(save_path), fps=15)
        print(f"    Saved: {save_path}")
    except Exception as e:
        print(f"    Failed to save GIF: {e}")


def visualize_all_animations(character, output_dir):
    """Generate GIFs for all animations."""
    print("\n" + "=" * 60)
    print("Generating Animation GIFs")
    print("=" * 60)

    if not character['animations']:
        print("  No animations to visualize")
        return

    # Map internal names to clean display names
    name_mapping = {
        'mixamo.com': 'walking',
        'x_bot_walking_mixamo.com': 'walking',
        'x_bot_breakdance_1990_mixamo.com': 'breakdance',
    }

    for anim_name in character['animations'].keys():
        # Try to extract a clean name from the animation name
        display_name = name_mapping.get(anim_name)
        if display_name is None:
            # Extract from filename pattern if possible
            if 'walking' in anim_name.lower():
                display_name = 'walking'
            elif 'breakdance' in anim_name.lower():
                display_name = 'breakdance'
            else:
                display_name = anim_name

        visualize_animation(character, anim_name, output_dir, display_name=display_name)


# =============================================================================
# 4. Training Canonical SDF
# =============================================================================

def train_canonical_sdf(character, device, epochs=500, batch_size=5000):
    """Train a PGA-INR SDF on the canonical (T-pose) mesh."""
    print("\n" + "=" * 60)
    print("Phase 2: Training Canonical SDF")
    print("=" * 60)

    # Create dataset
    # Note: cache_size controls memory usage during SDF computation
    # 20000 points provides good coverage while keeping memory reasonable
    dataset = CanonicalMeshDataset(
        mesh=character['mesh'],
        num_samples=batch_size,
        surface_ratio=0.5,
        bounds=(-1.0, 1.0),
        cache_size=20000,
        device=device
    )

    # Create model
    model = PGA_INR_SDF(
        hidden_features=256,
        hidden_layers=4,
        omega_0=30.0,
        geometric_init=True
    ).to(device)

    # Training setup
    optimizer = create_optimizer(model, lr=1e-4)
    loss_fn = GeometricConsistencyLoss(lambda_eikonal=0.1, lambda_align=0.05)

    # Training loop
    print(f"\n  Training for {epochs} epochs...")
    losses = []

    for epoch in range(epochs):
        batch = dataset[0]  # Random batch from cache
        points = batch['points'].to(device).requires_grad_(True)
        gt_sdf = batch['sdf'].to(device)

        optimizer.zero_grad()

        outputs = model(points)

        # PGA_INR_SDF returns 'sdf' key, not 'density'
        sdf_output = outputs['sdf']

        # Compute normals from SDF gradient
        grad = torch.autograd.grad(
            outputs=sdf_output,
            inputs=points,
            grad_outputs=torch.ones_like(sdf_output),
            create_graph=True
        )[0]
        gt_normals = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)

        # GeometricConsistencyLoss expects 'density' key, so add it
        outputs['density'] = sdf_output

        loss, metrics = loss_fn(outputs, gt_sdf, gt_normals)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f}, "
                  f"SDF = {metrics['sdf']:.6f}, Eikonal = {metrics['eikonal']:.6f}")

    print(f"\n  Final loss: {losses[-1]:.6f}")
    return model


def visualize_sdf(model, device, output_dir):
    """Visualize the trained SDF with slices."""
    print("\n" + "=" * 60)
    print("Visualizing Trained SDF")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Helper to extract SDF from model output
    def sdf_fn(p):
        return model(p)['sdf']

    # XY slice (Z=0)
    plot_sdf_slice(
        sdf_fn,
        resolution=128,
        ax=axes[0],
        slice_axis='z',
        slice_value=0.0,
        bounds=(-1.0, 1.0),
        title="XY Slice (Z=0)",
        device=device
    )

    # XZ slice (Y=0)
    plot_sdf_slice(
        sdf_fn,
        resolution=128,
        ax=axes[1],
        slice_axis='y',
        slice_value=0.0,
        bounds=(-1.0, 1.0),
        title="XZ Slice (Y=0)",
        device=device
    )

    # YZ slice (X=0)
    plot_sdf_slice(
        sdf_fn,
        resolution=128,
        ax=axes[2],
        slice_axis='x',
        slice_value=0.0,
        bounds=(-1.0, 1.0),
        title="YZ Slice (X=0)",
        device=device
    )

    plt.suptitle("Canonical SDF (T-Pose)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "10_canonical_sdf.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# 5. Skinning Comparison (LBS vs DQS)
# =============================================================================

def compare_skinning_methods(character, output_dir):
    """Compare LBS vs DQS skinning on the same pose."""
    print("\n" + "=" * 60)
    print("Comparing LBS vs DQS Skinning")
    print("=" * 60)

    chain = character['kinematic_chain']
    lbs = character['lbs']
    dqs = character['dqs']

    if chain is None or lbs is None or dqs is None:
        print("  Skinning not available")
        return

    if not character['animations']:
        print("  No animations for comparison")
        return

    # Pick middle frame of first animation
    anim = list(character['animations'].values())[0]
    mid_frame = anim.keyframe_rotations.shape[0] // 2
    rotations = anim.keyframe_rotations[mid_frame]
    root_trans = anim.root_translations[mid_frame]

    transforms = chain.forward_kinematics(rotations, root_trans)

    lbs_deformed = lbs(transforms).detach().cpu().numpy()
    dqs_deformed = dqs(transforms).detach().cpu().numpy()
    rest = character['rest_vertices'].cpu().numpy()

    fig = plt.figure(figsize=(18, 6))
    step = max(1, len(rest) // 3000)

    # Rest pose
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(rest[::step, 0], rest[::step, 1], rest[::step, 2],
                c='gray', s=1, alpha=0.5)
    ax1.set_title('Rest Pose')

    # LBS
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(lbs_deformed[::step, 0], lbs_deformed[::step, 1], lbs_deformed[::step, 2],
                c='blue', s=1, alpha=0.5)
    ax2.set_title('Linear Blend Skinning (LBS)')

    # DQS
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(dqs_deformed[::step, 0], dqs_deformed[::step, 1], dqs_deformed[::step, 2],
                c='green', s=1, alpha=0.5)
    ax3.set_title('Dual Quaternion Skinning (DQS)')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle('Skinning Method Comparison', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "10_skinning_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# 6. Neural Mesh Extraction and Articulation
# =============================================================================

def extract_neural_mesh(model, device, resolution=128):
    """Extract mesh from trained neural SDF using marching cubes."""
    print("\n  Extracting mesh from neural SDF...")

    def sdf_fn(points):
        """SDF function for marching cubes."""
        # Add batch dimension if needed
        if points.dim() == 2:
            points = points.unsqueeze(0)
        with torch.no_grad():
            output = model(points)
        return output['sdf'].squeeze()

    neural_mesh = sdf_to_mesh(
        sdf_fn=sdf_fn,
        resolution=resolution,
        bounds=(-1.0, 1.0),
        level=0.0,
        device=device,
        batch_size=10000
    )

    print(f"    Extracted mesh: {len(neural_mesh.vertices)} vertices, {len(neural_mesh.faces)} faces")
    return neural_mesh


def create_neural_lbs(neural_mesh, character, device):
    """Create LinearBlendSkinning for neural-extracted mesh."""
    print("  Transferring skinning weights to neural mesh...")

    # Get original mesh data
    original_verts = character['rest_vertices'].cpu().numpy()
    original_weights = character['bone_weights'].cpu().numpy()
    original_indices = character['bone_indices'].cpu().numpy()

    # Transfer weights using K-nearest neighbors
    neural_weights, neural_indices = transfer_skinning_weights(
        source_vertices=original_verts,
        target_vertices=neural_mesh.vertices,
        source_weights=original_weights,
        source_indices=original_indices,
        k_nearest=4
    )

    print(f"    Weights transferred to {len(neural_weights)} vertices")

    # Create LBS for neural mesh
    neural_lbs = LinearBlendSkinning(
        kinematic_chain=character['kinematic_chain'],
        rest_vertices=torch.tensor(neural_mesh.vertices, dtype=torch.float32, device=device),
        bone_weights=torch.tensor(neural_weights, dtype=torch.float32, device=device),
        bone_indices=torch.tensor(neural_indices, dtype=torch.long, device=device)
    )

    return neural_lbs


def visualize_neural_animation(character, neural_mesh, neural_lbs, anim_name, output_dir,
                               num_frames=30, display_name=None):
    """Generate animation GIF using neural-extracted mesh."""
    display = display_name or anim_name
    print(f"\n  Generating neural animation: {display}")

    if anim_name not in character['animations']:
        print(f"    Animation '{anim_name}' not found, skipping")
        return

    anim = character['animations'][anim_name]
    chain = character['kinematic_chain']

    # Sample frames evenly
    total_frames = anim.keyframe_rotations.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

    frames = []
    for frame_idx in tqdm(frame_indices, desc=f"    Rendering neural {display}"):
        rotations = anim.keyframe_rotations[frame_idx]
        root_trans = anim.root_translations[frame_idx]

        # Apply skinning to neural mesh
        transforms = chain.forward_kinematics(rotations, root_trans)
        deformed = neural_lbs(transforms).detach().cpu().numpy()

        # Render frame
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample vertices for faster rendering
        step = max(1, len(deformed) // 3000)
        ax.scatter(
            deformed[::step, 0],
            deformed[::step, 1],
            deformed[::step, 2],
            c=deformed[::step, 1], cmap='plasma', s=1, alpha=0.6
        )

        t = frame_idx / max(total_frames - 1, 1) * anim.duration
        ax.set_title(f'Neural {display}: t = {t:.2f}s')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Render to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(image)
        plt.close(fig)

    # Save GIF
    clean_name = display.replace(':', '_').replace(' ', '_').replace('.', '_').lower()
    save_path = output_dir / f"10_neural_{clean_name}.gif"
    try:
        save_animation_gif(frames, str(save_path), fps=15)
        print(f"    Saved: {save_path}")
    except Exception as e:
        print(f"    Failed to save GIF: {e}")


def compare_original_vs_neural(character, neural_mesh, neural_lbs, output_dir):
    """Create side-by-side comparison of original mesh vs neural mesh animation."""
    print("\n  Creating original vs neural comparison...")

    if not character['animations']:
        print("    No animations for comparison")
        return

    chain = character['kinematic_chain']
    original_lbs = character['lbs']

    # Use first animation, middle frame
    anim = list(character['animations'].values())[0]
    mid_frame = anim.keyframe_rotations.shape[0] // 2
    rotations = anim.keyframe_rotations[mid_frame]
    root_trans = anim.root_translations[mid_frame]

    transforms = chain.forward_kinematics(rotations, root_trans)

    original_deformed = original_lbs(transforms).detach().cpu().numpy()
    neural_deformed = neural_lbs(transforms).detach().cpu().numpy()

    fig = plt.figure(figsize=(16, 8))

    # Original mesh
    ax1 = fig.add_subplot(121, projection='3d')
    step1 = max(1, len(original_deformed) // 3000)
    ax1.scatter(
        original_deformed[::step1, 0],
        original_deformed[::step1, 1],
        original_deformed[::step1, 2],
        c=original_deformed[::step1, 1], cmap='viridis', s=1, alpha=0.6
    )
    ax1.set_title(f'Original Mesh ({len(original_deformed)} verts)')

    # Neural mesh
    ax2 = fig.add_subplot(122, projection='3d')
    step2 = max(1, len(neural_deformed) // 3000)
    ax2.scatter(
        neural_deformed[::step2, 0],
        neural_deformed[::step2, 1],
        neural_deformed[::step2, 2],
        c=neural_deformed[::step2, 1], cmap='plasma', s=1, alpha=0.6
    )
    ax2.set_title(f'Neural SDF Mesh ({len(neural_deformed)} verts)')

    for ax in [ax1, ax2]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle('Original vs Neural SDF Mesh (Same Pose)', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "10_original_vs_neural.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close()


def render_mesh_multiview(verts, ax_front, ax_side, ax_top, cmap, step, alpha=0.6):
    """Render mesh from front, side, and top views.

    Args:
        verts: (N, 3) vertex positions
        ax_front: matplotlib axes for front view (XY plane, looking from +Z)
        ax_side: matplotlib axes for side view (ZY plane, looking from +X)
        ax_top: matplotlib axes for top view (XZ plane, looking from +Y)
        cmap: colormap name
        step: subsampling step
        alpha: point transparency
    """
    # Front view (XY plane) - looking at the character from the front
    ax_front.scatter(
        verts[::step, 0], verts[::step, 1],
        c=verts[::step, 1], cmap=cmap, s=1, alpha=alpha
    )

    # Side view (ZY plane) - looking at the character from the side
    ax_side.scatter(
        verts[::step, 2], verts[::step, 1],
        c=verts[::step, 1], cmap=cmap, s=1, alpha=alpha
    )

    # Top view (XZ plane) - looking at the character from above
    ax_top.scatter(
        verts[::step, 0], verts[::step, 2],
        c=verts[::step, 1], cmap=cmap, s=1, alpha=alpha
    )


def visualize_multiview_comparison_animation(character, neural_mesh, neural_lbs, output_dir,
                                              num_frames=30):
    """
    Generate a comprehensive comparison GIF showing original vs neural animation
    from front, side, and top views for all animations.

    Creates a 6-panel layout:
    - Top row: Original mesh (Front, Side, Top views)
    - Bottom row: Neural mesh (Front, Side, Top views)

    All animations are shown sequentially in the same GIF.
    """
    print("\n  Creating multi-view comparison animation...")

    if not character['animations']:
        print("    No animations for comparison")
        return

    chain = character['kinematic_chain']
    original_lbs = character['lbs']

    # Map internal names to clean display names
    name_mapping = {
        'mixamo.com': 'Walking',
        'x_bot_walking_mixamo.com': 'Walking',
        'x_bot_breakdance_1990_mixamo.com': 'Breakdance',
    }

    frames = []
    bounds = 1.5

    for anim_name, anim in character['animations'].items():
        # Get display name
        display_name = name_mapping.get(anim_name)
        if display_name is None:
            if 'walking' in anim_name.lower():
                display_name = 'Walking'
            elif 'breakdance' in anim_name.lower():
                display_name = 'Breakdance'
            else:
                display_name = anim_name

        print(f"    Rendering {display_name}...")

        # Sample frames evenly
        total_frames = anim.keyframe_rotations.shape[0]
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

        for frame_idx in tqdm(frame_indices, desc=f"      {display_name}", leave=False):
            rotations = anim.keyframe_rotations[frame_idx]
            root_trans = anim.root_translations[frame_idx]

            # Apply skinning
            transforms = chain.forward_kinematics(rotations, root_trans)
            original_deformed = original_lbs(transforms).detach().cpu().numpy()
            neural_deformed = neural_lbs(transforms).detach().cpu().numpy()

            # Create 6-panel figure (2 rows x 3 columns)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Subsampling steps
            step_orig = max(1, len(original_deformed) // 3000)
            step_neural = max(1, len(neural_deformed) // 3000)

            # Row 0: Original mesh
            render_mesh_multiview(
                original_deformed,
                axes[0, 0], axes[0, 1], axes[0, 2],
                cmap='viridis', step=step_orig
            )

            # Row 1: Neural mesh
            render_mesh_multiview(
                neural_deformed,
                axes[1, 0], axes[1, 1], axes[1, 2],
                cmap='plasma', step=step_neural
            )

            # Configure all axes
            view_titles = ['Front View', 'Side View', 'Top View']
            for col in range(3):
                axes[0, col].set_title(f'Original - {view_titles[col]}', fontsize=10)
                axes[1, col].set_title(f'Neural - {view_titles[col]}', fontsize=10)

            # Set axis limits and labels
            for row in range(2):
                # Front view (XY)
                axes[row, 0].set_xlim(-bounds, bounds)
                axes[row, 0].set_ylim(-bounds, bounds)
                axes[row, 0].set_xlabel('X')
                axes[row, 0].set_ylabel('Y')
                axes[row, 0].set_aspect('equal')

                # Side view (ZY)
                axes[row, 1].set_xlim(-bounds, bounds)
                axes[row, 1].set_ylim(-bounds, bounds)
                axes[row, 1].set_xlabel('Z')
                axes[row, 1].set_ylabel('Y')
                axes[row, 1].set_aspect('equal')

                # Top view (XZ)
                axes[row, 2].set_xlim(-bounds, bounds)
                axes[row, 2].set_ylim(-bounds, bounds)
                axes[row, 2].set_xlabel('X')
                axes[row, 2].set_ylabel('Z')
                axes[row, 2].set_aspect('equal')

            # Time info
            t = frame_idx / max(total_frames - 1, 1) * anim.duration
            fig.suptitle(
                f'{display_name} Animation: t = {t:.2f}s\n'
                f'Original: {len(original_deformed)} verts | Neural: {len(neural_deformed)} verts',
                fontsize=12
            )

            plt.tight_layout()

            # Render to image
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(image)
            plt.close(fig)

    # Save combined GIF
    save_path = output_dir / "10_comparison_multiview.gif"
    try:
        save_animation_gif(frames, str(save_path), fps=15)
        print(f"    Saved: {save_path}")
        print(f"    Total frames: {len(frames)} ({len(character['animations'])} animations)")
    except Exception as e:
        print(f"    Failed to save GIF: {e}")


def run_neural_articulation(model, character, device, output_dir):
    """
    Phase 6: Neural Mesh Extraction and Articulation

    This demonstrates the core PGA-INR capability:
    1. Extract mesh from the trained canonical SDF
    2. Transfer skinning weights from original mesh
    3. Animate the neural mesh using the skeleton

    Outputs:
    - 10_original_vs_neural.png: Static comparison at a single pose
    - 10_comparison_multiview.gif: All animations, 6-panel view (original vs neural
      from front/side/top)
    """
    print("\n" + "=" * 60)
    print("Phase 6: Neural Mesh Extraction and Articulation")
    print("=" * 60)

    # Step 1: Extract mesh from neural SDF
    neural_mesh = extract_neural_mesh(model, device, resolution=100)

    if len(neural_mesh.vertices) == 0:
        print("  Error: No mesh extracted from SDF")
        return

    # Step 2: Create LBS with transferred weights
    neural_lbs = create_neural_lbs(neural_mesh, character, device)

    # Step 3: Compare original vs neural mesh (static image)
    compare_original_vs_neural(character, neural_mesh, neural_lbs, output_dir)

    # Step 4: Generate comprehensive multi-view comparison animation
    # This creates a single GIF with all animations showing:
    # - Original vs Neural mesh side by side
    # - Front, Side, and Top views (6 panels total)
    visualize_multiview_comparison_animation(
        character, neural_mesh, neural_lbs, output_dir,
        num_frames=30
    )

    print("\n  Neural articulation complete!")


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def main():
    """Run the complete FBX character pipeline."""
    print("\n" + "=" * 60)
    print("FBX Character Loading & Articulated Neural Fields")
    print("=" * 60 + "\n")

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Phase 1: Load character
    try:
        character = load_character(device)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo run this example, you need:")
        print("  1. input/3d_meshes/x_bot_t_pose.fbx - T-pose mesh")
        print("  2. input/animations/x_bot_walking.fbx - Walking animation (optional)")
        print("  3. input/animations/x_bot_breakdance_1990.fbx - Breakdance animation (optional)")
        return
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo run this example, install pyassimp:")
        print("  pip install pyassimp")
        print("  brew install assimp  # macOS")
        return

    # Phase 2: Visualizations
    visualize_skeleton(character, OUTPUT_DIR)
    visualize_skinning_weights(character, OUTPUT_DIR)
    visualize_rest_pose(character, OUTPUT_DIR)

    # Phase 3: Animation GIFs
    visualize_all_animations(character, OUTPUT_DIR)

    # Phase 4: Skinning comparison
    compare_skinning_methods(character, OUTPUT_DIR)

    # Phase 5: Train canonical SDF
    # Use --skip-sdf flag to skip training (takes a few minutes)
    skip_sdf_flag = '--skip-sdf' in sys.argv

    model = None
    if skip_sdf_flag:
        print("\n  Skipping SDF training (--skip-sdf flag provided)")
    else:
        print("\n" + "=" * 60)
        print("Phase 5: Training Canonical SDF")
        print("=" * 60)
        model = train_canonical_sdf(character, device, epochs=300)
        visualize_sdf(model, device, OUTPUT_DIR)

        # Phase 6: Neural articulation (only if SDF was trained)
        run_neural_articulation(model, character, device, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("10_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
