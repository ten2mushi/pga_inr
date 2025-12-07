"""
Motion visualization utilities.

Provides functions for visualizing skeletal motion:
- Skeleton stick figure rendering
- Mesh deformation visualization
- Motion comparison plots
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


def visualize_skeleton_frame(
    joint_positions: torch.Tensor,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    color: str = 'blue',
    show_joints: bool = True
) -> plt.Axes:
    """
    Visualize a single skeleton frame as a 3D stick figure.

    Args:
        joint_positions: Joint positions (J, 3)
        skeleton_edges: List of (parent_idx, child_idx) tuples
        ax: Matplotlib 3D axes (created if None)
        title: Plot title
        color: Line color
        show_joints: Whether to show joint markers

    Returns:
        The matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    positions = joint_positions.detach().cpu().numpy() if isinstance(joint_positions, torch.Tensor) else joint_positions

    # Plot joints
    if show_joints:
        ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
                   c=color, s=20, alpha=0.8)

    # Plot bones if edges provided
    if skeleton_edges is not None:
        for parent_idx, child_idx in skeleton_edges:
            parent = positions[parent_idx]
            child = positions[child_idx]
            ax.plot([parent[0], child[0]],
                    [parent[2], child[2]],
                    [parent[1], child[1]],
                    c=color, linewidth=2)

    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.abs(positions).max()
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, 2 * max_range)

    return ax


def visualize_skeleton_sequence(
    joint_positions: torch.Tensor,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    fps: int = 30,
    title: str = "Motion Sequence",
    figsize: Tuple[int, int] = (8, 8)
) -> animation.FuncAnimation:
    """
    Render skeleton sequence as animated stick figure.

    Args:
        joint_positions: Joint positions (T, J, 3)
        skeleton_edges: List of (parent_idx, child_idx) tuples
        save_path: Path to save GIF (optional)
        fps: Frames per second
        title: Animation title
        figsize: Figure size

    Returns:
        Matplotlib animation object
    """
    positions = joint_positions.detach().cpu().numpy() if isinstance(joint_positions, torch.Tensor) else joint_positions
    num_frames = positions.shape[0]

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Compute bounds
    all_positions = positions.reshape(-1, 3)
    max_range = np.abs(all_positions).max() * 1.2

    def init():
        ax.clear()
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, 2 * max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        return []

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, 2 * max_range)

        frame_positions = positions[frame_idx]

        # Plot joints
        ax.scatter(frame_positions[:, 0], frame_positions[:, 2], frame_positions[:, 1],
                   c='blue', s=20, alpha=0.8)

        # Plot bones
        if skeleton_edges is not None:
            for parent_idx, child_idx in skeleton_edges:
                parent = frame_positions[parent_idx]
                child = frame_positions[child_idx]
                ax.plot([parent[0], child[0]],
                        [parent[2], child[2]],
                        [parent[1], child[1]],
                        c='blue', linewidth=2)

        ax.set_title(f"{title} - Frame {frame_idx + 1}/{num_frames}")
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        return []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=1000 / fps, blit=False
    )

    if save_path is not None:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved animation to {save_path}")

    plt.close(fig)
    return anim


def compute_skeleton_edges(kinematic_chain) -> List[Tuple[int, int]]:
    """
    Compute skeleton edges from KinematicChain.

    Args:
        kinematic_chain: KinematicChain object

    Returns:
        List of (parent_idx, child_idx) tuples
    """
    edges = []

    for child_name, joint in kinematic_chain.joints.items():
        if joint.parent is not None:
            parent_name = joint.parent.name
            parent_idx = kinematic_chain.get_joint_index(parent_name)
            child_idx = kinematic_chain.get_joint_index(child_name)
            edges.append((parent_idx, child_idx))

    return edges


def visualize_motion_comparison(
    gt_positions: torch.Tensor,
    pred_positions: torch.Tensor,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    fps: int = 30,
    figsize: Tuple[int, int] = (16, 8)
) -> animation.FuncAnimation:
    """
    Side-by-side comparison of ground truth vs predicted motion.

    Args:
        gt_positions: Ground truth positions (T, J, 3)
        pred_positions: Predicted positions (T, J, 3)
        skeleton_edges: List of edges
        save_path: Path to save GIF
        fps: Frames per second
        figsize: Figure size

    Returns:
        Matplotlib animation object
    """
    gt_pos = gt_positions.detach().cpu().numpy() if isinstance(gt_positions, torch.Tensor) else gt_positions
    pred_pos = pred_positions.detach().cpu().numpy() if isinstance(pred_positions, torch.Tensor) else pred_positions

    num_frames = min(gt_pos.shape[0], pred_pos.shape[0])

    # Compute bounds
    all_positions = np.concatenate([gt_pos.reshape(-1, 3), pred_pos.reshape(-1, 3)])
    max_range = np.abs(all_positions).max() * 1.2

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def init():
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(0, 2 * max_range)
        return []

    def animate(frame_idx):
        for ax, positions, title, color in [
            (ax1, gt_pos, "Ground Truth", "blue"),
            (ax2, pred_pos, "Predicted", "red")
        ]:
            ax.clear()
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(0, 2 * max_range)

            frame_positions = positions[frame_idx]

            ax.scatter(frame_positions[:, 0], frame_positions[:, 2], frame_positions[:, 1],
                       c=color, s=20, alpha=0.8)

            if skeleton_edges is not None:
                for parent_idx, child_idx in skeleton_edges:
                    parent = frame_positions[parent_idx]
                    child = frame_positions[child_idx]
                    ax.plot([parent[0], child[0]],
                            [parent[2], child[2]],
                            [parent[1], child[1]],
                            c=color, linewidth=2)

            ax.set_title(f"{title} - Frame {frame_idx + 1}/{num_frames}")
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

        return []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=1000 / fps, blit=False
    )

    if save_path is not None:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved comparison to {save_path}")

    plt.close(fig)
    return anim


def plot_motion_statistics(
    motions: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot motion statistics (mean, std, velocity).

    Args:
        motions: Dict of motion tensors {name: (T, J, 6 or 3)}
        save_path: Path to save plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = plt.cm.tab10.colors

    for idx, (name, motion) in enumerate(motions.items()):
        motion_np = motion.detach().cpu().numpy() if isinstance(motion, torch.Tensor) else motion
        color = colors[idx % len(colors)]

        # Mean over joints
        mean_per_frame = motion_np.mean(axis=1)  # (T, feat_dim)

        # Plot mean feature over time
        axes[0, 0].plot(mean_per_frame.mean(axis=1), label=name, color=color)
        axes[0, 0].set_title("Mean Feature Value Over Time")
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel("Mean Value")

        # Velocity magnitude
        velocity = np.diff(motion_np, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=-1).mean(axis=1)
        axes[0, 1].plot(vel_magnitude, label=name, color=color)
        axes[0, 1].set_title("Velocity Magnitude Over Time")
        axes[0, 1].set_xlabel("Frame")
        axes[0, 1].set_ylabel("Velocity")

        # Per-joint variance
        joint_var = motion_np.var(axis=0).mean(axis=-1)
        axes[1, 0].bar(np.arange(len(joint_var)) + idx * 0.3, joint_var,
                       width=0.3, label=name, color=color)
        axes[1, 0].set_title("Per-Joint Variance")
        axes[1, 0].set_xlabel("Joint Index")
        axes[1, 0].set_ylabel("Variance")

        # Histogram of values
        axes[1, 1].hist(motion_np.flatten(), bins=50, alpha=0.5,
                        label=name, color=color)
        axes[1, 1].set_title("Value Distribution")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].set_ylabel("Count")

    for ax in axes.flat:
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved statistics plot to {save_path}")

    plt.close(fig)


def rotation_6d_to_positions(
    rotations_6d: torch.Tensor,
    kinematic_chain,
    root_translation: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert 6D rotations to joint positions via FK.

    Args:
        rotations_6d: Rotations (batch, J, 6, T) or (T, J, 6)
        kinematic_chain: KinematicChain for FK
        root_translation: Root translation (batch, 3, T) or (T, 3)

    Returns:
        Joint positions (batch, J, 3, T) or (T, J, 3)
    """
    from ..utils.rotation import rotation_6d_to_quaternion, normalize_rotation_6d

    # Normalize 6D rotations before conversion (important for diffusion outputs)
    rotations_6d = normalize_rotation_6d(rotations_6d)

    # Handle different input shapes
    has_batch = rotations_6d.dim() == 4

    if has_batch:
        batch_size, num_joints, _, num_frames = rotations_6d.shape
        # Process each frame
        all_positions = []

        for f in range(num_frames):
            frame_rot = rotations_6d[:, :, :, f]  # (batch, J, 6)

            # Convert to quaternions
            quat = rotation_6d_to_quaternion(frame_rot.reshape(-1, 6))
            quat = quat.reshape(batch_size, num_joints, 4)

            root_trans = root_translation[:, :, f] if root_translation is not None else None

            # FK for each batch
            batch_positions = []
            for b in range(batch_size):
                root_t = root_trans[b] if root_trans is not None else None
                transforms = kinematic_chain.forward_kinematics(
                    local_rotations=quat[b],
                    root_translation=root_t
                )

                positions = []
                for name in kinematic_chain.joint_order:
                    trans, _ = transforms[name]
                    positions.append(trans)
                batch_positions.append(torch.stack(positions))

            all_positions.append(torch.stack(batch_positions))

        return torch.stack(all_positions, dim=-1)  # (batch, J, 3, T)

    else:
        # (T, J, 6) format
        num_frames, num_joints, _ = rotations_6d.shape
        all_positions = []

        for f in range(num_frames):
            frame_rot = rotations_6d[f]  # (J, 6)
            quat = rotation_6d_to_quaternion(frame_rot)  # (J, 4)

            root_trans = root_translation[f] if root_translation is not None else None

            transforms = kinematic_chain.forward_kinematics(
                local_rotations=quat,
                root_translation=root_trans
            )

            positions = []
            for name in kinematic_chain.joint_order:
                trans, _ = transforms[name]
                positions.append(trans)

            all_positions.append(torch.stack(positions))

        return torch.stack(all_positions)  # (T, J, 3)
