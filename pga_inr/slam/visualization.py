"""
SLAM visualization utilities.

Provides visualization for:
- Camera trajectory
- Neural map slices
- Keyframe positions
- Point clouds
- Loss curves
- Real-time SLAM state
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from .types import Keyframe, TrackingResult, SLAMState


class SLAMVisualizer:
    """
    Visualization utilities for SLAM.

    Provides methods for plotting:
    - Camera trajectories in 3D
    - Neural map SDF slices
    - Keyframe point clouds
    - Optimization loss curves
    - Real-time SLAM dashboard
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100
    ):
        """
        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_trajectory(
        self,
        poses: List[Tuple[int, torch.Tensor]],
        ground_truth: Optional[List[Tuple[int, torch.Tensor]]] = None,
        keyframe_ids: Optional[List[int]] = None,
        title: str = "Camera Trajectory",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot camera trajectory in 3D.

        Args:
            poses: List of (frame_id, translation) tuples
            ground_truth: Optional ground truth trajectory
            keyframe_ids: Frame IDs that are keyframes
            title: Plot title
            ax: Existing axis to plot on
            show: Whether to display
            save_path: Path to save figure

        Returns:
            Figure object
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # Extract positions
        frame_ids = [p[0] for p in poses]
        positions = torch.stack([p[1] for p in poses]).cpu().numpy()

        # Plot estimated trajectory
        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=1.5, label='Estimated', alpha=0.8
        )
        ax.scatter(
            positions[0, 0], positions[0, 1], positions[0, 2],
            c='green', s=100, marker='o', label='Start'
        )
        ax.scatter(
            positions[-1, 0], positions[-1, 1], positions[-1, 2],
            c='red', s=100, marker='s', label='End'
        )

        # Plot ground truth if available
        if ground_truth is not None:
            gt_positions = torch.stack([p[1] for p in ground_truth]).cpu().numpy()
            ax.plot(
                gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                'g--', linewidth=1.5, label='Ground Truth', alpha=0.8
            )

        # Highlight keyframes
        if keyframe_ids is not None:
            kf_mask = [fid in keyframe_ids for fid in frame_ids]
            if any(kf_mask):
                kf_positions = positions[kf_mask]
                ax.scatter(
                    kf_positions[:, 0], kf_positions[:, 1], kf_positions[:, 2],
                    c='orange', s=50, marker='^', label='Keyframes', alpha=0.7
                )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend(loc='upper right')

        # Equal aspect ratio
        self._set_axes_equal(ax)

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_trajectory_2d(
        self,
        poses: List[Tuple[int, torch.Tensor]],
        ground_truth: Optional[List[Tuple[int, torch.Tensor]]] = None,
        keyframe_ids: Optional[List[int]] = None,
        plane: str = 'xy',
        title: str = "Camera Trajectory (Top-down)",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot camera trajectory in 2D projection.

        Args:
            poses: List of (frame_id, translation) tuples
            ground_truth: Optional ground truth trajectory
            keyframe_ids: Frame IDs that are keyframes
            plane: Projection plane ('xy', 'xz', 'yz')
            title: Plot title
            ax: Existing axis
            show: Whether to display
            save_path: Path to save

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()

        # Get axis indices
        axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        labels = {'xy': ('X', 'Y'), 'xz': ('X', 'Z'), 'yz': ('Y', 'Z')}
        i, j = axis_map.get(plane, (0, 1))
        xlabel, ylabel = labels.get(plane, ('X', 'Y'))

        # Extract positions
        frame_ids = [p[0] for p in poses]
        positions = torch.stack([p[1] for p in poses]).cpu().numpy()

        # Plot trajectory
        ax.plot(
            positions[:, i], positions[:, j],
            'b-', linewidth=2, label='Estimated', alpha=0.8
        )
        ax.scatter(positions[0, i], positions[0, j], c='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(positions[-1, i], positions[-1, j], c='red', s=150, marker='s', label='End', zorder=5)

        # Ground truth
        if ground_truth is not None:
            gt_positions = torch.stack([p[1] for p in ground_truth]).cpu().numpy()
            ax.plot(
                gt_positions[:, i], gt_positions[:, j],
                'g--', linewidth=2, label='Ground Truth', alpha=0.8
            )

        # Keyframes
        if keyframe_ids is not None:
            kf_mask = [fid in keyframe_ids for fid in frame_ids]
            if any(kf_mask):
                kf_positions = positions[kf_mask]
                ax.scatter(
                    kf_positions[:, i], kf_positions[:, j],
                    c='orange', s=80, marker='^', label='Keyframes', zorder=4
                )

        ax.set_xlabel(f'{xlabel} (m)')
        ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_sdf_slice(
        self,
        model: torch.nn.Module,
        plane: str = 'xy',
        offset: float = 0.0,
        bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
        resolution: int = 256,
        title: str = "SDF Slice",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None,
        device: torch.device = torch.device('cuda')
    ) -> Figure:
        """
        Plot a 2D slice of the neural SDF.

        Args:
            model: Neural SDF model
            plane: Slice plane ('xy', 'xz', 'yz')
            offset: Offset along perpendicular axis
            bounds: (min1, max1, min2, max2) for the two in-plane axes
            resolution: Grid resolution
            title: Plot title
            ax: Existing axis
            show: Whether to display
            save_path: Path to save
            device: Device for computation

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=self.dpi)
        else:
            fig = ax.get_figure()

        model.eval()

        # Create grid
        v1 = torch.linspace(bounds[0], bounds[1], resolution, device=device)
        v2 = torch.linspace(bounds[2], bounds[3], resolution, device=device)
        grid_v1, grid_v2 = torch.meshgrid(v1, v2, indexing='xy')

        # Build 3D points based on plane
        if plane == 'xy':
            points = torch.stack([
                grid_v1.flatten(),
                grid_v2.flatten(),
                torch.full_like(grid_v1.flatten(), offset)
            ], dim=-1)
            xlabel, ylabel = 'X (m)', 'Y (m)'
        elif plane == 'xz':
            points = torch.stack([
                grid_v1.flatten(),
                torch.full_like(grid_v1.flatten(), offset),
                grid_v2.flatten()
            ], dim=-1)
            xlabel, ylabel = 'X (m)', 'Z (m)'
        else:  # yz
            points = torch.stack([
                torch.full_like(grid_v1.flatten(), offset),
                grid_v1.flatten(),
                grid_v2.flatten()
            ], dim=-1)
            xlabel, ylabel = 'Y (m)', 'Z (m)'

        # Query SDF
        with torch.no_grad():
            outputs = model(points.unsqueeze(0), observer_pose=None)
            sdf = outputs['sdf'].squeeze().cpu().numpy()

        sdf = sdf.reshape(resolution, resolution)

        # Plot
        vmax = min(abs(sdf.min()), abs(sdf.max()), 0.5)
        im = ax.imshow(
            sdf.T, origin='lower',
            extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
            cmap='RdBu', vmin=-vmax, vmax=vmax
        )

        # Zero level set (surface)
        ax.contour(
            sdf.T,
            levels=[0],
            extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
            colors=['black'],
            linewidths=[2]
        )

        plt.colorbar(im, ax=ax, label='SDF')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} (offset={offset:.2f}m)')
        ax.set_aspect('equal')

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_loss_history(
        self,
        loss_history: List[float],
        loss_breakdown: Optional[Dict[str, List[float]]] = None,
        title: str = "Training Loss",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot optimization loss over iterations.

        Args:
            loss_history: Total loss per iteration
            loss_breakdown: Optional per-component losses
            title: Plot title
            ax: Existing axis
            show: Whether to display
            save_path: Path to save

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()

        iterations = np.arange(len(loss_history))

        ax.semilogy(iterations, loss_history, 'b-', linewidth=2, label='Total Loss')

        if loss_breakdown is not None:
            colors = plt.cm.tab10.colors
            for i, (name, values) in enumerate(loss_breakdown.items()):
                if len(values) == len(loss_history):
                    ax.semilogy(
                        iterations, values,
                        color=colors[i % len(colors)],
                        linestyle='--', linewidth=1,
                        label=name, alpha=0.7
                    )

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_keyframe_pointcloud(
        self,
        keyframes: List[Keyframe],
        max_points_per_kf: int = 1000,
        title: str = "Keyframe Point Cloud",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot aggregated point cloud from keyframes.

        Args:
            keyframes: List of keyframes with 3D points
            max_points_per_kf: Max points to sample per keyframe
            title: Plot title
            ax: Existing 3D axis
            show: Whether to display
            save_path: Path to save

        Returns:
            Figure object
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        all_points = []
        all_colors = []

        cmap = plt.cm.viridis

        for i, kf in enumerate(keyframes):
            if kf.points_world is None:
                continue

            pts = kf.points_world.cpu()
            n = pts.shape[0]

            if n > max_points_per_kf:
                idx = torch.randperm(n)[:max_points_per_kf]
                pts = pts[idx]

            all_points.append(pts)

            # Color by keyframe index
            color = cmap(i / max(len(keyframes) - 1, 1))[:3]
            all_colors.extend([color] * len(pts))

        if all_points:
            points = torch.cat(all_points, dim=0).numpy()

            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=all_colors, s=1, alpha=0.5
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        self._set_axes_equal(ax)

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_slam_dashboard(
        self,
        state: SLAMState,
        loss_history: Optional[List[float]] = None,
        title: str = "SLAM Dashboard",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a dashboard showing multiple SLAM visualizations.

        Args:
            state: Current SLAM state
            loss_history: Optional loss history
            title: Overall title
            show: Whether to display
            save_path: Path to save

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)

        # Layout: 2x2 + info panel
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5])

        # 1. Trajectory 3D
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        poses = [(kf.frame_id, kf.translation) for kf in state.keyframes]
        if poses:
            self.plot_trajectory(
                poses,
                title="Trajectory",
                ax=ax1,
                show=False
            )

        # 2. Trajectory 2D (top-down)
        ax2 = fig.add_subplot(gs[0, 1])
        if poses:
            self.plot_trajectory_2d(
                poses,
                plane='xy',
                title="Top-Down View",
                ax=ax2,
                show=False
            )

        # 3. SDF slice
        ax3 = fig.add_subplot(gs[1, 0])
        if state.map_model is not None:
            try:
                device = next(state.map_model.parameters()).device
                self.plot_sdf_slice(
                    state.map_model,
                    plane='xy',
                    offset=state.current_translation[2].item() if len(poses) > 0 else 0.0,
                    title="SDF Slice",
                    ax=ax3,
                    show=False,
                    device=device
                )
            except Exception:
                ax3.text(0.5, 0.5, "SDF unavailable", ha='center', va='center')
                ax3.set_title("SDF Slice")

        # 4. Loss history
        ax4 = fig.add_subplot(gs[1, 1])
        if loss_history:
            self.plot_loss_history(
                loss_history,
                title="Loss History",
                ax=ax4,
                show=False
            )
        else:
            ax4.text(0.5, 0.5, "No loss data", ha='center', va='center')
            ax4.set_title("Loss History")

        # 5. Info panel
        ax5 = fig.add_subplot(gs[:, 2])
        ax5.axis('off')

        info_text = f"""SLAM Statistics
═══════════════════════

Frames Processed: {state.total_frames}
Total Keyframes: {state.total_keyframes}
Tracking FPS: {state.tracking_fps:.1f}

Current Position:
  X: {state.current_translation[0].item():.3f} m
  Y: {state.current_translation[1].item():.3f} m
  Z: {state.current_translation[2].item():.3f} m

Status:
  Initialized: {'✓' if state.is_initialized else '✗'}
  Lost: {'✓ LOST!' if state.is_lost else '✗'}
"""
        ax5.text(
            0.1, 0.9, info_text,
            transform=ax5.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10
        )

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_ate_over_time(
        self,
        estimated: List[Tuple[int, torch.Tensor]],
        ground_truth: List[Tuple[int, torch.Tensor]],
        title: str = "Absolute Trajectory Error",
        ax: Optional[Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Dict[str, float]]:
        """
        Plot Absolute Trajectory Error over time.

        Args:
            estimated: Estimated trajectory
            ground_truth: Ground truth trajectory
            title: Plot title
            ax: Existing axis
            show: Whether to display
            save_path: Path to save

        Returns:
            (Figure, metrics dict)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()

        # Match frames
        est_dict = {fid: pos for fid, pos in estimated}
        gt_dict = {fid: pos for fid, pos in ground_truth}

        common_frames = sorted(set(est_dict.keys()) & set(gt_dict.keys()))

        if not common_frames:
            ax.text(0.5, 0.5, "No common frames", ha='center', va='center')
            return fig, {}

        errors = []
        for fid in common_frames:
            est_pos = est_dict[fid]
            gt_pos = gt_dict[fid]
            if isinstance(est_pos, torch.Tensor):
                est_pos = est_pos.cpu()
            if isinstance(gt_pos, torch.Tensor):
                gt_pos = gt_pos.cpu()
            err = torch.norm(est_pos - gt_pos).item()
            errors.append((fid, err))

        frame_ids = [e[0] for e in errors]
        error_values = [e[1] for e in errors]

        ax.plot(frame_ids, error_values, 'b-', linewidth=1.5)
        ax.fill_between(frame_ids, 0, error_values, alpha=0.3)

        # Statistics
        errors_np = np.array(error_values)
        rmse = np.sqrt(np.mean(errors_np ** 2))
        mean_err = np.mean(errors_np)
        max_err = np.max(errors_np)

        ax.axhline(rmse, color='r', linestyle='--', label=f'RMSE: {rmse:.3f}m')
        ax.axhline(mean_err, color='g', linestyle=':', label=f'Mean: {mean_err:.3f}m')

        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Position Error (m)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        metrics = {
            'rmse': rmse,
            'mean': mean_err,
            'max': max_err,
            'num_frames': len(common_frames)
        }

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig, metrics

    def _set_axes_equal(self, ax: Axes):
        """Set equal aspect ratio for 3D axes."""
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ])

        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])


def create_trajectory_animation(
    poses: List[Tuple[int, torch.Tensor]],
    output_path: str,
    fps: int = 30,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100
):
    """
    Create an animated trajectory video.

    Args:
        poses: List of (frame_id, translation) tuples
        output_path: Path to save animation (GIF or MP4)
        fps: Frames per second
        figsize: Figure size
        dpi: Resolution
    """
    import matplotlib.animation as animation

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    positions = torch.stack([p[1] for p in poses]).cpu().numpy()

    # Set axis limits from full trajectory
    margin = 0.5
    ax.set_xlim([positions[:, 0].min() - margin, positions[:, 0].max() + margin])
    ax.set_ylim([positions[:, 1].min() - margin, positions[:, 1].max() + margin])
    ax.set_zlim([positions[:, 2].min() - margin, positions[:, 2].max() + margin])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Trajectory')

    line, = ax.plot([], [], [], 'b-', linewidth=2)
    point, = ax.plot([], [], [], 'ro', markersize=10)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(frame):
        idx = frame + 1
        line.set_data(positions[:idx, 0], positions[:idx, 1])
        line.set_3d_properties(positions[:idx, 2])
        point.set_data([positions[idx-1, 0]], [positions[idx-1, 1]])
        point.set_3d_properties([positions[idx-1, 2]])
        return line, point

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(poses), interval=1000/fps,
        blit=True
    )

    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    else:
        anim.save(output_path, writer='ffmpeg', fps=fps)

    plt.close(fig)


def visualize_tracking_result(
    result: TrackingResult,
    frame_rgb: Optional[torch.Tensor] = None,
    title: str = "Tracking Result"
) -> Figure:
    """
    Visualize a single tracking result.

    Args:
        result: TrackingResult from tracker
        frame_rgb: Optional RGB image
        title: Plot title

    Returns:
        Figure object
    """
    n_plots = 2 if frame_rgb is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Loss breakdown
    ax = axes[0]
    if result.loss_breakdown:
        names = list(result.loss_breakdown.keys())
        values = list(result.loss_breakdown.values())
        bars = ax.bar(names, values, color='steelblue')
        ax.set_ylabel('Loss Value')
        ax.set_title(f'Loss Breakdown (Total: {result.final_loss:.4f})')

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8
            )
    else:
        ax.text(0.5, 0.5, f'Final Loss: {result.final_loss:.4f}',
                ha='center', va='center', fontsize=12)
        ax.set_title('Tracking Result')

    # Info text
    info = f"Converged: {result.converged}\nIterations: {result.num_iterations}\nInlier Ratio: {result.inlier_ratio:.2f}"
    ax.text(
        0.02, 0.98, info, transform=ax.transAxes,
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # RGB image
    if frame_rgb is not None and n_plots > 1:
        ax = axes[1]
        rgb = frame_rgb.cpu().numpy()
        if rgb.max() > 1:
            rgb = rgb / 255.0
        ax.imshow(rgb)
        ax.set_title('Input Frame')
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    return fig
