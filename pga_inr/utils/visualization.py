"""
Visualization utilities for PGA-INR.

Provides comprehensive plotting and rendering helpers for:
- SDF visualization (slices, isosurfaces, 3D volumes)
- Point clouds and coordinate frames
- Training metrics and loss curves
- Kinematic chains and animations
- Rendered images (depth, normals, RGB)
- Scene composition and CSG operations
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Callable, Any
from dataclasses import dataclass
import warnings


# =============================================================================
# Configuration and Style
# =============================================================================

@dataclass
class PlotStyle:
    """Global plotting style configuration."""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    cmap_diverging: str = 'RdBu'
    cmap_sequential: str = 'viridis'
    cmap_depth: str = 'plasma'
    surface_color: str = '#4ECDC4'
    background_color: str = '#1a1a2e'
    grid_alpha: float = 0.3
    font_size: int = 12


DEFAULT_STYLE = PlotStyle()


def _ensure_matplotlib():
    """Ensure matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _ensure_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


# =============================================================================
# SDF Visualization
# =============================================================================

def plot_sdf_slice(
    sdf_fn: Callable,
    resolution: int = 128,
    slice_axis: str = 'z',
    slice_value: float = 0.0,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device = None,
    cmap: str = None,
    show_zero_contour: bool = True,
    ax: Any = None,
    title: str = None,
    style: PlotStyle = None,
):
    """
    Plot a 2D slice of an SDF.

    Args:
        sdf_fn: Function that takes (B, N, 3) points and returns dict with 'sdf'
        resolution: Grid resolution for the slice
        slice_axis: Which axis to slice ('x', 'y', or 'z')
        slice_value: Position along the slice axis
        bounds: (min, max) bounds for the grid
        device: Device for computation
        cmap: Colormap name (defaults to style.cmap_diverging)
        show_zero_contour: Whether to show the zero level set
        ax: Existing matplotlib axis (creates new figure if None)
        title: Plot title
        style: PlotStyle configuration

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE
    cmap = cmap or style.cmap_diverging
    device = device or torch.device('cpu')

    # Create 2D grid
    coords = torch.linspace(bounds[0], bounds[1], resolution, device=device)

    if slice_axis == 'z':
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        z = torch.full_like(x, slice_value)
        points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
        xlabel, ylabel = 'X', 'Y'
    elif slice_axis == 'y':
        x, z = torch.meshgrid(coords, coords, indexing='ij')
        y = torch.full_like(x, slice_value)
        points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
        xlabel, ylabel = 'X', 'Z'
    else:  # 'x'
        y, z = torch.meshgrid(coords, coords, indexing='ij')
        x = torch.full_like(y, slice_value)
        points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
        xlabel, ylabel = 'Y', 'Z'

    # Evaluate SDF
    with torch.no_grad():
        outputs = sdf_fn(points)
        if isinstance(outputs, dict):
            sdf = outputs.get('sdf', outputs.get('density'))
        else:
            sdf = outputs
        sdf = sdf.squeeze()

    sdf = sdf.reshape(resolution, resolution).cpu().numpy()

    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=style.figsize, dpi=style.dpi)
    else:
        fig = ax.get_figure()

    # Plot SDF
    vmax = max(abs(sdf.min()), abs(sdf.max()), 0.01)
    im = ax.imshow(
        sdf.T,
        origin='lower',
        extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax
    )

    if show_zero_contour:
        cs = ax.contour(
            sdf.T,
            levels=[0],
            colors='black',
            linewidths=2,
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]]
        )

    ax.set_xlabel(xlabel, fontsize=style.font_size)
    ax.set_ylabel(ylabel, fontsize=style.font_size)

    title = title or f'SDF slice at {slice_axis}={slice_value:.2f}'
    ax.set_title(title, fontsize=style.font_size + 2)

    cbar = plt.colorbar(im, ax=ax, label='SDF value')
    ax.set_aspect('equal')

    if created_fig:
        return fig, ax
    return ax


def plot_sdf_3d_isosurface(
    sdf_fn: Callable,
    resolution: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
    device: torch.device = None,
    color: str = None,
    ax: Any = None,
    title: str = None,
    style: PlotStyle = None,
    alpha: float = 0.8,
):
    """
    Plot 3D isosurface of an SDF using marching cubes.

    Args:
        sdf_fn: Function that takes (B, N, 3) points and returns dict with 'sdf'
        resolution: Grid resolution
        bounds: (min, max) bounds for the grid
        level: Isosurface level (0 for surface)
        device: Device for computation
        color: Surface color
        ax: Existing 3D matplotlib axis
        title: Plot title
        style: PlotStyle configuration
        alpha: Surface transparency

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE
    color = color or style.surface_color
    device = device or torch.device('cpu')

    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "scikit-image is required for isosurface extraction. "
            "Install with: pip install scikit-image"
        )

    # Create 3D grid
    coords = torch.linspace(bounds[0], bounds[1], resolution, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

    # Evaluate SDF
    with torch.no_grad():
        outputs = sdf_fn(points)
        if isinstance(outputs, dict):
            sdf = outputs.get('sdf', outputs.get('density'))
        else:
            sdf = outputs
        sdf = sdf.squeeze()

    sdf = sdf.reshape(resolution, resolution, resolution).cpu().numpy()

    # Marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            sdf, level=level, spacing=(
                (bounds[1] - bounds[0]) / (resolution - 1),
            ) * 3
        )
        verts = verts + bounds[0]
    except ValueError:
        warnings.warn("No isosurface found at the specified level")
        verts, faces = np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if len(verts) > 0:
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            color=color, alpha=alpha, edgecolor='none'
        )

    ax.set_xlabel('X', fontsize=style.font_size)
    ax.set_ylabel('Y', fontsize=style.font_size)
    ax.set_zlabel('Z', fontsize=style.font_size)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_zlim(bounds)

    title = title or f'SDF Isosurface (level={level})'
    ax.set_title(title, fontsize=style.font_size + 2)

    if created_fig:
        return fig, ax
    return ax


def plot_sdf_comparison(
    sdf_fns: List[Callable],
    labels: List[str],
    resolution: int = 100,
    slice_axis: str = 'z',
    slice_value: float = 0.0,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device = None,
    style: PlotStyle = None,
):
    """
    Compare multiple SDFs side by side.

    Args:
        sdf_fns: List of SDF functions
        labels: Labels for each SDF
        resolution: Grid resolution
        slice_axis: Which axis to slice
        slice_value: Position along slice axis
        bounds: Grid bounds
        device: Device for computation
        style: PlotStyle configuration

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE
    n = len(sdf_fns)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=style.dpi)
    if n == 1:
        axes = [axes]

    for ax, sdf_fn, label in zip(axes, sdf_fns, labels):
        plot_sdf_slice(
            sdf_fn, resolution, slice_axis, slice_value, bounds,
            device, ax=ax, title=label, style=style
        )

    plt.tight_layout()
    return fig, axes


# =============================================================================
# Point Cloud and Coordinate Frame Visualization
# =============================================================================

def plot_point_cloud(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    point_size: float = 10.0,
    ax: Any = None,
    title: str = 'Point Cloud',
    style: PlotStyle = None,
    elev: float = 30,
    azim: float = 45,
    show_normals: bool = False,
    normal_scale: float = 0.1,
    cmap: str = None,
):
    """
    Plot a 3D point cloud with optional colors and normals.

    Args:
        points: (N, 3) or (B, N, 3) point positions
        colors: (N, 3) RGB colors or (N,) scalar values
        normals: (N, 3) normal vectors
        point_size: Size of points
        ax: Existing 3D matplotlib axis
        title: Plot title
        style: PlotStyle configuration
        elev: Elevation angle
        azim: Azimuth angle
        show_normals: Whether to show normal vectors
        normal_scale: Scale factor for normal visualization
        cmap: Colormap for scalar colors

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE
    cmap = cmap or style.cmap_sequential

    points = _ensure_numpy(points)
    if points.ndim == 3:
        points = points.reshape(-1, 3)

    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot points
    if colors is not None:
        colors = _ensure_numpy(colors)
        if colors.ndim == 1:
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=point_size, cmap=cmap
            )
            plt.colorbar(scatter, ax=ax, shrink=0.6)
        else:
            if colors.ndim == 2 and colors.shape[0] != points.shape[0]:
                colors = colors.reshape(-1, 3)
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=point_size
            )
    else:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=point_size, alpha=0.6
        )

    # Plot normals
    if normals is not None and show_normals:
        normals = _ensure_numpy(normals)
        if normals.ndim == 3:
            normals = normals.reshape(-1, 3)
        step = max(1, len(points) // 100)
        ax.quiver(
            points[::step, 0], points[::step, 1], points[::step, 2],
            normals[::step, 0], normals[::step, 1], normals[::step, 2],
            length=normal_scale, normalize=True, color='red', alpha=0.6
        )

    ax.set_xlabel('X', fontsize=style.font_size)
    ax.set_ylabel('Y', fontsize=style.font_size)
    ax.set_zlabel('Z', fontsize=style.font_size)
    ax.set_title(title, fontsize=style.font_size + 2)
    ax.view_init(elev=elev, azim=azim)

    if created_fig:
        return fig, ax
    return ax


def plot_coordinate_frame(
    ax: Any,
    translation: torch.Tensor,
    quaternion: torch.Tensor,
    scale: float = 0.3,
    label: str = None,
    colors: Tuple[str, str, str] = ('red', 'green', 'blue'),
):
    """
    Plot a coordinate frame (XYZ axes) at a given pose.

    Args:
        ax: 3D matplotlib axis
        translation: (3,) position
        quaternion: (4,) rotation quaternion [w, x, y, z]
        scale: Length of axes
        label: Optional label
        colors: Colors for X, Y, Z axes
    """
    from pga_inr.utils.quaternion import quaternion_to_matrix

    translation = _ensure_numpy(translation)
    quaternion = _ensure_numpy(quaternion)

    # Convert quaternion to rotation matrix
    q_tensor = torch.tensor(quaternion).unsqueeze(0)
    R = quaternion_to_matrix(q_tensor).squeeze().numpy()

    # Plot axes
    origin = translation
    for i, (color, axis_name) in enumerate(zip(colors, ['X', 'Y', 'Z'])):
        direction = R[:, i] * scale
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color, arrow_length_ratio=0.1, linewidth=2
        )

    if label:
        ax.text(origin[0], origin[1], origin[2], f'  {label}', fontsize=10)


def plot_observer_transforms(
    points: torch.Tensor,
    local_coords: torch.Tensor,
    translations: torch.Tensor,
    quaternions: torch.Tensor,
    style: PlotStyle = None,
):
    """
    Visualize world points and their local coordinates under different observers.

    Args:
        points: (N, 3) world space points
        local_coords: (B, N, 3) local coordinates for each observer
        translations: (B, 3) observer translations
        quaternions: (B, 4) observer rotations

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    points = _ensure_numpy(points)
    local_coords = _ensure_numpy(local_coords)
    translations = _ensure_numpy(translations)
    quaternions = _ensure_numpy(quaternions)

    n_observers = len(translations)
    fig = plt.figure(figsize=(6 * (n_observers + 1), 6), dpi=style.dpi)

    # World space plot
    ax_world = fig.add_subplot(1, n_observers + 1, 1, projection='3d')
    ax_world.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, alpha=0.6, label='Points')

    # Plot observer frames
    for i in range(n_observers):
        plot_coordinate_frame(
            ax_world, translations[i], quaternions[i],
            scale=0.3, label=f'Obs {i+1}'
        )

    ax_world.set_title('World Space', fontsize=style.font_size + 2)
    ax_world.set_xlabel('X')
    ax_world.set_ylabel('Y')
    ax_world.set_zlabel('Z')
    ax_world.legend()

    # Local space plots
    for i in range(n_observers):
        ax = fig.add_subplot(1, n_observers + 1, i + 2, projection='3d')
        local = local_coords[i] if local_coords.ndim == 3 else local_coords
        ax.scatter(local[:, 0], local[:, 1], local[:, 2], s=20, alpha=0.6)
        plot_coordinate_frame(
            ax, np.zeros(3), np.array([1, 0, 0, 0]),
            scale=0.3, label='Origin'
        )
        ax.set_title(f'Observer {i+1} Local Space', fontsize=style.font_size + 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    return fig, fig.axes


# =============================================================================
# Training Metrics Visualization
# =============================================================================

def plot_training_curves(
    metrics: Dict[str, List[float]],
    ax: Any = None,
    title: str = 'Training Progress',
    style: PlotStyle = None,
    log_scale: bool = False,
):
    """
    Plot training loss curves.

    Args:
        metrics: Dictionary mapping metric names to lists of values
        ax: Existing matplotlib axis
        title: Plot title
        style: PlotStyle configuration
        log_scale: Use logarithmic y-axis

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=style.figsize, dpi=style.dpi)
    else:
        fig = ax.get_figure()

    for name, values in metrics.items():
        ax.plot(values, label=name, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=style.font_size)
    ax.set_ylabel('Loss', fontsize=style.font_size)
    ax.set_title(title, fontsize=style.font_size + 2)
    ax.legend(fontsize=style.font_size - 2)
    ax.grid(True, alpha=style.grid_alpha)

    if log_scale:
        ax.set_yscale('log')

    if created_fig:
        return fig, ax
    return ax


# =============================================================================
# Kinematic Chain and Animation Visualization
# =============================================================================

def plot_kinematic_chain(
    joint_positions: Dict[str, torch.Tensor],
    joint_tree: Dict[str, Dict],
    ax: Any = None,
    title: str = 'Kinematic Chain',
    style: PlotStyle = None,
    joint_size: float = 100,
    bone_width: float = 3,
    joint_color: str = 'red',
    bone_color: str = 'blue',
):
    """
    Plot a kinematic chain skeleton.

    Args:
        joint_positions: Dictionary mapping joint names to (3,) positions
        joint_tree: Joint hierarchy definition
        ax: Existing 3D matplotlib axis
        title: Plot title
        style: PlotStyle configuration
        joint_size: Size of joint markers
        bone_width: Width of bone lines
        joint_color: Color for joint markers
        bone_color: Color for bones

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot joints
    positions = []
    labels = []
    for name, pos in joint_positions.items():
        pos = _ensure_numpy(pos)
        positions.append(pos)
        labels.append(name)
        ax.scatter([pos[0]], [pos[1]], [pos[2]], s=joint_size, c=joint_color, marker='o')
        ax.text(pos[0], pos[1], pos[2], f'  {name}', fontsize=8)

    # Plot bones (connections)
    for joint_name, joint_info in joint_tree.items():
        parent_name = joint_info.get('parent')
        if parent_name is not None and parent_name in joint_positions:
            parent_pos = _ensure_numpy(joint_positions[parent_name])
            child_pos = _ensure_numpy(joint_positions[joint_name])
            ax.plot(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                c=bone_color, linewidth=bone_width
            )

    ax.set_xlabel('X', fontsize=style.font_size)
    ax.set_ylabel('Y', fontsize=style.font_size)
    ax.set_zlabel('Z', fontsize=style.font_size)
    ax.set_title(title, fontsize=style.font_size + 2)

    if created_fig:
        return fig, ax
    return ax


def plot_skeleton_animation(
    joint_positions_sequence: List[Dict[str, torch.Tensor]],
    joint_tree: Dict[str, Dict],
    times: Optional[List[float]] = None,
    style: PlotStyle = None,
    trail_alpha: float = 0.3,
):
    """
    Plot skeleton poses at multiple timesteps showing motion.

    Args:
        joint_positions_sequence: List of joint position dicts over time
        joint_tree: Joint hierarchy definition
        times: Optional list of time values
        style: PlotStyle configuration
        trail_alpha: Alpha for previous poses

    Returns:
        Tuple of (figure, axis)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = fig.add_subplot(111, projection='3d')

    n_frames = len(joint_positions_sequence)
    cmap = plt.cm.viridis

    for i, joint_positions in enumerate(joint_positions_sequence):
        alpha = trail_alpha + (1 - trail_alpha) * (i / max(1, n_frames - 1))
        color = cmap(i / max(1, n_frames - 1))

        # Plot bones
        for joint_name, joint_info in joint_tree.items():
            parent_name = joint_info.get('parent')
            if parent_name is not None and parent_name in joint_positions:
                parent_pos = _ensure_numpy(joint_positions[parent_name])
                child_pos = _ensure_numpy(joint_positions[joint_name])
                ax.plot(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]],
                    c=color, linewidth=2, alpha=alpha
                )

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Time')

    ax.set_xlabel('X', fontsize=style.font_size)
    ax.set_ylabel('Y', fontsize=style.font_size)
    ax.set_zlabel('Z', fontsize=style.font_size)
    ax.set_title('Skeleton Animation', fontsize=style.font_size + 2)

    return fig, ax


def plot_trajectory(
    positions: torch.Tensor,
    orientations: Optional[torch.Tensor] = None,
    ax: Any = None,
    title: str = 'Motor Trajectory',
    style: PlotStyle = None,
    show_frames: bool = True,
    frame_interval: int = 5,
    line_color: str = 'blue',
    line_width: float = 2,
):
    """
    Plot a 3D trajectory with optional orientation frames.

    Args:
        positions: (T, 3) trajectory positions
        orientations: (T, 4) quaternions [w, x, y, z]
        ax: Existing 3D matplotlib axis
        title: Plot title
        style: PlotStyle configuration
        show_frames: Show coordinate frames along trajectory
        frame_interval: Interval between frame visualizations
        line_color: Color for trajectory line
        line_width: Width of trajectory line

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    positions = _ensure_numpy(positions)

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            c=line_color, linewidth=line_width, label='Trajectory')

    # Plot start and end points
    ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
               c='green', s=100, marker='o', label='Start')
    ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
               c='red', s=100, marker='s', label='End')

    # Plot coordinate frames
    if orientations is not None and show_frames:
        orientations = _ensure_numpy(orientations)
        for i in range(0, len(positions), frame_interval):
            plot_coordinate_frame(
                ax,
                torch.tensor(positions[i]),
                torch.tensor(orientations[i]),
                scale=0.15
            )

    ax.set_xlabel('X', fontsize=style.font_size)
    ax.set_ylabel('Y', fontsize=style.font_size)
    ax.set_zlabel('Z', fontsize=style.font_size)
    ax.set_title(title, fontsize=style.font_size + 2)
    ax.legend()

    if created_fig:
        return fig, ax
    return ax


def plot_skinning_deformation(
    rest_vertices: torch.Tensor,
    deformed_vertices: torch.Tensor,
    bone_weights: Optional[torch.Tensor] = None,
    ax: Any = None,
    title: str = 'Skinning Deformation',
    style: PlotStyle = None,
):
    """
    Visualize skinning deformation showing rest pose vs deformed pose.

    Args:
        rest_vertices: (N, 3) rest pose vertices
        deformed_vertices: (N, 3) deformed vertices
        bone_weights: (N, K) bone weights for coloring
        ax: Existing 3D matplotlib axis
        title: Plot title
        style: PlotStyle configuration

    Returns:
        Tuple of (figure, axis) or axis if ax was provided
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    rest = _ensure_numpy(rest_vertices)
    deformed = _ensure_numpy(deformed_vertices)

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(14, 6), dpi=style.dpi)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        fig = ax.get_figure()
        ax1 = ax2 = ax

    # Colors based on bone weights
    if bone_weights is not None:
        weights = _ensure_numpy(bone_weights)
        colors = plt.cm.coolwarm(weights[:, 0] if weights.ndim > 1 else weights)
    else:
        colors = 'blue'

    # Rest pose
    ax1.scatter(rest[:, 0], rest[:, 1], rest[:, 2], c=colors, s=30, alpha=0.7)
    ax1.set_title('Rest Pose', fontsize=style.font_size + 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Deformed pose
    ax2.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2], c=colors, s=30, alpha=0.7)
    ax2.set_title('Deformed Pose', fontsize=style.font_size + 2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Match axis limits
    all_points = np.vstack([rest, deformed])
    margin = 0.1
    for a in [ax1, ax2]:
        a.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        a.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        a.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

    plt.suptitle(title, fontsize=style.font_size + 4)

    if created_fig:
        plt.tight_layout()
        return fig, (ax1, ax2)
    return ax


# =============================================================================
# Rendered Image Visualization
# =============================================================================

def plot_render_result(
    rgb: torch.Tensor = None,
    depth: torch.Tensor = None,
    normals: torch.Tensor = None,
    hit_mask: torch.Tensor = None,
    style: PlotStyle = None,
    title: str = 'Render Result',
):
    """
    Plot rendered images (RGB, depth, normals) in a grid.

    Args:
        rgb: (H, W, 3) RGB image
        depth: (H, W) or (H, W, 1) depth map
        normals: (H, W, 3) normal map
        hit_mask: (H, W) hit mask
        style: PlotStyle configuration
        title: Overall title

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    plots = []
    titles = []

    if rgb is not None:
        rgb = _ensure_numpy(rgb)
        if hit_mask is not None:
            mask = _ensure_numpy(hit_mask)
            rgb = rgb * mask[..., None]
        plots.append(('rgb', np.clip(rgb, 0, 1)))
        titles.append('RGB')

    if depth is not None:
        depth = _ensure_numpy(depth)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        if hit_mask is not None:
            mask = _ensure_numpy(hit_mask)
            depth = np.where(mask, depth, np.nan)
        plots.append(('depth', depth))
        titles.append('Depth')

    if normals is not None:
        normals = _ensure_numpy(normals)
        # Map normals from [-1, 1] to [0, 1] for visualization
        normal_rgb = (normals + 1) / 2
        if hit_mask is not None:
            mask = _ensure_numpy(hit_mask)
            normal_rgb = normal_rgb * mask[..., None]
        plots.append(('normals', np.clip(normal_rgb, 0, 1)))
        titles.append('Normals')

    n_plots = len(plots)
    if n_plots == 0:
        raise ValueError("At least one of rgb, depth, or normals must be provided")

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=style.dpi)
    if n_plots == 1:
        axes = [axes]

    for ax, (plot_type, data), plot_title in zip(axes, plots, titles):
        if plot_type == 'depth':
            im = ax.imshow(data, cmap=style.cmap_depth)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.imshow(data)
        ax.set_title(plot_title, fontsize=style.font_size + 2)
        ax.axis('off')

    plt.suptitle(title, fontsize=style.font_size + 4)
    plt.tight_layout()
    return fig, axes


def plot_orbit_views(
    images: List[torch.Tensor],
    angles: List[float],
    style: PlotStyle = None,
    title: str = 'Orbit Views',
):
    """
    Plot multiple rendered views from different angles.

    Args:
        images: List of (H, W, 3) RGB images
        angles: List of angles in degrees
        style: PlotStyle configuration
        title: Overall title

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=style.dpi)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (img, angle) in enumerate(zip(images, angles)):
        img = _ensure_numpy(img)
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(f'{angle:.0f}°', fontsize=style.font_size)
        axes[i].axis('off')

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=style.font_size + 4)
    plt.tight_layout()
    return fig, axes


# =============================================================================
# Scene Composition Visualization
# =============================================================================

def plot_csg_operation(
    sdf_a_fn: Callable,
    sdf_b_fn: Callable,
    composed_fn: Callable,
    operation: str,
    resolution: int = 100,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device = None,
    style: PlotStyle = None,
):
    """
    Visualize a CSG operation between two SDFs.

    Args:
        sdf_a_fn: First SDF function
        sdf_b_fn: Second SDF function
        composed_fn: Composed result function
        operation: Name of operation ('union', 'intersection', 'subtraction')
        resolution: Grid resolution
        bounds: Grid bounds
        device: Device for computation
        style: PlotStyle configuration

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE
    device = device or torch.device('cpu')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=style.dpi)

    # Plot individual SDFs
    plot_sdf_slice(sdf_a_fn, resolution, 'z', 0, bounds, device, ax=axes[0], title='Shape A', style=style)
    plot_sdf_slice(sdf_b_fn, resolution, 'z', 0, bounds, device, ax=axes[1], title='Shape B', style=style)
    plot_sdf_slice(composed_fn, resolution, 'z', 0, bounds, device, ax=axes[2], title=f'{operation.title()}', style=style)

    # Create legend for operation
    axes[3].axis('off')
    axes[3].text(0.5, 0.7, f'Operation: {operation}', fontsize=16, ha='center', va='center',
                 transform=axes[3].transAxes, fontweight='bold')

    op_formulas = {
        'union': 'min(A, B)',
        'intersection': 'max(A, B)',
        'subtraction': 'max(A, -B)',
        'smooth_union': '-log(e^(-kA) + e^(-kB))/k'
    }
    formula = op_formulas.get(operation, '')
    axes[3].text(0.5, 0.5, f'Formula: {formula}', fontsize=14, ha='center', va='center',
                 transform=axes[3].transAxes)

    plt.tight_layout()
    return fig, axes


def plot_multi_object_scene(
    sdf_fn: Callable,
    object_poses: List[Tuple[torch.Tensor, torch.Tensor]],
    object_labels: List[str] = None,
    resolution: int = 100,
    bounds: Tuple[float, float] = (-1.5, 1.5),
    device: torch.device = None,
    style: PlotStyle = None,
):
    """
    Visualize a multi-object scene with object positions marked.

    Args:
        sdf_fn: Composed scene SDF function
        object_poses: List of (translation, quaternion) tuples
        object_labels: Labels for each object
        resolution: Grid resolution
        bounds: Grid bounds
        device: Device for computation
        style: PlotStyle configuration

    Returns:
        Tuple of (figure, axis)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    fig, ax = plt.subplots(1, 1, figsize=style.figsize, dpi=style.dpi)

    # Plot SDF slice
    plot_sdf_slice(sdf_fn, resolution, 'z', 0, bounds, device, ax=ax, style=style)

    # Mark object positions
    colors = plt.cm.tab10.colors
    for i, (trans, quat) in enumerate(object_poses):
        pos = _ensure_numpy(trans)
        label = object_labels[i] if object_labels else f'Object {i+1}'
        ax.scatter([pos[0]], [pos[1]], s=100, c=[colors[i % len(colors)]],
                   marker='*', edgecolors='white', linewidths=2, zorder=10)
        ax.annotate(label, (pos[0], pos[1]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    color=colors[i % len(colors)])

    ax.set_title('Multi-Object Scene', fontsize=style.font_size + 2)
    return fig, ax


# =============================================================================
# Latent Space Visualization
# =============================================================================

def plot_latent_interpolation(
    images: List[torch.Tensor],
    t_values: List[float],
    style: PlotStyle = None,
    title: str = 'Latent Space Interpolation',
):
    """
    Visualize shape interpolation in latent space.

    Args:
        images: List of rendered images or SDF slices
        t_values: Interpolation parameters
        style: PlotStyle configuration
        title: Overall title

    Returns:
        Tuple of (figure, axes)
    """
    plt = _ensure_matplotlib()
    style = style or DEFAULT_STYLE

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), dpi=style.dpi)
    if n == 1:
        axes = [axes]

    for ax, img, t in zip(axes, images, t_values):
        img = _ensure_numpy(img)
        if img.ndim == 2:
            # SDF slice
            vmax = max(abs(img.min()), abs(img.max()), 0.01)
            ax.imshow(img, cmap=style.cmap_diverging, vmin=-vmax, vmax=vmax)
            ax.contour(img, levels=[0], colors='black', linewidths=2)
        else:
            # RGB image
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f't = {t:.2f}', fontsize=style.font_size)
        ax.axis('off')

    plt.suptitle(title, fontsize=style.font_size + 4)
    plt.tight_layout()
    return fig, axes


# =============================================================================
# Animation and Video Export
# =============================================================================

def render_turntable(
    render_fn: Callable,
    num_frames: int = 36,
    elevation: float = 30,
    distance: float = 2.0,
) -> List[np.ndarray]:
    """
    Render a turntable animation.

    Args:
        render_fn: Function(camera_pos, camera_quat) -> RGB image
        num_frames: Number of frames
        elevation: Camera elevation angle in degrees
        distance: Distance from origin

    Returns:
        List of RGB images as numpy arrays
    """
    from pga_inr.utils.quaternion import quaternion_from_axis_angle

    frames = []
    axis_y = torch.tensor([0.0, 1.0, 0.0])

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames

        # Camera position
        x = distance * np.cos(angle) * np.cos(np.radians(elevation))
        y = distance * np.sin(np.radians(elevation))
        z = distance * np.sin(angle) * np.cos(np.radians(elevation))

        camera_pos = torch.tensor([x, y, z])

        # Camera rotation (look at origin)
        camera_quat = quaternion_from_axis_angle(
            axis_y.unsqueeze(0),
            torch.tensor([[-angle]])
        ).squeeze()

        # Render
        frame = render_fn(camera_pos, camera_quat)
        frames.append(_ensure_numpy(frame))

    return frames


def save_animation_gif(
    frames: List[np.ndarray],
    filepath: str,
    fps: int = 15,
    loop: int = 0,
):
    """
    Save frames as an animated GIF.

    Args:
        frames: List of RGB images
        filepath: Output file path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for GIF export. "
            "Install with: pip install imageio"
        )

    # Ensure frames are uint8
    frames_uint8 = []
    for frame in frames:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        frames_uint8.append(frame)

    imageio.mimsave(filepath, frames_uint8, fps=fps, loop=loop)


def create_animation_figure(
    update_fn: Callable,
    num_frames: int,
    interval: int = 50,
    style: PlotStyle = None,
):
    """
    Create a matplotlib animation.

    Args:
        update_fn: Function(frame_idx, ax) that updates the plot
        num_frames: Total number of frames
        interval: Milliseconds between frames
        style: PlotStyle configuration

    Returns:
        Matplotlib animation object
    """
    plt = _ensure_matplotlib()
    from matplotlib.animation import FuncAnimation

    style = style or DEFAULT_STYLE
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    def update(frame_idx):
        ax.clear()
        update_fn(frame_idx, ax)
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    return anim, fig, ax


# =============================================================================
# Mesh Utilities (kept from original)
# =============================================================================

def save_mesh(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    filepath: str,
    vertex_colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    vertex_normals: Optional[Union[torch.Tensor, np.ndarray]] = None,
):
    """
    Save a mesh to file.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        filepath: Output file path
        vertex_colors: (V, 3) or (V, 4) vertex colors
        vertex_normals: (V, 3) vertex normals
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh required for mesh saving")

    vertices = _ensure_numpy(vertices)
    faces = _ensure_numpy(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if vertex_colors is not None:
        vertex_colors = _ensure_numpy(vertex_colors)
        if vertex_colors.shape[-1] == 3:
            alpha = np.ones((len(vertex_colors), 1))
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=-1)
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

    if vertex_normals is not None:
        vertex_normals = _ensure_numpy(vertex_normals)
        mesh.vertex_normals = vertex_normals

    mesh.export(filepath)


def extract_mesh_from_sdf(
    sdf_fn: Callable,
    resolution: int = 64,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from SDF using marching cubes.

    Args:
        sdf_fn: SDF function
        resolution: Grid resolution
        bounds: Grid bounds
        level: Isosurface level
        device: Device for computation

    Returns:
        Tuple of (vertices, faces)
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image required for mesh extraction")

    device = device or torch.device('cpu')

    # Create 3D grid
    coords = torch.linspace(bounds[0], bounds[1], resolution, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

    # Evaluate SDF
    with torch.no_grad():
        outputs = sdf_fn(points)
        if isinstance(outputs, dict):
            sdf = outputs.get('sdf', outputs.get('density'))
        else:
            sdf = outputs
        sdf = sdf.squeeze()

    sdf = sdf.reshape(resolution, resolution, resolution).cpu().numpy()

    # Marching cubes
    spacing = (bounds[1] - bounds[0]) / (resolution - 1)
    verts, faces, normals, values = measure.marching_cubes(
        sdf, level=level, spacing=(spacing,) * 3
    )
    verts = verts + bounds[0]

    return verts, faces


# =============================================================================
# Quick Plot Functions (Convenience)
# =============================================================================

def quick_sdf_plot(model, device=None, resolution=100, save_path=None):
    """Quick visualization of an SDF model."""
    plt = _ensure_matplotlib()
    device = device or torch.device('cpu')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (axis, value) in enumerate([('z', 0), ('y', 0), ('x', 0)]):
        plot_sdf_slice(model, resolution, axis, value, device=device, ax=axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig, axes


def quick_3d_plot(model, device=None, resolution=48, save_path=None):
    """Quick 3D visualization of an SDF model."""
    plt = _ensure_matplotlib()
    device = device or torch.device('cpu')

    fig, ax = plot_sdf_3d_isosurface(model, resolution=resolution, device=device)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig, ax
