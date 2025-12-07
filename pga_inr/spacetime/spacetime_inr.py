"""
4D Spacetime Implicit Neural Representation.

Extends PGA-INR to handle time-varying geometry and dynamics.
"""

from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spacetime_PGA_INR(nn.Module):
    """
    4D Implicit Neural Representation with time-varying geometry.

    Input: (x, y, z, t) + observer pose + optional articulated skeleton
    Output: Density, color, normal, velocity field

    Applications:
    - Dynamic scene reconstruction
    - Human/animal motion capture
    - Physics simulation (fluid, cloth)
    """

    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
        time_encoding_freqs: int = 6,
        space_encoding_freqs: int = 10,
        output_velocity: bool = True
    ):
        """
        Args:
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            time_encoding_freqs: Fourier frequencies for time encoding
            space_encoding_freqs: Fourier frequencies for space encoding
            output_velocity: Whether to output velocity field
        """
        super().__init__()

        from ..models.layers import PGAMotorLayer, SirenMLP
        from ..models.encoders import FourierEncoder

        self.output_velocity = output_velocity

        # PGA motor for spatial transformation
        self.pga_motor = PGAMotorLayer()

        # Time encoding
        self.time_encoder = FourierEncoder(
            input_dim=1,
            num_frequencies=time_encoding_freqs,
            include_input=True
        )

        # Space encoding
        self.space_encoder = FourierEncoder(
            input_dim=3,
            num_frequencies=space_encoding_freqs,
            include_input=True
        )

        # Combined input dimension
        input_dim = self.space_encoder.output_dim + self.time_encoder.output_dim

        # SIREN backbone
        self.backbone = SirenMLP(
            in_features=input_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=hidden_features,
            omega_0=omega_0
        )

        # Output heads
        self.density_head = nn.Linear(hidden_features, 1)
        self.color_head = nn.Linear(hidden_features, 3)
        self.normal_head = nn.Linear(hidden_features, 3)

        if output_velocity:
            self.velocity_head = nn.Linear(hidden_features, 3)

        self._init_heads()

    def _init_heads(self):
        """Initialize output heads."""
        for head in [self.density_head, self.color_head, self.normal_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

        if self.output_velocity:
            nn.init.xavier_uniform_(self.velocity_head.weight, gain=0.1)
            nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        query_points: torch.Tensor,
        query_times: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_points: Spatial coordinates (B, N, 3)
            query_times: Time values (B, N, 1) or (B, 1)
            observer_pose: Optional (translation, quaternion) tuple

        Returns:
            Dictionary with density, rgb, normal, velocity, local_coords
        """
        B, N, _ = query_points.shape

        # Transform to local frame
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # Handle time broadcasting
        if query_times.shape[1] == 1:
            query_times = query_times.expand(B, N, 1)

        # Encode space and time
        space_encoded = self.space_encoder(local_points)
        time_encoded = self.time_encoder(query_times)

        # Concatenate
        features = torch.cat([space_encoded, time_encoded], dim=-1)

        # Forward through backbone
        features = self.backbone(features)

        # Output heads
        density = self.density_head(features)
        rgb = torch.sigmoid(self.color_head(features))
        normal = F.normalize(self.normal_head(features), dim=-1)

        outputs = {
            'density': density,
            'rgb': rgb,
            'normal': normal,
            'local_coords': local_points,
        }

        if self.output_velocity:
            velocity = self.velocity_head(features)
            outputs['velocity'] = velocity

        return outputs


class DeformableNeuralField(nn.Module):
    """
    Neural field with learned deformation.

    Canonical shape + time-varying deformation field.

    x_canonical = x_observed - D(x_observed, t)
    f(x) = f_canonical(x_canonical)
    """

    def __init__(
        self,
        canonical_field: nn.Module,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        deformation_scale: float = 0.5
    ):
        """
        Args:
            canonical_field: PGA-INR model for canonical shape
            hidden_features: Deformation network hidden features
            hidden_layers: Deformation network layers
            deformation_scale: Maximum deformation magnitude
        """
        super().__init__()

        from ..models.layers import SirenMLP

        self.canonical = canonical_field
        self.deformation_scale = deformation_scale

        # Deformation network: (x, t) -> displacement
        self.deformation_net = SirenMLP(
            in_features=4,  # (x, y, z, t)
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=3
        )

    def forward(
        self,
        query_points: torch.Tensor,
        query_times: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through deformable field.

        Args:
            query_points: Spatial coordinates (B, N, 3)
            query_times: Time values (B, N, 1) or (B, 1)
            observer_pose: Optional observer pose

        Returns:
            Dictionary with canonical field outputs plus deformation
        """
        B, N, _ = query_points.shape

        # Handle time broadcasting
        if query_times.shape[1] == 1:
            query_times = query_times.expand(B, N, 1)

        # Compute deformation
        xt = torch.cat([query_points, query_times], dim=-1)
        deformation = self.deformation_net(xt) * self.deformation_scale

        # Apply deformation to get canonical coordinates
        canonical_points = query_points - deformation

        # Query canonical field
        outputs = self.canonical(canonical_points, observer_pose)

        # Add deformation info
        outputs['deformation'] = deformation
        outputs['canonical_coords'] = canonical_points

        return outputs


class ArticulatedNeuralField(nn.Module):
    """
    Neural field with articulated skeleton deformation.

    Uses kinematic chain for structured deformation.
    """

    def __init__(
        self,
        canonical_field: nn.Module,
        kinematic_chain,
        blend_weights: Optional[torch.Tensor] = None,
        hidden_features: int = 64
    ):
        """
        Args:
            canonical_field: PGA-INR for canonical shape
            kinematic_chain: KinematicChain skeleton
            blend_weights: Precomputed blend weights (optional)
            hidden_features: Hidden features for weight prediction
        """
        super().__init__()

        from .kinematic_chain import ArticulatedMotor

        self.canonical = canonical_field
        self.skeleton = kinematic_chain
        self.articulated_motor = ArticulatedMotor(kinematic_chain)

        # If no precomputed weights, learn them
        if blend_weights is not None:
            self.register_buffer('blend_weights', blend_weights)
            self.weight_net = None
        else:
            # Network to predict blend weights from position
            num_joints = len(kinematic_chain.joints)
            self.weight_net = nn.Sequential(
                nn.Linear(3, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, num_joints),
                nn.Softmax(dim=-1)
            )

    def forward(
        self,
        query_points: torch.Tensor,
        query_times: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with articulated deformation.

        Args:
            query_points: Spatial coordinates (B, N, 3)
            query_times: Time values (B,) or scalar
            observer_pose: Optional observer pose

        Returns:
            Dictionary with outputs
        """
        from ..utils.quaternion import quaternion_to_matrix, quaternion_conjugate, quaternion_multiply

        B, N, _ = query_points.shape

        # Get global transforms at time t
        if query_times.dim() == 0:
            t = query_times
        else:
            t = query_times.mean()  # Use mean time for skeleton

        global_transforms = self.articulated_motor(t)

        # Get blend weights
        if self.weight_net is not None:
            weights = self.weight_net(query_points)  # (B, N, num_joints)
        else:
            # Use precomputed (would need to interpolate from vertices)
            weights = self.blend_weights

        # Compute inverse transformations
        canonical_points = torch.zeros_like(query_points)

        for i, name in enumerate(self.skeleton.joint_order):
            trans, quat = global_transforms[name]

            # Inverse transform
            # For rotation matrices, R^{-1} = R^T
            R = quaternion_to_matrix(quat)
            # Use transpose(-2, -1) to handle batched rotation matrices correctly
            # .T only works for 2D tensors; for (..., 3, 3) we need transpose on last 2 dims
            R_inv = R.transpose(-2, -1) if R.dim() > 2 else R.T
            t_inv = -torch.einsum('...ij,...j->...i', R_inv, trans)

            # Apply inverse transform
            points_local = torch.einsum('...ij,...nj->...ni', R_inv, query_points) + t_inv.unsqueeze(-2)

            # Weighted contribution
            w = weights[..., i:i+1]
            canonical_points = canonical_points + w * points_local

        # Query canonical field
        outputs = self.canonical(canonical_points, observer_pose)
        outputs['canonical_coords'] = canonical_points

        return outputs


class TemporalConsistencyLoss(nn.Module):
    """
    Loss for temporal consistency in dynamic fields.

    Encourages smooth motion and consistent geometry over time.
    """

    def __init__(
        self,
        lambda_velocity: float = 1.0,
        lambda_acceleration: float = 0.1,
        lambda_consistency: float = 0.5
    ):
        """
        Args:
            lambda_velocity: Weight for velocity smoothness
            lambda_acceleration: Weight for acceleration penalty
            lambda_consistency: Weight for temporal SDF consistency
        """
        super().__init__()

        self.lambda_velocity = lambda_velocity
        self.lambda_acceleration = lambda_acceleration
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        model: nn.Module,
        points: torch.Tensor,
        times: torch.Tensor,
        dt: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temporal consistency loss.

        Args:
            model: Spacetime model
            points: Query points (B, N, 3)
            times: Query times (B,)
            dt: Time step for finite differences

        Returns:
            (loss, metrics_dict)
        """
        B, N, _ = points.shape

        # Query at t, t-dt, t+dt
        times_center = times.view(B, 1, 1).expand(B, N, 1)
        times_prev = times_center - dt
        times_next = times_center + dt

        out_center = model(points, times_center)
        out_prev = model(points, times_prev)
        out_next = model(points, times_next)

        metrics = {}
        total_loss = 0.0

        # Velocity smoothness: velocity should change slowly
        if 'velocity' in out_center:
            vel_center = out_center['velocity']
            vel_prev = out_prev['velocity']
            vel_next = out_next['velocity']

            # Velocity change
            vel_change = (vel_next - vel_prev) / (2 * dt)
            vel_smooth_loss = (vel_change ** 2).mean()

            total_loss = total_loss + self.lambda_velocity * vel_smooth_loss
            metrics['vel_smooth'] = vel_smooth_loss.item()

            # Acceleration penalty
            acceleration = (vel_next - 2 * vel_center + vel_prev) / (dt ** 2)
            accel_loss = (acceleration ** 2).mean()

            total_loss = total_loss + self.lambda_acceleration * accel_loss
            metrics['acceleration'] = accel_loss.item()

        # SDF consistency: SDF should change according to velocity
        sdf_center = out_center.get('density', out_center.get('sdf'))
        sdf_next = out_next.get('density', out_next.get('sdf'))

        if 'velocity' in out_center:
            # Predicted SDF change from velocity
            velocity = out_center['velocity']

            # Gradient of SDF
            points.requires_grad_(True)
            sdf_for_grad = model(points, times_center).get('density', model(points, times_center).get('sdf'))
            grad = torch.autograd.grad(
                sdf_for_grad.sum(), points,
                create_graph=True
            )[0]

            # Expected SDF change: -v · ∇f (material derivative)
            expected_change = -(velocity * grad).sum(dim=-1, keepdim=True)
            actual_change = (sdf_next - sdf_center) / dt

            consistency_loss = ((expected_change - actual_change) ** 2).mean()
            total_loss = total_loss + self.lambda_consistency * consistency_loss
            metrics['consistency'] = consistency_loss.item()

        return total_loss, metrics


class FlowFieldLoss(nn.Module):
    """
    Loss for learning velocity/flow fields.

    Encourages divergence-free flow (incompressible) or
    other physical properties.
    """

    def __init__(
        self,
        lambda_divergence: float = 0.1,
        incompressible: bool = False
    ):
        """
        Args:
            lambda_divergence: Weight for divergence penalty
            incompressible: Whether to enforce incompressibility
        """
        super().__init__()

        self.lambda_divergence = lambda_divergence
        self.incompressible = incompressible

    def forward(
        self,
        velocity: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute flow field loss.

        Args:
            velocity: Velocity field (B, N, 3)
            coords: Coordinates (B, N, 3), must have requires_grad=True

        Returns:
            (loss, metrics_dict)
        """
        # Compute divergence: ∂u/∂x + ∂v/∂y + ∂w/∂z
        divergence = 0
        for i in range(3):
            grad_i = torch.autograd.grad(
                velocity[..., i].sum(), coords,
                create_graph=True
            )[0][..., i]
            divergence = divergence + grad_i

        metrics = {'divergence': divergence.abs().mean().item()}

        if self.incompressible:
            # Enforce zero divergence
            loss = self.lambda_divergence * (divergence ** 2).mean()
        else:
            # Just penalize large divergence
            loss = self.lambda_divergence * (divergence.abs()).mean()

        return loss, metrics


def render_dynamic_scene(
    model: nn.Module,
    camera_poses: torch.Tensor,
    times: torch.Tensor,
    object_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    image_size: int = 256,
    fov: float = 60.0
) -> torch.Tensor:
    """
    Render a sequence of frames from a dynamic scene.

    Args:
        model: Spacetime PGA-INR model
        camera_poses: Camera poses (T, 4, 4)
        times: Frame times (T,)
        object_pose: Object pose
        image_size: Image size
        fov: Field of view

    Returns:
        Rendered frames (T, H, W, 3)
    """
    from ..rendering.sphere_tracing import PGASphereTracer

    T = camera_poses.shape[0]
    device = next(model.parameters()).device

    frames = []

    tracer = PGASphereTracer(
        model=None,  # We'll query manually
        width=image_size,
        height=image_size,
        fov=fov
    )

    for i in range(T):
        t = times[i]
        camera_pose = camera_poses[i]

        # Generate rays
        origins, directions = tracer.generate_rays(camera_pose, device)

        # Trace (simplified - would need full implementation)
        # For now, use dense sampling
        near, far = 0.1, 5.0
        num_samples = 64

        t_vals = torch.linspace(near, far, num_samples, device=device)
        sample_points = origins.unsqueeze(1) + t_vals.view(1, -1, 1) * directions.unsqueeze(1)

        # Query model
        B, N_rays = origins.shape[0], 1
        sample_points = sample_points.view(1, -1, 3)
        time_tensor = torch.full((1, sample_points.shape[1], 1), t.item(), device=device)

        with torch.no_grad():
            outputs = model(sample_points, time_tensor, object_pose)

        # Simple volume rendering (approximate)
        density = outputs['density'].view(N_rays, num_samples)
        rgb = outputs['rgb'].view(N_rays, num_samples, 3)

        # Alpha compositing
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=device)])

        alpha = 1 - torch.exp(-density * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=1),
            dim=1
        )[:, :-1]

        # Final color
        color = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        color = color.view(image_size, image_size, 3)

        frames.append(color)

    return torch.stack(frames)
