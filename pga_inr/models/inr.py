"""
PGA-INR: Observer-Independent Implicit Neural Representation.

The core architecture that combines:
1. PGAMotorLayer: Transform points to object-local frame
2. SIREN backbone: Extract geometric features
3. Multi-head output: Density, color, normals

This network learns geometry in a canonical object frame, making it
invariant to observer position and rotation.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PGAMotorLayer, SineLayer, SirenMLP
from .encoders import PositionalEncoder, FourierEncoder


class PGA_INR(nn.Module):
    """
    Observer-Independent Implicit Neural Representation using PGA.

    The network learns to map local (object-frame) coordinates to
    geometric properties (density/SDF, color, normals).

    Key innovation: By transforming query points into the object's rest
    frame before processing, the network learns geometry independent of
    the observer's viewpoint.

    Architecture:
        1. PGAMotorLayer: Transform world points → local points
        2. Optional encoder: Positional/Fourier encoding
        3. SIREN backbone: Feature extraction
        4. Output heads:
           - Density/SDF (grade-0 scalar)
           - RGB color (grade-1 vector attribute)
           - Surface normal (grade-2 bivector, as vector)
    """

    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        output_normals: bool = True,
        use_encoder: bool = False,
        encoder_type: str = 'positional',
        encoder_frequencies: int = 6
    ):
        """
        Args:
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            output_normals: Whether to predict surface normals
            use_encoder: Whether to use positional encoding
            encoder_type: 'positional' or 'fourier'
            encoder_frequencies: Number of frequency bands
        """
        super().__init__()

        self.output_normals = output_normals

        # 1. PGA Motor Interface
        self.pga_motor = PGAMotorLayer()

        # 2. Optional positional encoder
        self.use_encoder = use_encoder
        if use_encoder:
            if encoder_type == 'positional':
                self.encoder = PositionalEncoder(
                    input_dim=3,
                    num_frequencies=encoder_frequencies,
                    include_input=True
                )
            else:
                self.encoder = FourierEncoder(
                    input_dim=3,
                    num_frequencies=encoder_frequencies * 2 * 3,
                    include_input=True
                )
            input_dim = self.encoder.output_dim
        else:
            self.encoder = None
            input_dim = 3

        # 3. SIREN backbone
        self.backbone = SirenMLP(
            in_features=input_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=hidden_features,
            omega_0=omega_0
        )

        # 4. Output heads
        # Density/SDF head (scalar, grade-0)
        self.density_head = nn.Linear(hidden_features, 1)

        # Color head (RGB, grade-1 vector attribute)
        self.color_head = nn.Linear(hidden_features, 3)

        # Normal head (grade-2 bivector, represented as 3D vector)
        if output_normals:
            self.normal_head = nn.Linear(hidden_features, 3)

        # Initialize output heads
        self._init_heads()

    def _init_heads(self):
        """Initialize output head weights."""
        # Small initialization for stability
        nn.init.xavier_uniform_(self.density_head.weight, gain=0.1)
        nn.init.zeros_(self.density_head.bias)

        nn.init.xavier_uniform_(self.color_head.weight, gain=0.1)
        nn.init.zeros_(self.color_head.bias)

        if self.output_normals:
            nn.init.xavier_uniform_(self.normal_head.weight, gain=0.1)
            nn.init.zeros_(self.normal_head.bias)

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_points: World-space coordinates of shape (B, N, 3)
            observer_pose: Tuple of (translation, quaternion) defining object pose.
                          - translation: (B, 3) or (3,)
                          - quaternion: (B, 4) or (4,) as [w, x, y, z]
                          If None, points are assumed to be in local frame.

        Returns:
            Dictionary with:
                - 'density': SDF/density values (B, N, 1)
                - 'rgb': Color values in [0, 1] (B, N, 3)
                - 'normal': Unit surface normals (B, N, 3) if output_normals=True
                - 'local_coords': Coordinates in local frame (B, N, 3)
        """
        # Step 1: Transform to local frame (observer-independent)
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # Step 2: Optional encoding
        if self.use_encoder:
            encoded = self.encoder(local_points)
        else:
            encoded = local_points

        # Step 3: Feature extraction
        features = self.backbone(encoded)

        # Step 4: Output heads
        density = self.density_head(features)
        rgb = torch.sigmoid(self.color_head(features))

        outputs = {
            'density': density,
            'rgb': rgb,
            'local_coords': local_points,
        }

        if self.output_normals:
            # Normals should be unit vectors
            normals = F.normalize(self.normal_head(features), dim=-1)
            outputs['normal'] = normals

        return outputs

    def forward_with_gradient(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that also computes the gradient of the density field.

        The gradient ∇f(x) is the true surface normal for SDFs.

        Args:
            query_points: World-space coordinates (B, N, 3)
            observer_pose: Object pose

        Returns:
            Dictionary with outputs plus 'gradient' (B, N, 3)
        """
        # Transform to local frame
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # Enable gradient computation
        local_points = local_points.requires_grad_(True)

        # Forward pass
        if self.use_encoder:
            encoded = self.encoder(local_points)
        else:
            encoded = local_points

        features = self.backbone(encoded)
        density = self.density_head(features)
        rgb = torch.sigmoid(self.color_head(features))

        # Compute gradient of density w.r.t. local coordinates
        gradient = torch.autograd.grad(
            outputs=density,
            inputs=local_points,
            grad_outputs=torch.ones_like(density),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        outputs = {
            'density': density,
            'rgb': rgb,
            'local_coords': local_points,
            'gradient': gradient,
        }

        if self.output_normals:
            normals = F.normalize(self.normal_head(features), dim=-1)
            outputs['normal'] = normals

        return outputs


class PGA_INR_SDF(PGA_INR):
    """
    PGA-INR specialized for Signed Distance Function (SDF) output.

    Removes color output and focuses on geometric representation.

    When geometric_init=True, uses a skip connection from input coordinates
    to output, allowing proper initialization to a unit sphere SDF.
    """

    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
        geometric_init: bool = True
    ):
        """
        Args:
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            geometric_init: Use geometric initialization for SDF
        """
        super().__init__(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            output_normals=True,
            use_encoder=False
        )

        # Remove color head
        del self.color_head

        # Store whether we're using geometric init
        self.geometric_init = geometric_init

        if geometric_init:
            self._geometric_init()

    def _geometric_init(self):
        """
        Initialize network to approximate a sphere SDF: f(x) = |x| - 1.

        We use a direct computation of |x| in the forward pass rather than
        trying to approximate it with a linear layer. The skip connection
        directly computes torch.norm(x) and adds a learned correction.

        We initialize:
        - Network output (density_head) to be approximately 0
        - Skip connection provides |x| - 1 directly
        """
        with torch.no_grad():
            # Initialize density head to output near-zero values
            # The skip connection will provide |x| - 1, and the network
            # learns corrections during training
            self.density_head.weight.zero_()
            self.density_head.bias.zero_()

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for SDF."""
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        features = self.backbone(local_points)

        network_sdf = self.density_head(features)

        if self.geometric_init:
            # Compute skip connection: exact |x| - 1 (unit sphere SDF)
            # This gives the network a good starting point
            point_norm = torch.norm(local_points, dim=-1, keepdim=True)
            skip = point_norm - 1.0
            sdf = network_sdf + skip
        else:
            sdf = network_sdf

        normals = F.normalize(self.normal_head(features), dim=-1)

        return {
            'sdf': sdf,
            'normal': normals,
            'local_coords': local_points,
        }


class PGA_INR_NeRF(PGA_INR):
    """
    PGA-INR for Neural Radiance Fields.

    Includes view-dependent color prediction.
    """

    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
        view_dependent: bool = True,
        view_features: int = 64
    ):
        """
        Args:
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            view_dependent: Whether color depends on view direction
            view_features: Features for view-dependent branch
        """
        super().__init__(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            output_normals=False,
            use_encoder=True,
            encoder_frequencies=10
        )

        self.view_dependent = view_dependent

        if view_dependent:
            # View direction encoder
            self.view_encoder = PositionalEncoder(
                input_dim=3,
                num_frequencies=4,
                include_input=True
            )

            # Replace color head with view-dependent version
            del self.color_head
            self.color_net = nn.Sequential(
                nn.Linear(hidden_features + self.view_encoder.output_dim, view_features),
                nn.ReLU(),
                nn.Linear(view_features, 3),
            )

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        view_dirs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional view direction.

        Args:
            query_points: World-space coordinates (B, N, 3)
            observer_pose: Object pose
            view_dirs: View directions (B, N, 3), required if view_dependent=True

        Returns:
            Dictionary with 'density', 'rgb', 'local_coords'
        """
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        encoded = self.encoder(local_points)
        features = self.backbone(encoded)

        density = self.density_head(features)

        if self.view_dependent and view_dirs is not None:
            # Encode view direction
            view_encoded = self.view_encoder(F.normalize(view_dirs, dim=-1))

            # Concatenate features and view encoding
            combined = torch.cat([features, view_encoded], dim=-1)
            rgb = torch.sigmoid(self.color_net(combined))
        else:
            # Use simple color prediction
            rgb = torch.sigmoid(self.color_head(features))

        return {
            'density': density,
            'rgb': rgb,
            'local_coords': local_points,
        }


def compose_scenes(
    query_points: torch.Tensor,
    models: list,
    poses: list,
    operation: str = 'union',
    blend_k: float = 32.0
) -> Dict[str, torch.Tensor]:
    """
    Compose multiple PGA-INR models into a scene.

    This demonstrates the compositionality advantage of PGA-INR:
    trained models can be placed at arbitrary poses without retraining.

    Supports both SDF models (with 'sdf' key) and regular INR models
    (with 'density' key). For SDF models, union uses min (closest surface),
    intersection uses max.

    Args:
        query_points: Query points in world space (B, N, 3)
        models: List of PGA_INR models
        poses: List of (translation, quaternion) tuples for each model
        operation: 'union' (max for density, min for SDF),
                  'intersection' (min for density, max for SDF),
                  or 'smooth_union'
        blend_k: Smoothness parameter for smooth_union (higher = sharper)

    Returns:
        Combined output dictionary with 'density' or 'sdf', optionally 'rgb',
        and 'object_idx'
    """
    if len(models) != len(poses):
        raise ValueError("Number of models must match number of poses")

    outputs_list = []
    for model, pose in zip(models, poses):
        out = model(query_points, pose)
        outputs_list.append(out)

    # Determine if using SDF or density models
    is_sdf = 'sdf' in outputs_list[0]
    value_key = 'sdf' if is_sdf else 'density'

    # Combine values
    values = torch.stack([o[value_key] for o in outputs_list], dim=0)

    if operation == 'union':
        if is_sdf:
            # For SDF: union = min (take closest surface)
            combined_value, idx = values.min(dim=0)
        else:
            # For density: union = max
            combined_value, idx = values.max(dim=0)
    elif operation == 'intersection':
        if is_sdf:
            # For SDF: intersection = max (take farthest surface)
            combined_value, idx = values.max(dim=0)
        else:
            # For density: intersection = min
            combined_value, idx = values.min(dim=0)
    elif operation == 'smooth_union':
        k = 1.0 / blend_k if blend_k > 0 else 32.0  # Convert blend_k to smoothness
        if is_sdf:
            # Smooth min for SDF
            combined_value = -torch.logsumexp(-values / k, dim=0) * k
            idx = values.argmin(dim=0)
        else:
            # Smooth max for density
            combined_value = torch.logsumexp(values / k, dim=0) * k
            idx = values.argmax(dim=0)
    elif operation == 'subtraction':
        # Boolean subtraction: A - B = A ∩ ¬B
        # For SDF: max(sdf_A, -sdf_B) - first model minus all others
        if len(models) != 2:
            raise ValueError("Subtraction requires exactly 2 models")
        if is_sdf:
            # Negate second model's SDF
            combined_value = torch.max(values[0], -values[1])
            # Index is 0 (first model) where we're inside it, 1 otherwise
            idx = (values[0] > -values[1]).long()
        else:
            # For density: min(density_A, 1 - density_B)
            combined_value = torch.min(values[0], 1 - values[1])
            idx = (values[0] < 1 - values[1]).long()
    else:
        raise ValueError(f"Unknown operation: {operation}")

    result = {
        value_key: combined_value,
        'object_idx': idx,
    }

    # Handle color if available
    if 'rgb' in outputs_list[0]:
        colors = torch.stack([o['rgb'] for o in outputs_list], dim=0)
        B, N, _ = query_points.shape
        idx_expanded = idx.expand(-1, -1, 3)
        combined_color = torch.gather(colors, 0, idx_expanded.unsqueeze(0)).squeeze(0)
        result['rgb'] = combined_color

    # Handle normals if available
    if 'normal' in outputs_list[0]:
        normals = torch.stack([o['normal'] for o in outputs_list], dim=0)
        B, N, _ = query_points.shape
        idx_expanded = idx.expand(-1, -1, 3)
        combined_normal = torch.gather(normals, 0, idx_expanded.unsqueeze(0)).squeeze(0)
        result['normal'] = combined_normal

    return result
