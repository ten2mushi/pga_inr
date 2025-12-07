"""
Generative PGA-INR with HyperNetwork conditioning.

This module implements the generative extension of PGA-INR that disentangles:
- Intrinsic shape (latent code z)
- Extrinsic state (PGA motor M)

This enables:
- Shape interpolation in latent space
- Multi-object scene composition
- Zero-shot pose generalization

All models inherit from BaseGenerativeINR and follow the standard interface:
- forward() returns Dict[str, Tensor]
- Standard output keys: 'sdf'/'density', 'rgb', 'normal', 'local_coords'
"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseGenerativeINR
from ..core.constants import (
    DEFAULT_HIDDEN_FEATURES,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_OMEGA_0,
    DEFAULT_LATENT_DIM,
    OUTPUT_SDF,
    OUTPUT_DENSITY,
    OUTPUT_RGB,
    OUTPUT_NORMAL,
    OUTPUT_LOCAL_COORDS,
)
from ..core.types import TensorDict, ObserverPose

from .layers import PGAMotorLayer, HyperLayer, SirenHyperNetwork


class Generative_PGA_INR(BaseGenerativeINR):
    """
    Generative PGA-INR with HyperNetwork conditioning.

    Instead of fixed network weights, the weights are generated from a
    latent code z by a HyperNetwork. This allows learning a continuous
    manifold of shapes.

    Inherits from BaseGenerativeINR for consistent API.

    Architecture:
        1. PGAMotorLayer: Transform points to local frame (handles extrinsic pose)
        2. HyperNetwork: Generate network weights from latent code z
        3. FunctionalSiren: Apply generated weights to process coordinates
        4. Output: Density + RGB (or SDF + normal)

    Standard Output Keys:
        - 'density': Density values (B, N, 1)
        - 'rgb': RGB colors (B, N, 3)
        - 'local_coords': Points in local frame (B, N, 3)
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES // 4,  # 64
        hidden_layers: int = 2,
        omega_0: float = DEFAULT_OMEGA_0,
        output_dim: int = 4,  # density (1) + RGB (3)
    ):
        """
        Args:
            latent_dim: Dimension of the shape latent code
            hidden_features: Width of generated network layers
            hidden_layers: Number of hidden layers in generated network
            omega_0: SIREN frequency parameter
            output_dim: Output dimension (4 for density+RGB, 4 for SDF+normal)
        """
        super().__init__()

        self._latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.omega_0 = omega_0
        self.output_dim = output_dim

        # 1. PGA Motor for coordinate-free transformation
        self.pga_motor = PGAMotorLayer()

        # 2. Define target network structure
        # Input: 3D coordinates, Output: density + features
        self.layer_shapes = [(3, hidden_features)]
        for _ in range(hidden_layers):
            self.layer_shapes.append((hidden_features, hidden_features))
        self.layer_shapes.append((hidden_features, output_dim))

        # 3. HyperNetwork to generate weights
        self.hyper_net = SirenHyperNetwork(
            latent_dim=latent_dim,
            target_shapes=self.layer_shapes,
            hidden_dim=256,
            omega_0=omega_0,
        )

        # 4. Functional layers (apply generated weights)
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_shapes) - 1):
            # Hidden layers use sin activation
            self.layers.append(HyperLayer(activation=torch.sin))
        # Final layer has no activation
        self.layers.append(HyperLayer(activation=None))

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim

    def get_output_type(self) -> str:
        """Return 'density' as primary output type."""
        return OUTPUT_DENSITY

    def _forward_raw(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal forward pass returning raw outputs.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Tuple (translation, quaternion) for object pose
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Tuple of (raw_output, local_coords)
        """
        B, N, _ = query_points.shape

        # A. Extrinsic disentanglement (PGA)
        # Transform points to object's local frame
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # B. Intrinsic conditioning (HyperNet)
        # Generate network weights from latent code
        weights, biases = self.hyper_net(latent_code)

        # C. Forward through generated network
        x = local_points
        for i, layer in enumerate(self.layers):
            x = layer(x, weights[i], biases[i])

        return x, local_points

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """
        Forward pass returning a dictionary of outputs.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Tuple (translation, quaternion) for object pose
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Dictionary with standard keys:
                - 'density': Density values (B, N, 1)
                - 'rgb': RGB colors (B, N, 3)
                - 'local_coords': Points in local frame (B, N, 3)
        """
        output, local_coords = self._forward_raw(
            query_points, observer_pose, latent_code
        )

        # Split output: first channel is density, rest is RGB
        density = output[..., :1]
        rgb = torch.sigmoid(output[..., 1:4])

        return {
            OUTPUT_DENSITY: density,
            OUTPUT_RGB: rgb,
            OUTPUT_LOCAL_COORDS: local_coords,
        }

    # Alias for backwards compatibility
    def forward_dict(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """Deprecated: Use forward() directly. This method is kept for backwards compatibility."""
        return self.forward(query_points, observer_pose, latent_code)


class LatentCodeBank(nn.Module):
    """
    Learnable latent codes for auto-decoding training.

    Instead of using an encoder network, each object in the dataset
    gets a trainable latent vector that is optimized alongside the
    network weights during training.

    This is the approach used in DeepSDF.
    """

    def __init__(
        self,
        num_objects: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        init_std: float = 0.01,
    ):
        """
        Args:
            num_objects: Number of objects in the training set
            latent_dim: Dimension of each latent code
            init_std: Standard deviation for initialization
        """
        super().__init__()

        self.num_objects = num_objects
        self.latent_dim = latent_dim

        # Embedding layer for latent codes
        self.codes = nn.Embedding(num_objects, latent_dim)

        # Initialize with small random values
        nn.init.normal_(self.codes.weight, mean=0, std=init_std)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up latent codes by index.

        Args:
            indices: Object indices (B,) or (B, 1)

        Returns:
            Latent codes (B, latent_dim)
        """
        if indices.dim() > 1:
            indices = indices.squeeze(-1)
        return self.codes(indices)

    def get_all_codes(self) -> torch.Tensor:
        """Get all latent codes as a tensor."""
        return self.codes.weight

    def interpolate(self, idx1: int, idx2: int, t: float) -> torch.Tensor:
        """
        Linearly interpolate between two latent codes.

        Args:
            idx1: First object index
            idx2: Second object index
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated latent code (latent_dim,)
        """
        z1 = self.codes.weight[idx1]
        z2 = self.codes.weight[idx2]
        return (1 - t) * z1 + t * z2


class Generative_PGA_INR_SDF(BaseGenerativeINR):
    """
    Generative PGA-INR specialized for SDF output.

    Inherits from BaseGenerativeINR for consistent API.

    Standard Output Keys:
        - 'sdf': SDF values (B, N, 1)
        - 'normal': Predicted normals (B, N, 3)
        - 'local_coords': Points in local frame (B, N, 3)
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        omega_0: float = DEFAULT_OMEGA_0,
    ):
        """
        Args:
            latent_dim: Dimension of the shape latent code
            hidden_features: Width of generated network layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
        """
        super().__init__()

        self._latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.omega_0 = omega_0

        # 1. PGA Motor for coordinate-free transformation
        self.pga_motor = PGAMotorLayer()

        # 2. Define target network structure
        # Output: SDF (1) + normal (3)
        self.layer_shapes = [(3, hidden_features)]
        for _ in range(hidden_layers):
            self.layer_shapes.append((hidden_features, hidden_features))
        self.layer_shapes.append((hidden_features, 4))  # SDF + normal

        # 3. HyperNetwork to generate weights
        self.hyper_net = SirenHyperNetwork(
            latent_dim=latent_dim,
            target_shapes=self.layer_shapes,
            hidden_dim=256,
            omega_0=omega_0,
        )

        # 4. Functional layers (apply generated weights)
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_shapes) - 1):
            self.layers.append(HyperLayer(activation=torch.sin))
        self.layers.append(HyperLayer(activation=None))

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim

    def get_output_type(self) -> str:
        """Return 'sdf' as primary output type."""
        return OUTPUT_SDF

    def _forward_raw(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal forward pass."""
        B, N, _ = query_points.shape

        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        weights, biases = self.hyper_net(latent_code)

        x = local_points
        for i, layer in enumerate(self.layers):
            x = layer(x, weights[i], biases[i])

        return x, local_points

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """
        Forward pass returning SDF and normals.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Tuple (translation, quaternion) for object pose
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Dictionary with standard keys:
                - 'sdf': SDF values (B, N, 1)
                - 'normal': Surface normals (B, N, 3)
                - 'local_coords': Points in local frame (B, N, 3)
        """
        output, local_coords = self._forward_raw(
            query_points, observer_pose, latent_code
        )

        sdf = output[..., :1]
        normals = F.normalize(output[..., 1:4], dim=-1)

        return {
            OUTPUT_SDF: sdf,
            OUTPUT_NORMAL: normals,
            OUTPUT_LOCAL_COORDS: local_coords,
        }


class ConditionalPGA_INR(BaseGenerativeINR):
    """
    PGA-INR with FiLM-style conditioning instead of full HyperNetwork.

    Uses a smaller conditioning network that modulates hidden activations
    via scale and shift operations. More parameter-efficient than full
    weight generation.

    Inherits from BaseGenerativeINR for consistent API.

    Standard Output Keys:
        - 'density': Density values (B, N, 1)
        - 'rgb': RGB colors (B, N, 3)
        - 'local_coords': Points in local frame (B, N, 3)
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        omega_0: float = DEFAULT_OMEGA_0,
    ):
        """
        Args:
            latent_dim: Dimension of the shape latent code
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
        """
        super().__init__()

        from .layers.sine import ModulatedSirenMLP

        self._latent_dim = latent_dim
        self.hidden_features = hidden_features

        self.pga_motor = PGAMotorLayer()

        # Main network with modulation
        self.siren = ModulatedSirenMLP(
            in_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=4,  # density + RGB
            omega_0=omega_0,
        )

        # Conditioning network: generates (gamma, beta) for each layer
        self.num_layers = hidden_layers + 1
        self.condition_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Per-layer modulation generators
        self.gamma_nets = nn.ModuleList(
            [nn.Linear(256, hidden_features) for _ in range(self.num_layers)]
        )
        self.beta_nets = nn.ModuleList(
            [nn.Linear(256, hidden_features) for _ in range(self.num_layers)]
        )

        # Initialize to identity modulation
        for g, b in zip(self.gamma_nets, self.beta_nets):
            nn.init.zeros_(g.weight)
            nn.init.ones_(g.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim

    def get_output_type(self) -> str:
        """Return 'density' as primary output type."""
        return OUTPUT_DENSITY

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """Forward pass with FiLM conditioning."""
        B, N, _ = query_points.shape

        # Transform to local frame
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # Generate modulation parameters
        cond_features = self.condition_net(latent_code)

        modulations = []
        for gamma_net, beta_net in zip(self.gamma_nets, self.beta_nets):
            gamma = gamma_net(cond_features).unsqueeze(1)  # (B, 1, hidden)
            beta = beta_net(cond_features).unsqueeze(1)  # (B, 1, hidden)
            modulations.append((gamma, beta))

        # Forward with modulation
        output = self.siren(local_points, modulations)

        density = output[..., :1]
        rgb = torch.sigmoid(output[..., 1:4])

        return {
            OUTPUT_DENSITY: density,
            OUTPUT_RGB: rgb,
            OUTPUT_LOCAL_COORDS: local_points,
        }


def create_shape_manifold(
    model: BaseGenerativeINR,
    latent_codes: List[torch.Tensor],
    grid_resolution: int = 64,
    bounds: Tuple[float, float] = (-1, 1),
) -> List[torch.Tensor]:
    """
    Create SDF grids for a list of latent codes.

    Useful for visualization and analysis of the learned shape space.

    Args:
        model: Trained generative model (must inherit from BaseGenerativeINR)
        latent_codes: List of latent codes to evaluate
        grid_resolution: Resolution of output SDF grid
        bounds: Spatial bounds

    Returns:
        List of SDF tensors, each of shape (res, res, res)
    """
    device = next(model.parameters()).device

    # Create evaluation grid
    coords_1d = torch.linspace(bounds[0], bounds[1], grid_resolution, device=device)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    grid_points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

    sdfs = []
    model.eval()

    with torch.no_grad():
        for z_code in latent_codes:
            if z_code.dim() == 1:
                z_code = z_code.unsqueeze(0)

            output = model.forward(grid_points, None, z_code)
            sdf = output.get(OUTPUT_SDF, output.get(OUTPUT_DENSITY))
            sdf = sdf.reshape(grid_resolution, grid_resolution, grid_resolution)
            sdfs.append(sdf.cpu())

    return sdfs


def interpolate_shapes(
    model: BaseGenerativeINR,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int = 10,
    interpolation: str = "linear",
) -> List[torch.Tensor]:
    """
    Interpolate between two shapes in latent space.

    Args:
        model: Trained generative model
        z1: Start latent code
        z2: End latent code
        num_steps: Number of interpolation steps
        interpolation: 'linear' or 'spherical'

    Returns:
        List of latent codes along interpolation path
    """
    interpolated = []

    for i in range(num_steps):
        t = i / (num_steps - 1)

        if interpolation == "linear":
            z = (1 - t) * z1 + t * z2
        elif interpolation == "spherical":
            # Spherical linear interpolation
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            omega = torch.acos(torch.clamp((z1_norm * z2_norm).sum(), -1, 1))

            if omega.abs() < 1e-6:
                z = z1
            else:
                z = (
                    torch.sin((1 - t) * omega) * z1 + torch.sin(t * omega) * z2
                ) / torch.sin(omega)
        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

        interpolated.append(z)

    return interpolated


# =============================================================================
# Motor Field Networks for Non-Rigid Deformation
# =============================================================================


class MotorFieldINR(BaseGenerativeINR):
    """
    Predict per-point SE(3) motors for non-rigid deformation.

    A motor field network learns to map spatial coordinates to rigid
    transformations (rotation + translation). This enables:
    - Local non-rigid deformations composed from local rigid motions
    - Articulated body modeling (bones as regions of coherent transformation)
    - Smooth deformation fields with geometric regularity

    The motor at each point describes how to locally transform space.
    By applying different motors at different locations, we can model
    complex non-rigid deformations while maintaining local rigidity.

    Output Keys:
        - 'motor': Full motor parameters (B, N, 8) - rotation quaternion + translation
        - 'rotation': Rotation quaternions (B, N, 4)
        - 'translation': Translation vectors (B, N, 3)
        - 'local_coords': Input points in local frame (B, N, 3)
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        omega_0: float = DEFAULT_OMEGA_0,
        motor_representation: str = "se3",
        scale_translation: float = 1.0,
    ):
        """
        Args:
            latent_dim: Dimension of conditioning latent code
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            omega_0: SIREN frequency parameter
            motor_representation: 'se3' for quaternion+translation, 'so3' for rotation only
            scale_translation: Scale factor for translation predictions
        """
        super().__init__()

        from .layers.sine import SirenMLP

        self._latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.motor_representation = motor_representation
        self.scale_translation = scale_translation

        self.pga_motor = PGAMotorLayer()

        # Input: 3D coords + latent code
        input_dim = 3 + latent_dim

        # Output dimension based on representation
        if motor_representation == "se3":
            output_dim = 7  # quaternion (4) + translation (3)
        elif motor_representation == "so3":
            output_dim = 4  # quaternion only
        else:
            raise ValueError(f"Unknown motor_representation: {motor_representation}")

        self.network = SirenMLP(
            in_features=input_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=output_dim,
            omega_0=omega_0,
        )

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim

    def get_output_type(self) -> str:
        """Return 'motor' as primary output type."""
        return "motor"

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
    ) -> TensorDict:
        """
        Forward pass predicting motor field.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Optional observer pose for frame transformation
            latent_code: Conditioning code (B, latent_dim)

        Returns:
            Dictionary with motor field outputs
        """
        B, N, _ = query_points.shape

        # Transform to local frame if needed
        if observer_pose is not None:
            local_points = self.pga_motor(query_points, observer_pose)
        else:
            local_points = query_points

        # Expand latent code to all points
        latent_expanded = latent_code.unsqueeze(1).expand(-1, N, -1)

        # Concatenate position and latent
        x = torch.cat([local_points, latent_expanded], dim=-1)

        # Predict motor
        output = self.network(x)

        # Parse output based on representation
        if self.motor_representation == "se3":
            # Quaternion (normalized) + translation
            quat_raw = output[..., :4]
            quat = F.normalize(quat_raw, dim=-1)
            translation = output[..., 4:7] * self.scale_translation
            motor = torch.cat([quat, translation], dim=-1)
        else:
            # Rotation only
            quat = F.normalize(output[..., :4], dim=-1)
            translation = torch.zeros_like(local_points)
            motor = quat

        return {
            "motor": motor,
            "rotation": quat,
            "translation": translation,
            OUTPUT_LOCAL_COORDS: local_points,
        }

    def deform_points(
        self,
        canonical_points: torch.Tensor,
        latent_code: torch.Tensor,
        observer_pose: Optional[ObserverPose] = None,
    ) -> torch.Tensor:
        """
        Apply motor field to deform canonical points.

        Args:
            canonical_points: Points in canonical space (B, N, 3)
            latent_code: Deformation conditioning (B, latent_dim)
            observer_pose: Optional global pose

        Returns:
            Deformed points (B, N, 3)
        """
        # Get motor field
        output = self.forward(canonical_points, observer_pose, latent_code)
        quat = output["rotation"]
        trans = output["translation"]

        # Apply transformation: R * p + t
        # Quaternion rotation
        deformed = self._quaternion_rotate(canonical_points, quat) + trans

        return deformed

    def _quaternion_rotate(
        self,
        points: torch.Tensor,
        quaternions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate points using quaternions.

        Args:
            points: Points to rotate (B, N, 3)
            quaternions: Unit quaternions [w, x, y, z] (B, N, 4)

        Returns:
            Rotated points (B, N, 3)
        """
        # Extract quaternion components
        w, x, y, z = quaternions[..., 0:1], quaternions[..., 1:2], quaternions[..., 2:3], quaternions[..., 3:4]
        px, py, pz = points[..., 0:1], points[..., 1:2], points[..., 2:3]

        # Quaternion rotation formula: q * p * q^-1
        # Optimized computation
        rx = px * (1 - 2*y*y - 2*z*z) + py * (2*x*y - 2*w*z) + pz * (2*x*z + 2*w*y)
        ry = px * (2*x*y + 2*w*z) + py * (1 - 2*x*x - 2*z*z) + pz * (2*y*z - 2*w*x)
        rz = px * (2*x*z - 2*w*y) + py * (2*y*z + 2*w*x) + pz * (1 - 2*x*x - 2*y*y)

        return torch.cat([rx, ry, rz], dim=-1)

    def inverse_deform(
        self,
        deformed_points: torch.Tensor,
        latent_code: torch.Tensor,
        num_iterations: int = 5,
        observer_pose: Optional[ObserverPose] = None,
    ) -> torch.Tensor:
        """
        Find canonical points that map to given deformed points.

        Uses iterative refinement since the inverse is not analytical
        for non-linear motor fields.

        Args:
            deformed_points: Target deformed points (B, N, 3)
            latent_code: Deformation conditioning (B, latent_dim)
            num_iterations: Number of refinement iterations
            observer_pose: Optional global pose

        Returns:
            Estimated canonical points (B, N, 3)
        """
        # Initialize canonical points at deformed locations
        canonical = deformed_points.clone()

        for _ in range(num_iterations):
            # Forward map current estimate
            current_deformed = self.deform_points(canonical, latent_code, observer_pose)

            # Compute residual
            residual = deformed_points - current_deformed

            # Update canonical points (simplified gradient step)
            canonical = canonical + residual

        return canonical


class DeformableSDF(BaseGenerativeINR):
    """
    SDF with motor field deformation.

    Combines a canonical SDF (representing the base shape) with a
    motor field (representing pose-dependent deformation). This enables
    modeling of articulated or non-rigidly deforming objects.

    The canonical SDF is evaluated at deformed coordinates, effectively
    warping the shape according to the motor field.

    Output Keys:
        - 'sdf': SDF values at query points (B, N, 1)
        - 'normal': Surface normals (B, N, 3)
        - 'deformation': Per-point deformation vectors (B, N, 3)
        - 'canonical_points': Points in canonical space (B, N, 3)
    """

    def __init__(
        self,
        canonical_sdf: BaseGenerativeINR,
        motor_field: MotorFieldINR,
    ):
        """
        Args:
            canonical_sdf: SDF model for the canonical shape
            motor_field: Motor field for deformation
        """
        super().__init__()

        self.canonical_sdf = canonical_sdf
        self.motor_field = motor_field

    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self.motor_field.latent_dim

    def get_output_type(self) -> str:
        """Return 'sdf' as primary output type."""
        return OUTPUT_SDF

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[ObserverPose],
        latent_code: torch.Tensor,
        canonical_latent: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        """
        Forward pass through deformable SDF.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Optional observer pose
            latent_code: Deformation latent code (B, latent_dim)
            canonical_latent: Optional separate latent for canonical SDF

        Returns:
            Dictionary with SDF and deformation outputs
        """
        # Inverse deform query points to canonical space
        canonical_points = self.motor_field.inverse_deform(
            query_points, latent_code, observer_pose=observer_pose
        )

        # Evaluate canonical SDF
        if canonical_latent is None:
            # Use same latent for canonical SDF
            canonical_latent = latent_code

        sdf_output = self.canonical_sdf.forward(
            canonical_points, observer_pose, canonical_latent
        )

        # Compute deformation vectors
        deformation = query_points - canonical_points

        # Combine outputs
        result = {
            OUTPUT_SDF: sdf_output.get(OUTPUT_SDF, sdf_output.get(OUTPUT_DENSITY)),
            OUTPUT_NORMAL: sdf_output.get(OUTPUT_NORMAL),
            "deformation": deformation,
            "canonical_points": canonical_points,
        }

        return result


# =============================================================================
# Latent Space Arithmetic
# =============================================================================


class LatentSpaceArithmetic:
    """
    Shape manipulation via latent arithmetic operations.

    Provides tools for:
    - Finding semantic directions in latent space
    - Applying transformations to latent codes
    - Performing analogies (A:B :: C:D)
    - Style transfer between shapes

    This is inspired by the linear structure of GAN latent spaces,
    where arithmetic operations often correspond to semantic changes.
    """

    def __init__(
        self,
        latent_bank: LatentCodeBank,
        normalize: bool = False,
    ):
        """
        Args:
            latent_bank: Bank of learned latent codes
            normalize: Whether to normalize directions to unit length
        """
        self.latent_bank = latent_bank
        self.normalize = normalize

    def find_semantic_direction(
        self,
        positive_indices: List[int],
        negative_indices: List[int],
    ) -> torch.Tensor:
        """
        Find a semantic direction by contrasting positive and negative examples.

        Direction = mean(positive) - mean(negative)

        Args:
            positive_indices: Indices of shapes with desired attribute
            negative_indices: Indices of shapes without attribute

        Returns:
            Direction vector (latent_dim,)
        """
        codes = self.latent_bank.get_all_codes()

        positive_mean = codes[positive_indices].mean(dim=0)
        negative_mean = codes[negative_indices].mean(dim=0)

        direction = positive_mean - negative_mean

        if self.normalize:
            direction = F.normalize(direction, dim=-1)

        return direction

    def apply_direction(
        self,
        z: torch.Tensor,
        direction: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply a semantic direction to a latent code.

        Args:
            z: Base latent code (latent_dim,) or (B, latent_dim)
            direction: Semantic direction (latent_dim,)
            scale: Magnitude of transformation (can be negative for reversal)

        Returns:
            Modified latent code with same shape as z
        """
        return z + scale * direction

    def analogy(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform analogy: A is to B as C is to D.

        Finds D such that the relationship A->B is transferred to C->D.

        D = C + (B - A)

        Args:
            z_a: Source base (latent_dim,)
            z_b: Source target (latent_dim,)
            z_c: Destination base (latent_dim,)

        Returns:
            z_d: Destination target (latent_dim,)
        """
        relationship = z_b - z_a
        return z_c + relationship

    def linear_combination(
        self,
        indices: List[int],
        weights: List[float],
    ) -> torch.Tensor:
        """
        Create a new shape via linear combination of existing shapes.

        Args:
            indices: Indices of shapes to combine
            weights: Weights for each shape (will be normalized)

        Returns:
            Combined latent code (latent_dim,)
        """
        codes = self.latent_bank.get_all_codes()
        weights = torch.tensor(weights, device=codes.device)
        weights = weights / weights.sum()  # Normalize

        combined = sum(w * codes[i] for i, w in zip(indices, weights))
        return combined

    def project_to_manifold(
        self,
        z: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        Project an arbitrary latent to the learned manifold.

        Finds the convex combination of k nearest neighbors in the
        latent bank that best approximates z.

        Args:
            z: Arbitrary latent code (latent_dim,)
            k: Number of nearest neighbors to use

        Returns:
            Projected latent code (latent_dim,)
        """
        codes = self.latent_bank.get_all_codes()

        # Find k nearest neighbors
        distances = ((codes - z.unsqueeze(0)) ** 2).sum(dim=-1)
        _, nearest_idx = distances.topk(k, largest=False)

        # Compute weights based on inverse distance
        nearest_codes = codes[nearest_idx]
        nearest_dists = distances[nearest_idx]

        weights = 1.0 / (nearest_dists + 1e-8)
        weights = weights / weights.sum()

        # Weighted combination
        projected = (weights.unsqueeze(-1) * nearest_codes).sum(dim=0)

        return projected


class StyleTransfer:
    """
    Style transfer between shapes via latent decomposition.

    Decomposes latent codes into content (what) and style (how) components,
    enabling transfer of style attributes between different shapes.

    Assumes a learned separation where:
    - Early dimensions encode structure/content
    - Later dimensions encode style/appearance
    """

    def __init__(
        self,
        latent_dim: int,
        content_dim: Optional[int] = None,
        style_dim: Optional[int] = None,
    ):
        """
        Args:
            latent_dim: Total latent dimension
            content_dim: Dimension for content (defaults to latent_dim // 2)
            style_dim: Dimension for style (defaults to remaining)
        """
        self.latent_dim = latent_dim

        if content_dim is None:
            content_dim = latent_dim // 2
        if style_dim is None:
            style_dim = latent_dim - content_dim

        self.content_dim = content_dim
        self.style_dim = style_dim

        # Indices for content and style
        self.content_idx = slice(0, content_dim)
        self.style_idx = slice(content_dim, content_dim + style_dim)

    def decompose(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose a latent into content and style.

        Args:
            z: Full latent code (latent_dim,) or (B, latent_dim)

        Returns:
            Tuple of (content, style) tensors
        """
        content = z[..., self.content_idx]
        style = z[..., self.style_idx]
        return content, style

    def compose(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose content and style into a full latent.

        Args:
            content: Content component
            style: Style component

        Returns:
            Combined latent code
        """
        return torch.cat([content, style], dim=-1)

    def transfer(
        self,
        content_z: torch.Tensor,
        style_z: torch.Tensor,
        blend: float = 1.0,
    ) -> torch.Tensor:
        """
        Transfer style from one latent to another.

        Args:
            content_z: Latent providing content/structure
            style_z: Latent providing style/appearance
            blend: Blend factor (0 = keep original style, 1 = full transfer)

        Returns:
            New latent with transferred style
        """
        content, original_style = self.decompose(content_z)
        _, new_style = self.decompose(style_z)

        # Blend styles
        blended_style = (1 - blend) * original_style + blend * new_style

        return self.compose(content, blended_style)

    def interpolate_style(
        self,
        z: torch.Tensor,
        styles: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        """
        Interpolate between multiple styles while keeping content.

        Args:
            z: Base latent providing content
            styles: List of latents providing styles
            weights: Weights for each style (will be normalized)

        Returns:
            Latent with interpolated style
        """
        content, _ = self.decompose(z)

        # Normalize weights
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        # Weighted combination of styles
        combined_style = sum(
            w * self.decompose(s)[1] for w, s in zip(weights, styles)
        )

        return self.compose(content, combined_style)
