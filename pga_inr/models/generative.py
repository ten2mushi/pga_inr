"""
Generative PGA-INR with HyperNetwork conditioning.

This module implements the generative extension of PGA-INR that disentangles:
- Intrinsic shape (latent code z)
- Extrinsic state (PGA motor M)

This enables:
- Shape interpolation in latent space
- Multi-object scene composition
- Zero-shot pose generalization
"""

from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PGAMotorLayer, HyperLayer, HyperNetwork, SirenHyperNetwork


class Generative_PGA_INR(nn.Module):
    """
    Generative PGA-INR with HyperNetwork conditioning.

    Instead of fixed network weights, the weights are generated from a
    latent code z by a HyperNetwork. This allows learning a continuous
    manifold of shapes.

    Architecture:
        1. PGAMotorLayer: Transform points to local frame (handles extrinsic pose)
        2. HyperNetwork: Generate network weights from latent code z
        3. FunctionalSiren: Apply generated weights to process coordinates
        4. Output: Density + RGB (or SDF + normal)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        omega_0: float = 30.0,
        output_dim: int = 4  # density (1) + RGB (3)
    ):
        """
        Args:
            latent_dim: Dimension of the shape latent code
            hidden_features: Width of generated network layers
            hidden_layers: Number of hidden layers in generated network
            omega_0: SIREN frequency parameter
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_features = hidden_features
        self.omega_0 = omega_0

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
            omega_0=omega_0
        )

        # 4. Functional layers (apply generated weights)
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_shapes) - 1):
            # Hidden layers use sin activation
            self.layers.append(HyperLayer(activation=torch.sin))
        # Final layer has no activation
        self.layers.append(HyperLayer(activation=None))

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]],
        latent_code: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query_points: World coordinates (B, N, 3)
            observer_pose: Tuple (translation, quaternion) for object pose
            latent_code: Shape latent codes (B, latent_dim)

        Returns:
            Output tensor (B, N, output_dim) where output_dim = density + RGB
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

        return x

    def forward_dict(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]],
        latent_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning a dictionary of outputs.

        Returns:
            Dictionary with 'density', 'rgb', 'local_coords'
        """
        output = self.forward(query_points, observer_pose, latent_code)

        # Split output: first channel is density, rest is RGB
        density = output[..., :1]
        rgb = torch.sigmoid(output[..., 1:4])

        # Get local coordinates
        if observer_pose is not None:
            local_coords = self.pga_motor(query_points, observer_pose)
        else:
            local_coords = query_points

        return {
            'density': density,
            'rgb': rgb,
            'local_coords': local_coords,
        }


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
        latent_dim: int = 64,
        init_std: float = 0.01
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

    def interpolate(
        self,
        idx1: int,
        idx2: int,
        t: float
    ) -> torch.Tensor:
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


class Generative_PGA_INR_SDF(Generative_PGA_INR):
    """
    Generative PGA-INR specialized for SDF output.

    Output is SDF value + predicted normal.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        omega_0: float = 30.0
    ):
        super().__init__(
            latent_dim=latent_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            output_dim=4  # SDF (1) + normal (3)
        )

    def forward_dict(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]],
        latent_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning SDF and normals."""
        output = self.forward(query_points, observer_pose, latent_code)

        sdf = output[..., :1]
        normals = F.normalize(output[..., 1:4], dim=-1)

        if observer_pose is not None:
            local_coords = self.pga_motor(query_points, observer_pose)
        else:
            local_coords = query_points

        return {
            'sdf': sdf,
            'normal': normals,
            'local_coords': local_coords,
        }


class ConditionalPGA_INR(nn.Module):
    """
    PGA-INR with FiLM-style conditioning instead of full HyperNetwork.

    Uses a smaller conditioning network that modulates hidden activations
    via scale and shift operations. More parameter-efficient than full
    weight generation.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0
    ):
        super().__init__()

        from .layers.sine import ModulatedSirenMLP

        self.pga_motor = PGAMotorLayer()

        # Main network with modulation
        self.siren = ModulatedSirenMLP(
            in_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=4,  # density + RGB
            omega_0=omega_0
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
        self.gamma_nets = nn.ModuleList([
            nn.Linear(256, hidden_features)
            for _ in range(self.num_layers)
        ])
        self.beta_nets = nn.ModuleList([
            nn.Linear(256, hidden_features)
            for _ in range(self.num_layers)
        ])

        # Initialize to identity modulation
        for g, b in zip(self.gamma_nets, self.beta_nets):
            nn.init.zeros_(g.weight)
            nn.init.ones_(g.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)

    def forward(
        self,
        query_points: torch.Tensor,
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]],
        latent_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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
            beta = beta_net(cond_features).unsqueeze(1)    # (B, 1, hidden)
            modulations.append((gamma, beta))

        # Forward with modulation
        output = self.siren(local_points, modulations)

        density = output[..., :1]
        rgb = torch.sigmoid(output[..., 1:4])

        return {
            'density': density,
            'rgb': rgb,
            'local_coords': local_points,
        }


def create_shape_manifold(
    model: Generative_PGA_INR,
    latent_codes: List[torch.Tensor],
    grid_resolution: int = 64,
    bounds: Tuple[float, float] = (-1, 1)
) -> List[torch.Tensor]:
    """
    Create SDF grids for a list of latent codes.

    Useful for visualization and analysis of the learned shape space.

    Args:
        model: Trained generative model
        latent_codes: List of latent codes to evaluate
        grid_resolution: Resolution of output SDF grid
        bounds: Spatial bounds

    Returns:
        List of SDF tensors, each of shape (res, res, res)
    """
    device = next(model.parameters()).device

    # Create evaluation grid
    coords_1d = torch.linspace(bounds[0], bounds[1], grid_resolution, device=device)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    grid_points = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)

    sdfs = []
    model.eval()

    with torch.no_grad():
        for z_code in latent_codes:
            if z_code.dim() == 1:
                z_code = z_code.unsqueeze(0)

            output = model.forward_dict(grid_points, None, z_code)
            sdf = output.get('sdf', output.get('density'))
            sdf = sdf.reshape(grid_resolution, grid_resolution, grid_resolution)
            sdfs.append(sdf.cpu())

    return sdfs


def interpolate_shapes(
    model: Generative_PGA_INR,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int = 10,
    interpolation: str = 'linear'
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

        if interpolation == 'linear':
            z = (1 - t) * z1 + t * z2
        elif interpolation == 'spherical':
            # Spherical linear interpolation
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            omega = torch.acos(torch.clamp((z1_norm * z2_norm).sum(), -1, 1))

            if omega.abs() < 1e-6:
                z = z1
            else:
                z = (torch.sin((1 - t) * omega) * z1 + torch.sin(t * omega) * z2) / torch.sin(omega)
        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

        interpolated.append(z)

    return interpolated
