"""
Time-varying motors for dynamic scenes.

Provides parameterizations of motor trajectories M(t):
- Keyframe interpolation
- Neural network parameterization
- Learned control signals
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMotor(nn.Module):
    """
    Motor as a function of time: M(t).

    Supports various parameterizations for time-varying rigid motions.
    """

    def __init__(
        self,
        parameterization: str = 'keyframe',
        num_keyframes: int = 10,
        hidden_dim: int = 64,
        interpolation: str = 'slerp'
    ):
        """
        Args:
            parameterization: 'keyframe', 'mlp', or 'fourier'
            num_keyframes: Number of keyframes (for keyframe mode)
            hidden_dim: Hidden dimension (for mlp mode)
            interpolation: Interpolation method for keyframes
        """
        super().__init__()

        self.parameterization = parameterization
        self.interpolation = interpolation

        if parameterization == 'keyframe':
            # Learnable keyframe poses
            self.translations = nn.Parameter(torch.zeros(num_keyframes, 3))
            self.quaternions = nn.Parameter(torch.zeros(num_keyframes, 4))
            # Initialize to identity rotation
            self.quaternions.data[:, 0] = 1.0

            # Learnable keyframe times (optional)
            self.keyframe_times = nn.Parameter(
                torch.linspace(0, 1, num_keyframes),
                requires_grad=False
            )

        elif parameterization == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 7)  # translation(3) + quaternion(4)
            )
            # Initialize output to identity
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        elif parameterization == 'fourier':
            self.num_frequencies = 8
            # Fourier coefficients for translation and rotation
            self.trans_coeffs = nn.Parameter(torch.zeros(self.num_frequencies * 2, 3))
            self.quat_coeffs = nn.Parameter(torch.zeros(self.num_frequencies * 2, 4))
            # DC component
            self.trans_dc = nn.Parameter(torch.zeros(3))
            self.quat_dc = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get motor parameters at time t.

        Args:
            t: Time values (...) in [0, 1]

        Returns:
            (translation, quaternion) each of shape (..., 3) and (..., 4)
        """
        if self.parameterization == 'keyframe':
            return self._interpolate_keyframes(t)
        elif self.parameterization == 'mlp':
            return self._mlp_forward(t)
        else:
            return self._fourier_forward(t)

    def _interpolate_keyframes(
        self,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between keyframes."""
        from .interpolation import quaternion_slerp

        # Normalize quaternions
        quaternions = F.normalize(self.quaternions, dim=-1)

        # Find surrounding keyframes
        num_keyframes = len(self.keyframe_times)

        # Handle batched t
        original_shape = t.shape
        t_flat = t.view(-1)

        translations_out = []
        quaternions_out = []

        for t_val in t_flat:
            # Find keyframe indices
            idx = torch.searchsorted(self.keyframe_times, t_val)
            idx = idx.clamp(1, num_keyframes - 1)

            idx_prev = idx - 1
            idx_next = idx

            # Local interpolation parameter
            t_prev = self.keyframe_times[idx_prev]
            t_next = self.keyframe_times[idx_next]
            local_t = (t_val - t_prev) / (t_next - t_prev + 1e-8)

            # Get keyframe values
            trans_prev = self.translations[idx_prev]
            trans_next = self.translations[idx_next]
            quat_prev = quaternions[idx_prev]
            quat_next = quaternions[idx_next]

            # Interpolate
            if self.interpolation == 'slerp':
                trans = (1 - local_t) * trans_prev + local_t * trans_next
                quat = quaternion_slerp(quat_prev, quat_next, local_t)
            else:
                trans = (1 - local_t) * trans_prev + local_t * trans_next
                quat = F.normalize((1 - local_t) * quat_prev + local_t * quat_next, dim=-1)

            translations_out.append(trans)
            quaternions_out.append(quat)

        translations_out = torch.stack(translations_out).view(*original_shape, 3)
        quaternions_out = torch.stack(quaternions_out).view(*original_shape, 4)

        return translations_out, quaternions_out

    def _mlp_forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP parameterization."""
        original_shape = t.shape
        t_input = t.view(-1, 1)

        output = self.net(t_input)

        translation = output[..., :3]
        quaternion = F.normalize(output[..., 3:7], dim=-1)

        return (
            translation.view(*original_shape, 3),
            quaternion.view(*original_shape, 4)
        )

    def _fourier_forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fourier series parameterization."""
        original_shape = t.shape
        t_flat = t.view(-1, 1)

        # Compute Fourier basis
        frequencies = torch.arange(
            1, self.num_frequencies + 1,
            device=t.device, dtype=t.dtype
        )
        phases = 2 * torch.pi * frequencies * t_flat  # (N, num_freq)

        sin_terms = torch.sin(phases)  # (N, num_freq)
        cos_terms = torch.cos(phases)  # (N, num_freq)

        # Stack sin and cos
        basis = torch.cat([sin_terms, cos_terms], dim=-1)  # (N, 2*num_freq)

        # Compute translation
        translation = self.trans_dc + torch.einsum('nf,fd->nd', basis, self.trans_coeffs)

        # Compute quaternion
        quat = self.quat_dc + torch.einsum('nf,fd->nd', basis, self.quat_coeffs)
        quat = F.normalize(quat, dim=-1)

        return (
            translation.view(*original_shape, 3),
            quat.view(*original_shape, 4)
        )


class LearnableKeyframes(nn.Module):
    """
    Learnable keyframe motor representation.

    Separates keyframe values from interpolation logic.
    """

    def __init__(
        self,
        num_keyframes: int = 10,
        learnable_times: bool = False
    ):
        """
        Args:
            num_keyframes: Number of keyframes
            learnable_times: Whether keyframe times are learnable
        """
        super().__init__()

        self.num_keyframes = num_keyframes

        # Keyframe translations
        self.translations = nn.Parameter(torch.zeros(num_keyframes, 3))

        # Keyframe rotations (axis-angle for unconstrained optimization)
        self.axis_angles = nn.Parameter(torch.zeros(num_keyframes, 3))

        # Keyframe times
        times = torch.linspace(0, 1, num_keyframes)
        if learnable_times:
            self.times = nn.Parameter(times)
        else:
            self.register_buffer('times', times)

    def get_quaternions(self) -> torch.Tensor:
        """Convert axis-angles to quaternions."""
        from ..utils.quaternion import quaternion_from_axis_angle

        angles = self.axis_angles.norm(dim=-1, keepdim=True)
        axes = self.axis_angles / (angles + 1e-8)

        return quaternion_from_axis_angle(axes, angles.squeeze(-1))

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get motor at time t.

        Args:
            t: Time in [0, 1]

        Returns:
            (translation, quaternion)
        """
        from .interpolation import motor_slerp

        quaternions = self.get_quaternions()

        # Find segment
        idx = torch.searchsorted(self.times, t.clamp(0, 1))
        idx = idx.clamp(1, self.num_keyframes - 1)

        idx_prev = idx - 1
        t_prev = self.times[idx_prev]
        t_next = self.times[idx]

        local_t = (t - t_prev) / (t_next - t_prev + 1e-8)

        return motor_slerp(
            self.translations[idx_prev], quaternions[idx_prev],
            self.translations[idx], quaternions[idx],
            local_t
        )


class NeuralODEMotor(nn.Module):
    """
    Motor dynamics via Neural ODE.

    Learns the velocity field and integrates to get motor at time t.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Args:
            hidden_dim: Hidden dimension of velocity network
            num_layers: Number of hidden layers
        """
        super().__init__()

        # Velocity network: (t, state) -> d(state)/dt
        layers = [nn.Linear(1 + 7, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 7))

        self.velocity_net = nn.Sequential(*layers)

        # Initialize to zero velocity
        self.velocity_net[-1].weight.data.zero_()
        self.velocity_net[-1].bias.data.zero_()

        # Initial state
        self.initial_translation = nn.Parameter(torch.zeros(3))
        self.initial_quaternion = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def velocity(
        self,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity at (t, state).

        Args:
            t: Time scalar
            state: Current state (7,) [translation(3), quaternion(4)]

        Returns:
            Velocity (7,)
        """
        t_input = t.view(1) if t.dim() == 0 else t
        input_vec = torch.cat([t_input, state], dim=-1)
        return self.velocity_net(input_vec)

    def forward(
        self,
        t: torch.Tensor,
        num_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate to get motor at time t.

        Uses simple Euler integration.

        Args:
            t: Target time in [0, 1]
            num_steps: Integration steps

        Returns:
            (translation, quaternion)
        """
        dt = t / num_steps

        # Initial state
        state = torch.cat([
            self.initial_translation,
            F.normalize(self.initial_quaternion, dim=-1)
        ])

        # Euler integration
        current_t = torch.tensor(0.0, device=t.device)
        for _ in range(num_steps):
            vel = self.velocity(current_t, state)
            state = state + dt * vel
            # Normalize quaternion
            state = torch.cat([
                state[:3],
                F.normalize(state[3:], dim=-1)
            ])
            current_t = current_t + dt

        return state[:3], state[3:]


class ControlledMotor(nn.Module):
    """
    Motor controlled by external signals.

    Maps control signal c(t) to motor parameters.
    """

    def __init__(
        self,
        control_dim: int = 6,
        hidden_dim: int = 64
    ):
        """
        Args:
            control_dim: Dimension of control signal
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.control_dim = control_dim

        # Control to motor mapping
        self.control_net = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)
        )

        # Initialize to identity
        self.control_net[-1].weight.data.zero_()
        self.control_net[-1].bias.data = torch.tensor([0, 0, 0, 1, 0, 0, 0])

    def forward(
        self,
        control: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map control signal to motor.

        Args:
            control: Control signal (..., control_dim)

        Returns:
            (translation, quaternion)
        """
        output = self.control_net(control)

        translation = output[..., :3]
        quaternion = F.normalize(output[..., 3:7], dim=-1)

        return translation, quaternion


class CompositMotor(nn.Module):
    """
    Composition of multiple temporal motors.

    Useful for representing complex motions as product of simpler ones.
    """

    def __init__(self, motors: List[TemporalMotor]):
        """
        Args:
            motors: List of temporal motors to compose
        """
        super().__init__()
        self.motors = nn.ModuleList(motors)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compose all motors at time t.

        M_total = M_1 * M_2 * ... * M_n

        Args:
            t: Time in [0, 1]

        Returns:
            (translation, quaternion)
        """
        from ..utils.quaternion import quaternion_multiply, quaternion_to_matrix

        # Start with identity
        total_translation = torch.zeros(3, device=t.device)
        total_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=t.device)

        for motor in self.motors:
            trans, quat = motor(t)

            # Compose: M_total = M_total * M_i
            # First rotate translation by current rotation
            R = quaternion_to_matrix(total_quaternion)
            total_translation = total_translation + R @ trans

            # Compose rotations
            total_quaternion = quaternion_multiply(total_quaternion, quat)

        return total_translation, total_quaternion


class PeriodicMotor(nn.Module):
    """
    Periodic motor for cyclic motions (walking, spinning, etc.).
    """

    def __init__(
        self,
        period: float = 1.0,
        num_harmonics: int = 4
    ):
        """
        Args:
            period: Motion period
            num_harmonics: Number of Fourier harmonics
        """
        super().__init__()

        self.period = period
        self.num_harmonics = num_harmonics

        # Fourier coefficients
        # [sin_1, cos_1, sin_2, cos_2, ...] for each component
        self.trans_coeffs = nn.Parameter(torch.zeros(num_harmonics * 2, 3))
        self.rot_coeffs = nn.Parameter(torch.zeros(num_harmonics * 2, 3))  # axis-angle

        # DC offset
        self.trans_offset = nn.Parameter(torch.zeros(3))

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get motor at time t.

        Args:
            t: Time (any value, will be wrapped to period)

        Returns:
            (translation, quaternion)
        """
        from ..utils.quaternion import quaternion_from_axis_angle

        # Wrap time to period
        phase = 2 * torch.pi * t / self.period

        # Compute harmonics
        harmonics = []
        for k in range(1, self.num_harmonics + 1):
            harmonics.append(torch.sin(k * phase))
            harmonics.append(torch.cos(k * phase))
        harmonics = torch.stack(harmonics)  # (2*num_harmonics,)

        # Compute translation
        translation = self.trans_offset + torch.einsum('k,kd->d', harmonics, self.trans_coeffs)

        # Compute rotation
        axis_angle = torch.einsum('k,kd->d', harmonics, self.rot_coeffs)
        angle = axis_angle.norm()
        axis = axis_angle / (angle + 1e-8)
        quaternion = quaternion_from_axis_angle(axis.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

        return translation, quaternion
