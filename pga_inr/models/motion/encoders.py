"""
Encoders for motion conditioning signals.

Provides encoders for:
- Past motion sequences
- Future trajectory
- Diffusion timesteps
- Joint and time coordinates (for INR queries)
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import positional encoder from main encoders module for consistency
# PositionalEncoder is the classic NeRF-style Fourier encoding
from ..encoders import PositionalEncoder

# Alias for backwards compatibility and semantic clarity in motion context
FourierEncoder = PositionalEncoder


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.

    Uses sinusoidal embedding followed by MLP projection.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_timesteps: int = 1000
    ):
        """
        Args:
            hidden_dim: Output dimension
            max_timesteps: Maximum timestep value
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Sinusoidal embedding dimension
        half_dim = hidden_dim // 2

        # Precompute embedding frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb_freqs', emb)

        # MLP to process embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Args:
            t: Integer timesteps of shape (batch,)

        Returns:
            Embeddings of shape (batch, hidden_dim)
        """
        t = t.float()

        # Sinusoidal embedding
        emb = t[:, None] * self.emb_freqs[None, :]  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, hidden_dim)

        # MLP projection
        return self.mlp(emb)


class JointEmbedding(nn.Module):
    """
    Learnable embedding for joint indices.

    Each joint gets a unique learnable embedding vector.
    """

    def __init__(
        self,
        num_joints: int = 65,
        embedding_dim: int = 32,
        init_std: float = 0.02
    ):
        """
        Args:
            num_joints: Number of skeleton joints
            embedding_dim: Dimension of each embedding
            init_std: Standard deviation for initialization
        """
        super().__init__()

        self.num_joints = num_joints
        self.embedding_dim = embedding_dim

        # Learnable embeddings
        self.embeddings = nn.Embedding(num_joints, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=init_std)

    def forward(self, joint_indices: torch.Tensor) -> torch.Tensor:
        """
        Get joint embeddings.

        Args:
            joint_indices: Joint indices of shape (batch, num_queries)

        Returns:
            Embeddings of shape (batch, num_queries, embedding_dim)
        """
        return self.embeddings(joint_indices)


class MotionEncoder(nn.Module):
    """
    Encoder for past motion sequences.

    Processes past joint rotations using Conv1D and Transformer.
    """

    def __init__(
        self,
        num_joints: int = 65,
        rotation_dim: int = 6,
        past_frames: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            num_joints: Number of skeleton joints
            rotation_dim: Rotation representation dimension (6 for 6D)
            past_frames: Number of past frames
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_joints = num_joints
        self.rotation_dim = rotation_dim
        self.past_frames = past_frames
        self.hidden_dim = hidden_dim

        # Input dimension: flattened joints and rotations
        input_dim = num_joints * rotation_dim

        # Temporal convolution
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Positional encoding for frames
        self.pos_enc = nn.Parameter(
            torch.randn(1, past_frames, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, past_motion: torch.Tensor) -> torch.Tensor:
        """
        Encode past motion.

        Args:
            past_motion: Past motion of shape (batch, num_joints, rotation_dim, past_frames)

        Returns:
            Encoded motion of shape (batch, hidden_dim)
        """
        batch_size = past_motion.shape[0]

        # Reshape: (batch, num_joints * rotation_dim, past_frames)
        x = past_motion.reshape(batch_size, -1, self.past_frames)

        # Conv layers
        x = self.conv_layers(x)  # (batch, hidden_dim, past_frames)

        # Transpose for transformer: (batch, past_frames, hidden_dim)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.pos_enc

        # Transformer
        x = self.transformer(x)  # (batch, past_frames, hidden_dim)

        # Aggregate (mean pooling)
        x = x.mean(dim=1)  # (batch, hidden_dim)

        # Output projection
        x = self.output_proj(x)

        return x


class TrajectoryEncoder(nn.Module):
    """
    Encoder for future trajectory.

    Processes root position and orientation trajectory.
    """

    def __init__(
        self,
        future_frames: int = 20,
        hidden_dim: int = 256,
        trans_dim: int = 2,  # XZ plane
        rot_dim: int = 6,    # 6D rotation
    ):
        """
        Args:
            future_frames: Number of future frames
            hidden_dim: Hidden dimension
            trans_dim: Translation dimension (2 for XZ plane)
            rot_dim: Rotation dimension (6 for 6D)
        """
        super().__init__()

        self.future_frames = future_frames
        self.hidden_dim = hidden_dim

        # Trajectory is typically subsampled
        traj_frames = future_frames // 2

        # Translation encoder
        self.trans_encoder = nn.Sequential(
            nn.Conv1d(trans_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Rotation encoder
        self.rot_encoder = nn.Sequential(
            nn.Conv1d(rot_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * traj_frames, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        traj_translation: torch.Tensor,
        traj_rotation: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode trajectory.

        Args:
            traj_translation: Root translation of shape (batch, trans_dim, traj_frames)
            traj_rotation: Root rotation of shape (batch, rot_dim, traj_frames)

        Returns:
            Encoded trajectory of shape (batch, hidden_dim)
        """
        batch_size = traj_translation.shape[0]

        # Encode translation and rotation
        trans_feat = self.trans_encoder(traj_translation)  # (batch, hidden_dim//2, T)
        rot_feat = self.rot_encoder(traj_rotation)         # (batch, hidden_dim//2, T)

        # Concatenate features
        combined = torch.cat([trans_feat, rot_feat], dim=1)  # (batch, hidden_dim, T)

        # Flatten and combine
        combined = combined.reshape(batch_size, -1)
        return self.combine(combined)


class QueryEncoder(nn.Module):
    """
    Encoder for INR query coordinates (joint index, time).

    Combines joint embedding with time Fourier encoding.
    """

    def __init__(
        self,
        num_joints: int = 65,
        joint_embedding_dim: int = 32,
        time_freq_bands: int = 6,
        hidden_dim: int = 256
    ):
        """
        Args:
            num_joints: Number of skeleton joints
            joint_embedding_dim: Joint embedding dimension
            time_freq_bands: Number of Fourier frequency bands for time
            hidden_dim: Output hidden dimension
        """
        super().__init__()

        self.num_joints = num_joints

        # Joint embedding
        self.joint_embedding = JointEmbedding(
            num_joints=num_joints,
            embedding_dim=joint_embedding_dim
        )

        # Time encoding
        self.time_encoder = FourierEncoder(
            input_dim=1,
            num_frequencies=time_freq_bands,
            include_input=True
        )

        # Combined dimension
        query_dim = joint_embedding_dim + self.time_encoder.output_dim

        # Project to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        joint_indices: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode query coordinates.

        Args:
            joint_indices: Joint indices of shape (batch, num_queries)
            times: Time values of shape (batch, num_queries) in [0, 1]

        Returns:
            Query encodings of shape (batch, num_queries, hidden_dim)
        """
        # Joint embedding: (batch, num_queries, joint_embedding_dim)
        joint_emb = self.joint_embedding(joint_indices)

        # Time encoding: (batch, num_queries, time_enc_dim)
        time_emb = self.time_encoder(times.unsqueeze(-1))

        # Concatenate
        query = torch.cat([joint_emb, time_emb], dim=-1)

        # Project
        return self.proj(query)


class ConditionAggregator(nn.Module):
    """
    Aggregates multiple condition embeddings into a single representation.

    Combines: past motion, trajectory, style, timestep.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_conditions: int = 4
    ):
        """
        Args:
            hidden_dim: Dimension of each condition embedding
            num_conditions: Number of conditions to aggregate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Projection for concatenated conditions
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_conditions, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        conditions: list
    ) -> torch.Tensor:
        """
        Aggregate condition embeddings.

        Args:
            conditions: List of condition tensors, each (batch, hidden_dim)

        Returns:
            Aggregated condition of shape (batch, hidden_dim)
        """
        # Concatenate all conditions
        combined = torch.cat(conditions, dim=-1)

        # Project
        return self.proj(combined)
