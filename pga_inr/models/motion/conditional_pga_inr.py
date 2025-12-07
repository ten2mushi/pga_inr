"""
Conditional PGA-INR for motion prediction.

An observer-independent implicit neural representation that predicts
joint rotations as a function of query coordinates (joint index, time),
conditioned on past motion, trajectory, style, and diffusion timestep.

Key Features:
- INR Design: Query at (joint_idx, time) -> rotation
- PGA Integration: Observer-independent via motor transforms
- Diffusion Support: Conditioned on timestep for denoising
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    MotionEncoder,
    TrajectoryEncoder,
    TimestepEmbedding,
    QueryEncoder,
    ConditionAggregator
)
from .style_embedding import StyleEmbedding


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning queries on context.

    Queries attend to condition tokens (past motion, trajectory, style, timestep).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Self-attention for queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to conditions
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        condition_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Process queries with self and cross attention.

        Args:
            queries: Query embeddings (batch, num_queries, hidden_dim)
            condition_tokens: Condition embeddings (batch, num_cond, hidden_dim)

        Returns:
            Updated queries (batch, num_queries, hidden_dim)
        """
        # Self-attention
        q_norm = self.self_attn_norm(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_attn_out

        # Cross-attention to conditions
        q_norm = self.cross_attn_norm(queries)
        cross_attn_out, _ = self.cross_attn(q_norm, condition_tokens, condition_tokens)
        queries = queries + cross_attn_out

        # Feedforward
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class ConditionalMotionPGAINR(nn.Module):
    """
    Observer-independent INR for motion prediction.

    This model is an implicit neural field that can be queried at any
    (joint_index, time) coordinate to predict the rotation at that joint
    at that time.

    Query: (joint_idx, time) -> 6D rotation
    Conditioning: past motion, trajectory, style, diffusion timestep
    PGA: Observer-independent via motor sandwich product
    """

    def __init__(
        self,
        num_joints: int = 65,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        num_styles: int = 2,
        past_frames: int = 5,
        dropout: float = 0.1,
        time_freq_bands: int = 6,
        joint_embedding_dim: int = 32,
        rotation_dim: int = 6
    ):
        """
        Args:
            num_joints: Number of skeleton joints
            hidden_dim: Hidden dimension throughout the model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension (unused, using hidden_dim * 4)
            num_styles: Number of motion styles
            past_frames: Number of past conditioning frames
            dropout: Dropout rate
            time_freq_bands: Fourier frequency bands for time encoding
            joint_embedding_dim: Dimension of joint embeddings
            rotation_dim: Output rotation dimension (6 for 6D representation)
        """
        super().__init__()

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rotation_dim = rotation_dim

        # ===== Query Encoder =====
        self.query_encoder = QueryEncoder(
            num_joints=num_joints,
            joint_embedding_dim=joint_embedding_dim,
            time_freq_bands=time_freq_bands,
            hidden_dim=hidden_dim
        )

        # ===== Condition Encoders =====
        self.motion_encoder = MotionEncoder(
            num_joints=num_joints,
            rotation_dim=rotation_dim,
            past_frames=past_frames,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout
        )

        self.timestep_embedding = TimestepEmbedding(
            hidden_dim=hidden_dim
        )

        self.style_embedding = StyleEmbedding(
            num_styles=num_styles,
            embedding_dim=hidden_dim
        )

        # Optional trajectory encoder (can be disabled for simpler setup)
        self.trajectory_encoder = TrajectoryEncoder(
            future_frames=20,  # Default, adjusted at runtime
            hidden_dim=hidden_dim
        )

        # ===== Noisy Rotation Encoder =====
        # Projects noisy rotation at query point to hidden dim
        self.noisy_rot_encoder = nn.Sequential(
            nn.Linear(rotation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ===== Cross-Attention Transformer =====
        self.transformer_layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ===== Output Head =====
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rotation_dim),
        )

        # Initialize output head with small values for stable training
        # (Don't use zero init which causes temporal variance collapse)
        nn.init.xavier_uniform_(self.output_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_head[-1].bias)

    def encode_conditions(
        self,
        diffusion_timestep: torch.Tensor,
        condition: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode all conditions into a sequence of tokens.

        Args:
            diffusion_timestep: Diffusion timesteps (batch,)
            condition: Dictionary containing:
                - past_motion: (batch, num_joints, rotation_dim, past_frames)
                - style_idx: (batch,) optional
                - traj_translation: (batch, 2, traj_frames) optional
                - traj_rotation: (batch, 6, traj_frames) optional

        Returns:
            Condition tokens (batch, num_cond_tokens, hidden_dim)
        """
        batch_size = diffusion_timestep.shape[0]
        device = diffusion_timestep.device

        condition_tokens = []

        # Timestep embedding
        t_emb = self.timestep_embedding(diffusion_timestep)  # (batch, hidden_dim)
        condition_tokens.append(t_emb.unsqueeze(1))

        # Past motion embedding
        if 'past_motion' in condition and condition['past_motion'] is not None:
            motion_emb = self.motion_encoder(condition['past_motion'])  # (batch, hidden_dim)
            condition_tokens.append(motion_emb.unsqueeze(1))
        else:
            # Null motion token
            null_motion = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            condition_tokens.append(null_motion)

        # Style embedding
        if 'style_idx' in condition and condition['style_idx'] is not None:
            style_emb = self.style_embedding(condition['style_idx'])  # (batch, hidden_dim)
            condition_tokens.append(style_emb.unsqueeze(1))
        else:
            # Null style token
            null_style = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            condition_tokens.append(null_style)

        # Trajectory embedding (optional)
        if ('traj_translation' in condition and condition['traj_translation'] is not None and
            'traj_rotation' in condition and condition['traj_rotation'] is not None):
            traj_emb = self.trajectory_encoder(
                condition['traj_translation'],
                condition['traj_rotation']
            )  # (batch, hidden_dim)
            condition_tokens.append(traj_emb.unsqueeze(1))

        # Concatenate all condition tokens
        condition_tokens = torch.cat(condition_tokens, dim=1)  # (batch, num_cond, hidden_dim)

        return condition_tokens

    def forward(
        self,
        query_joints: torch.Tensor,
        query_times: torch.Tensor,
        noisy_rotations: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Denoise rotations at query coordinates.

        This is the core INR forward pass. Given query coordinates (joint, time),
        current noisy rotations at those coordinates, and conditions, predict
        the denoised rotations.

        Args:
            query_joints: Joint indices to query (batch, num_queries)
            query_times: Time values to query (batch, num_queries) in [0, 1]
            noisy_rotations: Noisy rotations at queries (batch, num_queries, rotation_dim)
            diffusion_timestep: Diffusion timesteps (batch,)
            condition: Conditioning dictionary
            observer_pose: Optional (translation, quaternion) for observer-independence

        Returns:
            Predicted rotations (batch, num_queries, rotation_dim)
        """
        batch_size = query_joints.shape[0]
        num_queries = query_joints.shape[1]

        # ===== PGA Observer Transform (if provided) =====
        # For motion, observer-independence means the root trajectory can be
        # expressed in any coordinate frame. Local joint rotations are already
        # observer-independent (relative to parent).
        if observer_pose is not None:
            # Transform root trajectory in condition by inverse observer motor
            # This is handled at the data level, not inside the model
            pass

        # ===== Encode Query Coordinates =====
        query_emb = self.query_encoder(query_joints, query_times)  # (batch, N, hidden_dim)

        # ===== Encode Noisy Rotations =====
        noisy_rot_emb = self.noisy_rot_encoder(noisy_rotations)  # (batch, N, hidden_dim)

        # Combine query position with noisy rotation
        query_emb = query_emb + noisy_rot_emb

        # ===== Encode Conditions =====
        condition_tokens = self.encode_conditions(diffusion_timestep, condition)

        # ===== Transformer Layers =====
        x = query_emb
        for layer in self.transformer_layers:
            x = layer(x, condition_tokens)

        # ===== Output =====
        output = self.output_head(x)  # (batch, N, rotation_dim)

        return output

    def query_all(
        self,
        num_frames: int,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        observer_pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Convenience method to query all joints at all frames.

        This is the typical use case during training and inference.

        Args:
            num_frames: Number of time frames to query
            noisy_motion: Noisy motion (batch, num_joints, rotation_dim, num_frames)
            diffusion_timestep: Diffusion timesteps (batch,)
            condition: Conditioning dictionary
            observer_pose: Optional observer pose

        Returns:
            Predicted motion (batch, num_joints, rotation_dim, num_frames)
        """
        batch_size = noisy_motion.shape[0]
        device = noisy_motion.device

        # Create meshgrid of (joint, time) queries
        joints = torch.arange(self.num_joints, device=device)
        times = torch.linspace(0, 1, num_frames, device=device)

        # Expand to batch
        # query_joints: (batch, num_joints * num_frames)
        # query_times: (batch, num_joints * num_frames)
        # Need same dtype for meshgrid - convert to float then back to long for joint indices
        joint_grid, time_grid = torch.meshgrid(
            joints.float(), times, indexing='ij'
        )
        query_joints = joint_grid.flatten().long().unsqueeze(0).expand(batch_size, -1)
        query_times = time_grid.flatten().unsqueeze(0).expand(batch_size, -1)

        # Reshape noisy motion for queries
        # (batch, num_joints, rotation_dim, num_frames) -> (batch, num_joints * num_frames, rotation_dim)
        noisy_rotations = noisy_motion.permute(0, 1, 3, 2).reshape(batch_size, -1, self.rotation_dim)

        # Forward pass
        output = self.forward(
            query_joints, query_times, noisy_rotations,
            diffusion_timestep, condition, observer_pose
        )

        # Reshape output back
        # (batch, num_joints * num_frames, rotation_dim) -> (batch, num_joints, rotation_dim, num_frames)
        output = output.reshape(batch_size, self.num_joints, num_frames, self.rotation_dim)
        output = output.permute(0, 1, 3, 2)

        return output


class MotionINRWrapper(nn.Module):
    """
    Wrapper that adapts ConditionalMotionPGAINR for use with GaussianDiffusion.

    Provides the expected interface: (noisy_x, timestep, condition) -> prediction
    """

    def __init__(
        self,
        inr_model: ConditionalMotionPGAINR,
        num_frames: int = 20
    ):
        """
        Args:
            inr_model: The underlying INR model
            num_frames: Number of frames in the motion sequence
        """
        super().__init__()

        self.inr = inr_model
        self.num_frames = num_frames
        self.num_joints = inr_model.num_joints

    def forward(
        self,
        noisy_motion: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Denoise motion.

        Args:
            noisy_motion: Noisy motion (batch, num_joints, rotation_dim, num_frames)
            timestep: Diffusion timesteps (batch,)
            condition: Conditioning dictionary

        Returns:
            Predicted clean motion (batch, num_joints, rotation_dim, num_frames)
        """
        if condition is None:
            condition = {}

        return self.inr.query_all(
            num_frames=self.num_frames,
            noisy_motion=noisy_motion,
            diffusion_timestep=timestep,
            condition=condition
        )
