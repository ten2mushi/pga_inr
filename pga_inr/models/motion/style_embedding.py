"""
Style embedding for conditional motion generation.

Provides learnable style codes that can be optimized during training
or interpolated for style mixing.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEmbedding(nn.Module):
    """
    Learnable style embeddings.

    Each style (e.g., walking, running, dancing) gets a unique
    learnable embedding vector.
    """

    def __init__(
        self,
        num_styles: int,
        embedding_dim: int = 256,
        init_std: float = 0.02
    ):
        """
        Args:
            num_styles: Number of distinct styles
            embedding_dim: Dimension of each embedding
            init_std: Standard deviation for initialization
        """
        super().__init__()

        self.num_styles = num_styles
        self.embedding_dim = embedding_dim

        # Learnable embeddings
        self.embeddings = nn.Embedding(num_styles, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=init_std)

    def forward(self, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Get style embeddings.

        Args:
            style_idx: Style indices of shape (batch,)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.embeddings(style_idx)

    def interpolate(
        self,
        idx1: int,
        idx2: int,
        t: float
    ) -> torch.Tensor:
        """
        Interpolate between two styles.

        Args:
            idx1: First style index
            idx2: Second style index
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated embedding of shape (embedding_dim,)
        """
        e1 = self.embeddings.weight[idx1]
        e2 = self.embeddings.weight[idx2]
        return (1 - t) * e1 + t * e2

    def interpolate_batch(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch interpolation between styles.

        Args:
            idx1: First style indices of shape (batch,)
            idx2: Second style indices of shape (batch,)
            t: Interpolation parameters of shape (batch,) or scalar

        Returns:
            Interpolated embeddings of shape (batch, embedding_dim)
        """
        e1 = self.embeddings(idx1)  # (batch, embedding_dim)
        e2 = self.embeddings(idx2)  # (batch, embedding_dim)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=e1.device, dtype=e1.dtype)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        t = t.unsqueeze(-1)  # (batch, 1)

        return (1 - t) * e1 + t * e2

    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all style embeddings.

        Returns:
            All embeddings of shape (num_styles, embedding_dim)
        """
        return self.embeddings.weight

    @torch.no_grad()
    def normalize_embeddings(self):
        """Normalize all embeddings to unit length."""
        self.embeddings.weight.data = F.normalize(
            self.embeddings.weight.data, p=2, dim=-1
        )


class LearnableStyleBank(nn.Module):
    """
    Style bank with per-sample optimization support.

    Similar to auto-decoder pattern where each sample can have
    its own optimizable style code.
    """

    def __init__(
        self,
        num_styles: int,
        embedding_dim: int = 256,
        init_std: float = 0.02,
        regularization: str = 'l2'
    ):
        """
        Args:
            num_styles: Number of distinct styles
            embedding_dim: Dimension of each embedding
            init_std: Standard deviation for initialization
            regularization: Type of regularization ('l2', 'none')
        """
        super().__init__()

        self.num_styles = num_styles
        self.embedding_dim = embedding_dim
        self.regularization = regularization

        # Learnable style codes
        self.codes = nn.Parameter(
            torch.randn(num_styles, embedding_dim) * init_std
        )

    def forward(self, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Get style codes.

        Args:
            style_idx: Style indices of shape (batch,)

        Returns:
            Style codes of shape (batch, embedding_dim)
        """
        return self.codes[style_idx]

    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss on style codes.

        Returns:
            Regularization loss scalar
        """
        if self.regularization == 'l2':
            return (self.codes ** 2).mean()
        elif self.regularization == 'none':
            return torch.tensor(0.0, device=self.codes.device)
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")

    def get_optimizer_params(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0
    ) -> List[dict]:
        """
        Get optimizer parameter groups for style codes.

        Useful for separate learning rates for codes vs model params.

        Args:
            lr: Learning rate for style codes
            weight_decay: Weight decay

        Returns:
            List of parameter group dictionaries
        """
        return [{
            'params': [self.codes],
            'lr': lr,
            'weight_decay': weight_decay,
        }]

    @torch.no_grad()
    def initialize_from_data(
        self,
        style_features: torch.Tensor
    ):
        """
        Initialize style codes from extracted features.

        Args:
            style_features: Features of shape (num_styles, embedding_dim)
        """
        assert style_features.shape[0] == self.num_styles
        assert style_features.shape[1] == self.embedding_dim
        self.codes.data = style_features.clone()


class ConditionalStyleEmbedding(nn.Module):
    """
    Style embedding with classifier-free guidance support.

    Randomly drops style conditioning during training to enable
    guidance at inference time.
    """

    def __init__(
        self,
        num_styles: int,
        embedding_dim: int = 256,
        init_std: float = 0.02,
        drop_prob: float = 0.1
    ):
        """
        Args:
            num_styles: Number of distinct styles
            embedding_dim: Dimension of each embedding
            init_std: Standard deviation for initialization
            drop_prob: Probability of dropping style during training
        """
        super().__init__()

        self.num_styles = num_styles
        self.embedding_dim = embedding_dim
        self.drop_prob = drop_prob

        # Learnable embeddings (includes null embedding at index 0)
        self.embeddings = nn.Embedding(num_styles + 1, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=init_std)

        # Null embedding (index 0) is zero
        self.embeddings.weight.data[0] = 0.0

    def forward(
        self,
        style_idx: torch.Tensor,
        drop_style: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Get style embeddings with optional dropout.

        Args:
            style_idx: Style indices of shape (batch,) starting from 0
            drop_style: Override dropout behavior (None = use training mode)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Shift indices to account for null embedding at index 0
        style_idx = style_idx + 1

        if drop_style is None:
            drop_style = self.training

        if drop_style and self.training:
            # Randomly replace some indices with null embedding (index 0)
            mask = torch.rand(style_idx.shape, device=style_idx.device) < self.drop_prob
            style_idx = torch.where(mask, torch.zeros_like(style_idx), style_idx)

        return self.embeddings(style_idx)

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get null (unconditional) embedding for classifier-free guidance.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Null embeddings of shape (batch, embedding_dim)
        """
        null_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.embeddings(null_idx)

    def classifier_free_guidance(
        self,
        style_idx: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance scaling.

        Note: This returns the scaled embedding; actual guidance is applied
        to model outputs, not embeddings. This method is for reference.

        Args:
            style_idx: Style indices
            guidance_scale: Guidance scale (1.0 = no guidance)

        Returns:
            Style embeddings
        """
        cond_emb = self.embeddings(style_idx + 1)  # Conditional embedding
        null_emb = self.get_null_embedding(style_idx.shape[0], style_idx.device)

        # Guidance formula applied to outputs: guided = null + scale * (cond - null)
        # Here we just return conditional embedding; guidance applied in diffusion
        return cond_emb
