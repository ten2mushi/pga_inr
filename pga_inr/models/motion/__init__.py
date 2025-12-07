"""
Motion models for PGA-INR.

Provides conditional implicit neural representations for motion prediction,
including encoders, style embeddings, and the main INR model.
"""

from .encoders import (
    FourierEncoder,
    TimestepEmbedding,
    JointEmbedding,
    MotionEncoder,
    TrajectoryEncoder,
    QueryEncoder,
    ConditionAggregator,
)
from .style_embedding import (
    StyleEmbedding,
    LearnableStyleBank,
    ConditionalStyleEmbedding,
)
from .conditional_pga_inr import (
    ConditionalMotionPGAINR,
    MotionINRWrapper,
    CrossAttentionBlock,
)

__all__ = [
    # Encoders
    "FourierEncoder",
    "TimestepEmbedding",
    "JointEmbedding",
    "MotionEncoder",
    "TrajectoryEncoder",
    "QueryEncoder",
    "ConditionAggregator",
    # Style
    "StyleEmbedding",
    "LearnableStyleBank",
    "ConditionalStyleEmbedding",
    # Models
    "ConditionalMotionPGAINR",
    "MotionINRWrapper",
    "CrossAttentionBlock",
]
