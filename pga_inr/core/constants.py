"""
Centralized constants for PGA-INR.

This module defines all default values and numeric constants used throughout
the library. Using these constants ensures consistency and makes it easy to
adjust defaults globally.

Usage:
    from pga_inr.core.constants import DEFAULT_EPS, DEFAULT_OMEGA_0

    # Use in function definitions
    def my_function(eps: float = DEFAULT_EPS):
        ...
"""

# =============================================================================
# Numeric Constants
# =============================================================================

# Small epsilon for division stability (general use)
DEFAULT_EPS: float = 1e-8

# Epsilon for normalization operations (slightly larger for numerical stability)
DEFAULT_EPS_NORM: float = 1e-12

# Maximum gradient norm for clipping during training
DEFAULT_GRADIENT_CLIP: float = 1.0

# Pi constant (for rotation calculations)
PI: float = 3.14159265358979323846


# =============================================================================
# Model Architecture Defaults
# =============================================================================

# SIREN network defaults
DEFAULT_HIDDEN_FEATURES: int = 256
DEFAULT_HIDDEN_LAYERS: int = 4
DEFAULT_OMEGA_0: float = 30.0
DEFAULT_OMEGA_HIDDEN: float = 30.0

# Generative model defaults
DEFAULT_LATENT_DIM: int = 64

# Positional encoding defaults
DEFAULT_NUM_FREQUENCIES: int = 6


# =============================================================================
# SDF and Surface Defaults
# =============================================================================

# Surface detection threshold (distance from zero level set)
DEFAULT_SURFACE_THRESHOLD: float = 0.05

# Foot contact detection threshold (height above ground)
DEFAULT_CONTACT_THRESHOLD: float = 0.05

# Sphere tracing convergence threshold
DEFAULT_SPHERE_TRACE_EPS: float = 1e-4

# Maximum sphere tracing iterations
DEFAULT_SPHERE_TRACE_STEPS: int = 128

# Step scale for sphere tracing (multiplier on SDF value)
DEFAULT_SPHERE_TRACE_SCALE: float = 0.9


# =============================================================================
# Training Defaults
# =============================================================================

# Optimizer defaults
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_WEIGHT_DECAY: float = 0.0

# Sampling defaults
DEFAULT_POINTS_PER_SAMPLE: int = 5000
DEFAULT_SURFACE_SAMPLE_RATIO: float = 0.5


# =============================================================================
# Loss Weight Defaults
# =============================================================================

DEFAULT_LAMBDA_SDF: float = 1.0
DEFAULT_LAMBDA_EIKONAL: float = 0.1
DEFAULT_LAMBDA_NORMAL: float = 1.0
DEFAULT_LAMBDA_LATENT: float = 0.001

# Motion loss weights
DEFAULT_LAMBDA_FK_POS: float = 0.5
DEFAULT_LAMBDA_VELOCITY: float = 0.2
DEFAULT_LAMBDA_FOOT_CONTACT: float = 0.0


# =============================================================================
# Diffusion Defaults
# =============================================================================

DEFAULT_NUM_TIMESTEPS: int = 1000
DEFAULT_BETA_START: float = 1e-4
DEFAULT_BETA_END: float = 0.02


# =============================================================================
# Motion Defaults
# =============================================================================

# Number of joints in standard skeleton (Mixamo rig)
DEFAULT_NUM_JOINTS: int = 65

# Rotation representation dimension (6D continuous)
DEFAULT_ROTATION_DIM: int = 6

# Frame sampling
DEFAULT_PAST_FRAMES: int = 10
DEFAULT_FUTURE_FRAMES: int = 20


# =============================================================================
# Output Dictionary Keys
# =============================================================================

# These constants ensure consistent key naming across all models

# Primary output keys
OUTPUT_SDF: str = "sdf"
OUTPUT_DENSITY: str = "density"
OUTPUT_RGB: str = "rgb"
OUTPUT_NORMAL: str = "normal"
OUTPUT_GRADIENT: str = "gradient"
OUTPUT_LOCAL_COORDS: str = "local_coords"
OUTPUT_FEATURES: str = "features"

# Motion-specific output keys
OUTPUT_ROTATIONS: str = "rotations"
OUTPUT_POSITIONS: str = "positions"
OUTPUT_VELOCITIES: str = "velocities"
