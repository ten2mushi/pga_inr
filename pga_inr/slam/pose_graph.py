"""
Pose graph for loop closure and global optimization.

Maintains a graph of camera poses connected by relative transform constraints.
Supports loop closure detection and pose graph optimization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from ..utils.quaternion import quaternion_to_matrix, matrix_to_quaternion, normalize_quaternion


@dataclass
class PoseGraphEdge:
    """An edge in the pose graph representing a relative transform constraint."""
    from_id: int
    to_id: int
    relative_translation: torch.Tensor  # (3,)
    relative_quaternion: torch.Tensor   # (4,)
    information: Optional[torch.Tensor] = None  # (6, 6) information matrix
    is_loop_closure: bool = False


class PoseGraph:
    """
    Graph of camera poses connected by relative transform constraints.

    Supports:
    - Adding nodes (poses) and edges (constraints)
    - Loop closure constraint insertion
    - Pose graph optimization using gradient descent

    Uses SE(3) representation with quaternion rotation.
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Args:
            device: Device for computation
        """
        self.device = device

        # Nodes: frame_id -> (translation, quaternion)
        self.nodes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Edges: list of PoseGraphEdge
        self.edges: List[PoseGraphEdge] = []

        # For optimization
        self._optimized = False

    def add_node(
        self,
        node_id: int,
        translation: torch.Tensor,
        quaternion: torch.Tensor
    ):
        """
        Add a pose node to the graph.

        Args:
            node_id: Unique node identifier (typically frame_id)
            translation: (3,) translation
            quaternion: (4,) rotation as [w, x, y, z]
        """
        self.nodes[node_id] = (
            translation.to(self.device),
            quaternion.to(self.device)
        )
        self._optimized = False

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_translation: torch.Tensor,
        relative_quaternion: torch.Tensor,
        information: Optional[torch.Tensor] = None,
        is_loop_closure: bool = False
    ):
        """
        Add a relative transform constraint between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            relative_translation: Relative translation from source to target
            relative_quaternion: Relative rotation from source to target
            information: Optional 6x6 information matrix (inverse covariance)
            is_loop_closure: Whether this is a loop closure constraint
        """
        edge = PoseGraphEdge(
            from_id=from_id,
            to_id=to_id,
            relative_translation=relative_translation.to(self.device),
            relative_quaternion=relative_quaternion.to(self.device),
            information=information.to(self.device) if information is not None else None,
            is_loop_closure=is_loop_closure
        )
        self.edges.append(edge)
        self._optimized = False

    def add_odometry_edge(
        self,
        from_id: int,
        to_id: int
    ):
        """
        Add an odometry edge computed from current node poses.

        Computes relative transform from current poses.
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Node {from_id} or {to_id} not found")

        t_from, q_from = self.nodes[from_id]
        t_to, q_to = self.nodes[to_id]

        # Compute relative transform: T_rel = T_from^{-1} * T_to
        rel_t, rel_q = self._compute_relative_transform(
            t_from, q_from, t_to, q_to
        )

        self.add_edge(from_id, to_id, rel_t, rel_q, is_loop_closure=False)

    def _compute_relative_transform(
        self,
        t1: torch.Tensor,
        q1: torch.Tensor,
        t2: torch.Tensor,
        q2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative transform T_1^{-1} * T_2.

        Returns:
            (relative_translation, relative_quaternion)
        """
        R1 = quaternion_to_matrix(q1.unsqueeze(0)).squeeze(0)
        R2 = quaternion_to_matrix(q2.unsqueeze(0)).squeeze(0)

        # R_rel = R1^T * R2
        R_rel = R1.T @ R2

        # t_rel = R1^T * (t2 - t1)
        t_rel = R1.T @ (t2 - t1)

        q_rel = matrix_to_quaternion(R_rel.unsqueeze(0)).squeeze(0)

        return t_rel, q_rel

    def optimize(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.001,
        fix_first: bool = True
    ) -> List[float]:
        """
        Optimize pose graph using gradient descent.

        Minimizes sum of squared residuals from relative transform constraints.

        Args:
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            fix_first: Whether to fix the first node (remove gauge freedom)

        Returns:
            Loss history
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return []

        # Convert poses to optimization parameters
        node_ids = sorted(self.nodes.keys())
        n_nodes = len(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Initialize parameters: [translation, quaternion] for each node
        translations = nn.Parameter(torch.stack([
            self.nodes[nid][0] for nid in node_ids
        ]))
        quaternions = nn.Parameter(torch.stack([
            self.nodes[nid][1] for nid in node_ids
        ]))

        optimizer = torch.optim.Adam(
            [translations, quaternions],
            lr=learning_rate
        )

        loss_history = []

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=self.device)

            for edge in self.edges:
                i = node_id_to_idx[edge.from_id]
                j = node_id_to_idx[edge.to_id]

                # Current poses
                t_i = translations[i]
                q_i = normalize_quaternion(quaternions[i].unsqueeze(0)).squeeze(0)
                t_j = translations[j]
                q_j = normalize_quaternion(quaternions[j].unsqueeze(0)).squeeze(0)

                # Compute current relative transform
                curr_rel_t, curr_rel_q = self._compute_relative_transform(
                    t_i, q_i, t_j, q_j
                )

                # Compute residual vs measured
                residual_t = curr_rel_t - edge.relative_translation
                residual_q = self._quaternion_error(curr_rel_q, edge.relative_quaternion)

                # Combine into 6D residual
                residual = torch.cat([residual_t, residual_q[:3]])  # Drop w component

                # Compute loss
                if edge.information is not None:
                    loss = residual @ edge.information @ residual
                else:
                    loss = (residual ** 2).sum()

                # Weight loop closure edges higher
                if edge.is_loop_closure:
                    loss = loss * 10.0

                total_loss = total_loss + loss

            total_loss.backward()

            # Fix first node
            if fix_first:
                translations.grad[0].zero_()
                quaternions.grad[0].zero_()

            optimizer.step()

            # Normalize quaternions
            with torch.no_grad():
                quaternions.data = normalize_quaternion(quaternions.data)

            loss_history.append(total_loss.item())

        # Update nodes with optimized poses
        for i, nid in enumerate(node_ids):
            self.nodes[nid] = (
                translations[i].detach(),
                quaternions[i].detach()
            )

        self._optimized = True
        return loss_history

    def _quaternion_error(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quaternion error q1 * q2^{-1}.

        For small errors, [x, y, z] approximates the rotation vector.
        """
        # q2_inv = [w, -x, -y, -z]
        q2_inv = q2.clone()
        q2_inv[1:] = -q2_inv[1:]

        # q_error = q1 * q2_inv
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2_inv

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z])

    def get_trajectory(self) -> List[Tuple[int, torch.Tensor]]:
        """
        Get trajectory as list of (frame_id, position).

        Returns:
            List of (node_id, translation) tuples
        """
        trajectory = []
        for node_id in sorted(self.nodes.keys()):
            t, _ = self.nodes[node_id]
            trajectory.append((node_id, t))
        return trajectory

    def get_poses(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get all poses as dictionary."""
        return self.nodes.copy()

    def detect_loop_candidates(
        self,
        query_id: int,
        distance_threshold: float = 2.0,
        min_interval: int = 20
    ) -> List[int]:
        """
        Find potential loop closure candidates for a query node.

        Args:
            query_id: Query node ID
            distance_threshold: Maximum distance for candidates (meters)
            min_interval: Minimum node ID difference to avoid recent neighbors

        Returns:
            List of candidate node IDs
        """
        if query_id not in self.nodes:
            return []

        query_t, _ = self.nodes[query_id]
        candidates = []

        for node_id in self.nodes:
            # Skip recent neighbors
            if abs(node_id - query_id) < min_interval:
                continue

            t, _ = self.nodes[node_id]
            distance = (query_t - t).norm().item()

            if distance < distance_threshold:
                candidates.append(node_id)

        return candidates

    def num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)

    def num_edges(self) -> int:
        """Get number of edges."""
        return len(self.edges)

    def num_loop_closures(self) -> int:
        """Get number of loop closure edges."""
        return sum(1 for e in self.edges if e.is_loop_closure)
