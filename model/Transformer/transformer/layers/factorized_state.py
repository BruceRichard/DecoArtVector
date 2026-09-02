import torch
from torch import nn

from .position import PositionGRUEmbedding
from .token import MLPTokenizer


class FactorizedStateEmbedding(nn.Module):
    """Encode structure/kinematics and geometry separately, then fuse them."""

    def __init__(self, structure_dim, geometry_dim, hidden_dim, d_model,
                 position_dim_single_emb, dropout, use_tree_position=True):
        super().__init__()
        self.structure_encoder = MLPTokenizer(
            d_token=structure_dim,
            d_hidden=hidden_dim,
            d_model=d_model,
            drop_out=dropout,
        )
        self.geometry_encoder = MLPTokenizer(
            d_token=geometry_dim,
            d_hidden=hidden_dim,
            d_model=d_model,
            drop_out=dropout,
        )
        self.position_embedding = (
            PositionGRUEmbedding(
                d_model=d_model,
                dim_single_emb=position_dim_single_emb,
                dropout=dropout,
            )
            if use_tree_position
            else None
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, structure, geometry, fa, return_components=False):
        structure_embedding = self.structure_encoder(structure)
        geometry_embedding = self.geometry_encoder(geometry)

        if self.position_embedding is None:
            tree_embedding = torch.zeros_like(structure_embedding)
        else:
            tree_embedding = self.position_embedding({
                'token': structure_embedding,
                'fa': fa,
            })

        structure_tree = torch.cat(
            (structure_embedding, tree_embedding), dim=-1
        )
        gate = torch.sigmoid(self.gate(structure_tree))
        gated_geometry = gate * geometry_embedding
        fused = self.norm(
            self.fusion(torch.cat((structure_tree, gated_geometry), dim=-1))
        )

        if not return_components:
            return fused

        return fused, {
            'structure': structure_embedding,
            'tree': tree_embedding,
            'structure_tree': structure_tree,
            'geometry': geometry_embedding,
            'gate': gate,
            'gated_geometry': gated_geometry,
        }
