import unittest
import importlib.util
import sys
import types
from pathlib import Path

import torch


LAYERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "Transformer"
    / "transformer"
    / "layers"
)
PACKAGE_NAME = "_decoart_layers_under_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(LAYERS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.factorized_state", LAYERS_DIR / "factorized_state.py"
)
factorized_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factorized_state)
FactorizedStateEmbedding = factorized_state.FactorizedStateEmbedding


class FactorizedStateEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.embedding = FactorizedStateEmbedding(
            structure_dim=16,
            geometry_dim=8,
            hidden_dim=32,
            d_model=16,
            position_dim_single_emb=4,
            dropout=0.0,
            use_tree_position=True,
        ).eval()
        self.structure = torch.randn(2, 4, 16)
        self.geometry = torch.randn(2, 4, 8)
        self.fa = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 2]])

    def test_tree_state_is_independent_of_geometry(self):
        changed_geometry = self.geometry + 3.0

        _, first = self.embedding(
            self.structure, self.geometry, self.fa, return_components=True
        )
        _, second = self.embedding(
            self.structure, changed_geometry, self.fa, return_components=True
        )

        torch.testing.assert_close(first["tree_structure"], second["tree_structure"])

    def test_geometry_changes_fused_state_but_not_tree_state(self):
        changed_geometry = self.geometry.clone()
        changed_geometry[:, :, 0] += 5.0

        first, first_components = self.embedding(
            self.structure, self.geometry, self.fa, return_components=True
        )
        second, second_components = self.embedding(
            self.structure, changed_geometry, self.fa, return_components=True
        )

        torch.testing.assert_close(
            first_components["tree_structure"], second_components["tree_structure"]
        )
        self.assertFalse(torch.allclose(first, second))

    def test_gate_is_elementwise_and_bounded(self):
        fused, components = self.embedding(
            self.structure, self.geometry, self.fa, return_components=True
        )

        self.assertEqual(fused.shape, (2, 4, 16))
        self.assertEqual(components["gate"].shape, fused.shape)
        self.assertTrue(torch.all(components["gate"] >= 0.0))
        self.assertTrue(torch.all(components["gate"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
