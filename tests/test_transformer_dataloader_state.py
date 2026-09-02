import importlib.util
import unittest
from pathlib import Path

import torch


DATALOADER_FILE = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "Transformer"
    / "dataloader"
    / "__init__.py"
)
spec = importlib.util.spec_from_file_location(
    "_decoart_transformer_dataloader_under_test", DATALOADER_FILE
)
dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataloader)


class FactorizedStateDataTest(unittest.TestCase):
    def test_real_node_keeps_structure_and_geometry_latent(self):
        structure = list(range(16))
        geometry = [float(index) / 10 for index in range(8)]
        node = {
            "token": structure + [-1.0] * 8,
            "packed_info": {"latent": geometry},
        }

        state = dataloader.extract_factorized_state(node, geometry_dim=8)

        torch.testing.assert_close(state[:16], torch.tensor(structure).float())
        torch.testing.assert_close(state[16:], torch.tensor(geometry).float())

    def test_special_node_uses_zero_geometry(self):
        node = {
            "token": [2.0] * 24,
            "packed_info": {"text_hat": [0.0] * 4},
        }

        state = dataloader.extract_factorized_state(node, geometry_dim=8)

        torch.testing.assert_close(state[:16], torch.full((16,), 2.0))
        torch.testing.assert_close(state[16:], torch.zeros(8))


if __name__ == "__main__":
    unittest.main()
