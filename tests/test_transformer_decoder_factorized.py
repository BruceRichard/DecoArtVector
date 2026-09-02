import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_ROOT = REPO_ROOT / "model" / "Transformer"
TRANSFORMER_PACKAGE_ROOT = TRANSFORMER_ROOT / "transformer"

utils_package = types.ModuleType("utils")
utils_package.__path__ = [str(REPO_ROOT / "utils")]
sys.modules["utils"] = utils_package
logging_module = types.ModuleType("utils.mylogging")
logging_module.Log = logging.getLogger("factorized-decoder-test")
sys.modules["utils.mylogging"] = logging_module

transformer_package = types.ModuleType("model.Transformer")
transformer_package.__path__ = [str(TRANSFORMER_ROOT)]
sys.modules["model.Transformer"] = transformer_package

decoder_package = types.ModuleType("model.Transformer.transformer")
decoder_package.__path__ = [str(TRANSFORMER_PACKAGE_ROOT)]
sys.modules["model.Transformer.transformer"] = decoder_package

spec = importlib.util.spec_from_file_location(
    "model.Transformer.transformer.decoder", TRANSFORMER_PACKAGE_ROOT / "decoder.py"
)
decoder_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = decoder_module
spec.loader.exec_module(decoder_module)
TransformerDecoder = decoder_module.TransformerDecoder


class FactorizedTransformerDecoderTest(unittest.TestCase):
    def test_forward_accepts_structure_plus_geometry_state(self):
        config = {
            "device": "cpu",
            "part_structure": {
                "bounding_box": 6,
                "joint_data_origin": 3,
                "joint_data_direction": 3,
                "limit": 4,
                "condition": 4,
                "latentcode": 8,
            },
            "transformer_model_paramerter": {
                "d_model": 16,
                "tree_position_embedding": True,
                "shape_prior": True,
                "position_embedding_dim_single_emb": 4,
                "position_embedding_dropout": 0.0,
                "tokenizer_hidden_dim": 32,
                "tokenizer_dropout": 0.0,
                "encoder_kv_dim": 4,
                "post_encoder_dropout": 0.0,
                "post_encoder_deepth": 0,
                "n_layer": 0,
            },
            "diff_config": {
                "gsemb_num_embeddings": 4,
                "gsemb_latent_dim": 2,
                "diffusion_model_config": {"text_hat_dim": 3},
            },
        }
        decoder = TransformerDecoder(config).eval()
        state = torch.randn(2, 3, 24)
        fa = torch.tensor([[0, 0, 1], [0, 0, 0]])
        padding_mask = torch.ones(2, 3)
        encoded_condition = torch.randn(2, 5, 4)

        result = decoder(
            {"token": state, "fa": fa}, padding_mask, encoded_condition
        )

        self.assertEqual(result["is_end_token_logits"].shape, (6,))
        self.assertEqual(result["articulated_info"].shape, (6, 16))
        self.assertEqual(result["condition"]["text_hat"].shape, (6, 3))
        self.assertEqual(result["condition"]["z_logits"].shape, (6, 2, 4))


if __name__ == "__main__":
    unittest.main()
