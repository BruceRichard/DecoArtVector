import torch
from torch import nn
from rich import print
from functools import reduce

from .layers.decoder_layer import DecoderLayer
from .layers.post_encoder import PostEncoder, ResnetBlockFC
from .layers.token import MLPUnTokenizer
from .layers.factorized_state import FactorizedStateEmbedding
# from .layers.vq_embedding import VQEmbedding

from utils.mylogging import Log

# main part of Articulation Transformer.

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']

        self.config = config
        self.part_structure = self.config['part_structure']
        self.m_config = self.config['transformer_model_paramerter']
        self.d_model = self.m_config['d_model']
        # self.vq_dim = self.m_config['vq_expand_dim']

        self.diff_config = self.config['diff_config']

        self.to_z_logits_fc = nn.Linear(self.part_structure['condition'], self.diff_config['gsemb_num_embeddings'] * self.diff_config['gsemb_latent_dim'])
        self.to_text_hat_fc = nn.Linear(self.part_structure['condition'], self.diff_config['diffusion_model_config']['text_hat_dim'])

        d_structure_input = sum(
            [v for k, v in self.part_structure.items() if k != 'condition' and k != 'latentcode']
        )

        d_token_condition = sum(
            [v for k, v in self.part_structure.items() if k != 'latentcode']
        )

        d_token_condition_with_bbx_dis = d_token_condition + 3

        self.dim_latent = self.part_structure['latentcode']
        self.dim_condition = self.part_structure['condition']

        use_tree_position = self.m_config.get('tree_position_embedding', True)
        if not use_tree_position:
            # For ablation study.
            Log.critical("Didn't Use PositionGRUEmbedding")
            import time; time.sleep(3)

        self.use_shape_prior = self.m_config.get('shape_prior', True)
        if not self.use_shape_prior:
            Log.critical("Didn't Use use_shape_prior")
            import time; time.sleep(3)


        # self.expand_latent_dim = reduce(lambda x, y: x * y, self.m_config['vq_expand_dim'])

        # self.latentcode_encoder = nn.Sequential(*[
        #     ResnetBlockFC(self.expand_latent_dim, 0.1)
        #     for _ in range(self.m_config['before_vq_net_deepth'])
        # ])

        # self.latentcode_expand_fc = nn.Linear(self.dim_latent,  self.expand_latent_dim)
        # self.vq_embedding   = VQEmbedding(self.m_config['n_embed'], self.m_config['vq_expand_dim'][0], beta=self.m_config['vq_beta'])
        # self.latentcode_to_condition = nn.Linear(self.expand_latent_dim, self.dim_condition)

        self.state_embedding = FactorizedStateEmbedding(
            structure_dim=d_structure_input,
            geometry_dim=self.dim_latent,
            hidden_dim=self.m_config['tokenizer_hidden_dim'],
            d_model=self.d_model,
            position_dim_single_emb=self.m_config['position_embedding_dim_single_emb'],
            dropout=self.m_config['tokenizer_dropout'],
            use_tree_position=use_tree_position,
        )

        self.untokenizer    = MLPUnTokenizer(d_token_condition_with_bbx_dis,
                                             d_hidden=self.m_config['tokenizer_hidden_dim'],
                                             d_model=self.d_model,
                                             drop_out=self.m_config['tokenizer_dropout'])

        self.postencoder    = PostEncoder(dim=self.m_config['encoder_kv_dim'], d_model=self.d_model,
                                          dropout=self.m_config['post_encoder_dropout'],
                                          deepth=self.m_config['post_encoder_deepth'])

        self.end_token_logits = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1)
        )

        self.layers         = nn.ModuleList([
            DecoderLayer(config)
                for _ in range(self.m_config['n_layer'])
        ])


    def generate_mask(self, n_part):

        mask = torch.ones(n_part, n_part, device=self.device, dtype=torch.int16)
        # mask = torch.tril(mask) # no need mask
        return mask

    def forward(self, input, padding_mask, enc_data):
        # ('token'/'dfn'/'dfn_fa') * batch * part_idx * attribute_dim
        enc_data = self.postencoder(enc_data)

        batch, n_part, _ = input['token'].size()

        structure = input['token'][..., :-self.dim_latent]
        geometry = input['token'][..., -self.dim_latent:]
        tokens = self.state_embedding(structure, geometry, input['fa'])

        attn_mask = self.generate_mask(n_part)

        cross_attn_weight_list = []
        for idx, layer in enumerate(self.layers):
            # tokens = layer(tokens, padding_mask, attn_mask, enc_data, None)
            tokens, cross_attn_weight = layer(tokens, padding_mask, attn_mask, enc_data)
            cross_attn_weight_list.append(cross_attn_weight.detach().cpu().numpy())

        # skip padding mask.
        tokens = tokens[padding_mask > 0.5]

        end_token_logits = self.end_token_logits(tokens).squeeze(-1)

        tokens = self.untokenizer(tokens)

        conditions = tokens[:, -self.dim_condition:]
        raw_articulated_info = tokens[:, :-self.dim_condition]

        if self.use_shape_prior:
            text_hat_condition = self.to_text_hat_fc(conditions)
            z_logits_condition = self.to_z_logits_fc(conditions).view(-1, self.diff_config['gsemb_latent_dim'],
                                                                self.diff_config['gsemb_num_embeddings'])
            result_condition = { # Generate latentcode base on 'result_condition'.
                'text_hat': text_hat_condition,
                'z_logits': z_logits_condition
            }
        else:
            result_condition = conditions #  latentcode IS 'result_condition' it self.

        # process length of xyz of bounding box
        _b_mu = raw_articulated_info[:, 0:3]
        _b_logvar = raw_articulated_info[:, 3:6]
        # Sample base on predicted `mean` and `var`.
        # _b_std = torch.exp(0.5 * _b_logvar)
        # eps = torch.randn_like(_b_std)
        _b_length_xyz = _b_mu #  + eps * _b_std
        # Do Not try to sample from code.
        articulated_info = torch.cat((_b_length_xyz, raw_articulated_info[:, 6:]), dim=-1)

        result = {
            'is_end_token_logits': end_token_logits,
            'articulated_info': articulated_info,
            'condition': result_condition,
            'cross_attn_weight_list': cross_attn_weight_list
        }

        return result
