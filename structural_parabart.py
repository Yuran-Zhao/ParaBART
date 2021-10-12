import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bart import (PretrainedBartModel, LayerNorm,
                                        EncoderLayer, DecoderLayer,
                                        LearnedPositionalEmbedding,
                                        _prepare_bart_decoder_inputs,
                                        _make_linear_from_emb)

import dgl.nn as dglnn
import dgl
import pdb


class StructuralParaBart(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model,
                                   config.pad_token_id)

        self.encoder = ParaBartEncoder(config, self.shared)
        self.decoder = ParaBartDecoder(config, self.shared)

        self.linear = nn.Linear(config.d_model, config.vocab_size)

        # self.adversary = Discriminator(config)
        # self.adversary = StructuralDiscriminator(config, rank=64)
        self.adversary = StructuralDiscriminator(config, config.rank)

        self.init_weights()

    def forward(
        self,
        sent1_input_ids,
        synt2_input_ids,
        graph2,
        decoder_input_ids,
        attention_mask=None,
        decoder_padding_mask=None,
        encoder_outputs=None,
        return_encoder_outputs=False,
    ):
        if attention_mask is None:
            # attention_mask = input_ids == self.config.pad_token_id
            sent1_input_attention_mask = sent1_input_ids == self.config.pad_token_id
            sent2_input_attention_mask = synt2_input_ids == self.config.pad_token_id

        if encoder_outputs is None:
            # encoder_outputs = (encoder_outputs, sent_embeds, x)
            # pdb.set_trace()
            encoder_outputs = self.encoder(sent1_input_ids, synt2_input_ids,
                                           sent1_input_attention_mask,
                                           sent2_input_attention_mask, graph2)

        if return_encoder_outputs:
            return encoder_outputs

        assert encoder_outputs is not None
        assert decoder_input_ids is not None

        decoder_input_ids = decoder_input_ids[:, :-1]

        _, decoder_padding_mask, decoder_causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
            causal_mask_dtype=self.shared.weight.dtype,
        )

        # attention_mask2 = torch.cat(
        #     (torch.zeros(input_ids.shape[0], 1).bool().cuda(),
        #      attention_mask[:, self.config.max_sent_len + 2:]),
        #     dim=1)
        # NOTE: the old `attention_mask[:, self.config.max_sent_len + 2:] is sent2_input_attention_mask`
        attention_mask2 = torch.cat(
            (torch.zeros(sent1_input_ids.shape[0],
                         1).bool().cuda(), sent2_input_attention_mask),
            dim=1)

        # decoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            torch.cat((encoder_outputs[1],
                       encoder_outputs[0][:, self.config.max_sent_len + 2:]),
                      dim=1),
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=decoder_causal_mask,
            encoder_attention_mask=attention_mask2,
        )[0]

        batch_size = decoder_outputs.shape[0]
        outputs = self.linear(decoder_outputs.contiguous().view(
            -1, self.config.d_model))
        outputs = outputs.view(batch_size, -1, self.config.vocab_size)

        # discriminator
        for p in self.adversary.parameters():
            p.required_grad = False
        # adv_outputs = self.adversary(encoder_outputs[1])
        squared_distance, depth, bow = self.adversary(encoder_outputs[2],
                                                      encoder_outputs[1])

        return outputs, squared_distance, depth, bow

    def prepare_inputs_for_generation(self, decoder_input_ids, past,
                                      attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs = past[0]
        return {
            "input_ids":
            None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs":
            encoder_outputs,
            "decoder_input_ids":
            torch.cat(
                (decoder_input_ids,
                 torch.zeros(
                     (decoder_input_ids.shape[0], 1), dtype=torch.long).cuda()),
                1),
            "attention_mask":
            attention_mask,
        }

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)

    def get_input_embeddings(self):
        return self.shared

    @staticmethod
    def _reorder_cache(past, beam_idx):
        enc_out = past[0][0]

        new_enc_out = enc_out.index_select(0, beam_idx)

        past = ((new_enc_out, ), )
        return past

    def forward_adv(self,
                    input_token_ids,
                    attention_mask=None,
                    decoder_padding_mask=None):
        for p in self.adversary.parameters():
            p.required_grad = True
        # sent_embeds = self.encoder.embed(
        #     input_token_ids, attention_mask=attention_mask).detach()
        all_sent_embeds = self.encoder.forward_token(
            input_token_ids, attention_mask=attention_mask).detach()
        sent_embeds = self.encoder.pooling(all_sent_embeds, input_token_ids)
        # adv_outputs = self.adversary(sent_embeds)
        # return adv_outputs
        pred_dist, pred_depth, pred_pow = self.adversary(
            all_sent_embeds, sent_embeds)
        return pred_dist, pred_depth, pred_pow


class ParaBartEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config

        self.dropout = config.dropout
        self.embed_tokens = embed_tokens

        self.embed_synt = nn.Embedding(77, config.d_model, config.pad_token_id)
        self.embed_synt.weight.data.normal_(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id,
            config.extra_pos_embeddings)

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.encoder_layers)])

        if config.use_GAT:
            self.my_synt_layers = nn.ModuleList([
                dglnn.GATConv(
                    in_feats=config.d_model,
                    out_feats=config.d_model // config.encoder_attention_heads,
                    num_heads=config.encoder_attention_heads,
                    feat_drop=0.1,
                    attn_drop=config.attention_dropout,
                    negative_slope=0.2,
                    residual=False,
                    activation=F.relu,
                    allow_zero_in_degree=False,
                    bias=True) for _ in range(config.syntax_encoder_layer_num)
            ])
        else:
            self.my_synt_layers = nn.ModuleList([
                EncoderLayer(config)
                for _ in range(config.syntax_encoder_layer_num)
            ])

        self.layernorm_embedding = LayerNorm(config.d_model)

        self.synt_layernorm_embedding = LayerNorm(config.d_model)

        self.pooling = MeanPooling(config)

    def forward(self, input_token_ids, input_synt_ids, input_token_mask,
                input_synt_mask, graph):
        """[summary]

        Args:
            input_token_ids ([type]): ```(batch_size, max_sent_len + 2)```
            input_synt_ids ([type]): ```(batch_size, max_synt_len + 2)```
            input_token_mask ([type]): ```(batch_size, max_sent_len + 2)```
            input_synt_mask ([type]): ```(batch_size, max_synt_len + 2)```
            graph ([type]): a list of dgl.DGLGraph

        Returns:
            encoder_outputs ```(batch_size, max_sent_len + 2 + max_synt_len + 2)```
            sent_embeds ```(batch_size, 1)```
            x ```(batch_size, max_sent_len + 2, d_model)```
        """
        # pdb.set_trace()
        x = self.forward_token(input_token_ids, input_token_mask)
        # y, graph_representation = self.forward_synt(input_synt_ids, graph)
        if self.config.use_GAT:
            y = self.forward_synt(input_synt_ids, graph)
        else:
            y = self.forward_synt(input_synt_ids, input_synt_mask)

        encoder_outputs = torch.cat((x, y), dim=1)

        sent_embeds = self.pooling(x, input_token_ids)

        return encoder_outputs, sent_embeds, x

    def forward_token(self, input_token_ids, attention_mask):
        if self.training:
            drop_mask = torch.bernoulli(
                self.config.word_dropout *
                torch.ones(input_token_ids.shape)).bool().cuda()
            input_token_ids = input_token_ids.masked_fill(drop_mask, 50264)

        input_token_embeds = self.embed_tokens(
            input_token_ids) + self.embed_positions(input_token_ids)
        x = self.layernorm_embedding(input_token_embeds)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)

        for encoder_layer in self.layers:
            x, _ = encoder_layer(x, encoder_padding_mask=attention_mask)

        x = x.transpose(0, 1)
        return x

    def forward_synt(self, input_synt_ids, graph=None, attention_mask=None):
        if self.config.use_GAT:
            # input_synt_embeds : ```(batch_size, max_sent_len + 2, d_model)```
            input_synt_embeds = self.embed_synt(input_synt_ids)
            y = self.synt_layernorm_embedding(input_synt_embeds)
            batch_size, max_length, d_model = y.shape
            # ```(batch_size, max_length, d_model)``` -> ```(batch_size * max_length, d_model)```
            y = y.reshape(-1, self.config.d_model).contiguous()

            # 0: bos; 1: pad; 2: eos
            # these three kinds of mask doesn't affect the structure
            # flatten the `input_synt_ids` to a 1-D vector for convenience
            valid_input_synt_mask = input_synt_ids.reshape(-1) > 2

            # get the index of valid ids in the 1-D vector
            valid_input_synt_ids = torch.nonzero(valid_input_synt_mask).squeeze(
                1)
            # pdb.set_trace()
            # these embs will be updated by GAT
            valid_input_synt_embs = torch.index_select(y, 0,
                                                       valid_input_synt_ids)

            graph = [g.to(input_synt_ids.device) for g in graph]
            # using dgl in batch
            graphs = dgl.batch(graph)

            for encoder_synt_layer in self.my_synt_layers:
                valid_input_synt_embs = encoder_synt_layer(
                    graphs, valid_input_synt_embs)
                # valid_input_synt_embs ```(num_node, num_head, o_feat)```
                num_node, _, _ = valid_input_synt_embs.shape

                # concate the heads into one vector
                valid_input_synt_embs = valid_input_synt_embs.reshape(
                    num_node, -1).contiguous()

            # graph_representation = dgl.mean_nodes(valid_input_synt_embs)

            # substitute the embs in `y` by new generated embs from GAT
            y[valid_input_synt_mask] = valid_input_synt_embs

            # reshape `y` to 3-D
            return y.reshape(batch_size, max_length, d_model)
        else:
            input_synt_embeds = self.embed_synt(
                input_synt_ids) + self.embed_positions(input_synt_ids)
            y = self.synt_layernorm_embedding(input_synt_embeds)
            y = F.dropout(y, p=self.dropout, training=self.training)

            # B x T x C -> T x B x C
            y = y.transpose(0, 1)

            for encoder_synt_layer in self.my_synt_layers:
                y, _ = encoder_synt_layer(y,
                                          encoder_padding_mask=attention_mask)

            # T x B x C -> B x T x C
            y = y.transpose(0, 1)
            return y

    def embed(self, input_token_ids, attention_mask=None, pool='mean'):
        if attention_mask is None:
            attention_mask = input_token_ids == self.config.pad_token_id

        x = self.forward_token(input_token_ids, attention_mask)

        sent_embeds = self.pooling(x, input_token_ids)
        return sent_embeds


class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, input_token_ids):
        mask = input_token_ids != self.config.pad_token_id
        mean_mask = mask.float() / mask.float().sum(1, keepdim=True)
        x = (x * mean_mask.unsqueeze(2)).sum(1, keepdim=True)
        return x


class ParaBartDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id,
            config.extra_pos_embeddings)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(1)])
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(self, decoder_input_ids, encoder_hidden_states,
                decoder_padding_mask, decoder_causal_mask,
                encoder_attention_mask):

        x = self.embed_tokens(decoder_input_ids) + self.embed_positions(
            decoder_input_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        for idx, decoder_layer in enumerate(self.layers):
            x, _, _ = decoder_layer(x,
                                    encoder_hidden_states,
                                    encoder_attn_mask=encoder_attention_mask,
                                    decoder_padding_mask=decoder_padding_mask,
                                    causal_mask=decoder_causal_mask)

        x = x.transpose(0, 1)

        return x,


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sent_layernorm_embedding = LayerNorm(config.d_model,
                                                  elementwise_affine=False)
        self.adv = nn.Linear(config.d_model, 74)

    def forward(self, sent_embeds):
        x = self.sent_layernorm_embedding(sent_embeds).squeeze(1)
        x = self.adv(x)
        return x


class StructuralDiscriminator(nn.Module):
    def __init__(self, config, rank=64):
        super(StructuralDiscriminator, self).__init__()
        # d_B(h_i, h_j)^2 = (B(h_i - h_j))^T(B(h_i - h_j))
        self.distance_parser = nn.Linear(config.d_model, rank, bias=False)
        # ||depth_i|| = (B h_i)^(B h_i)
        self.depth_parser = nn.Linear(config.d_model, rank, bias=False)

        # self.semantic_to_structure_transform = nn.Linear(config.d_model,
        #                                                  config.d_model,
        #                                                  bias=True)
        # self.pooling = MeanPooling(config)
        self.sent_layernorm_embedding = LayerNorm(config.d_model,
                                                  elementwise_affine=False)
        self.adv = nn.Linear(config.d_model, 74)

    def forward(self, encoder_output, sent_embeds):
        # encoder_output ```(batch_size, max_sent_len + 2, d_model)```
        dist_transformed = self.distance_parser(encoder_output)
        batch_size, max_length, rank = dist_transformed.shape

        dist_transformed = dist_transformed.unsqueeze(2)
        # ```(batch_size, max_len + 2, max_len + 2, rank)```
        dist_transformed = dist_transformed.expand(-1, -1, max_length, -1)

        transposed = dist_transformed.transpose(1, 2)
        diffs = dist_transformed - transposed

        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)

        depth_transformed = self.depth_parser(encoder_output)
        batch_size, max_length, rank = depth_transformed.size()
        norms = torch.bmm(
            depth_transformed.view(batch_size * max_length, 1, rank),
            depth_transformed.view(batch_size * max_length, rank, 1))
        norms = norms.view(batch_size, max_length)

        # all_structure_embeds = self.semantic_to_structure_transform(
        #     encoder_output)
        # structure_embeds = self.pooling(all_structure_embeds, input_token_ids)

        # the splice is used to ignore the `bos` and `eos` token
        # actually, the `eos` is not always the last token in the input
        # so we only ignore the last `pad` or `eos` token.
        # but it doesn't matter
        # return squared_distances[:, 1:-1, 1:-1], norms[:, 1:-1]

        x = self.sent_layernorm_embedding(sent_embeds).squeeze(1)
        x = self.adv(x)

        return squared_distances, norms, x
