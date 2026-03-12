import datetime
import os
import pprint
import time
import threading
import torch as th
import csv
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import random
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math
import copy


from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from modules.transformer.embed import PositionalEmbedding,TemporalEmbedding,TimeFeatureEmbedding,DataEmbedding


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = th.ones(L, scores.shape[-1], dtype=th.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[th.arange(B)[:, None, None],
                    th.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if th.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads) # [256 / 4] = 64
        d_values = d_values or (d_model//n_heads) # [256 / 4] = 64

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model) # [256->256]
        self.n_heads = n_heads
        self.mix = mix # false

    def forward(self, queries, keys, values, attn_mask): # first three items are the same
        B, L, _ = queries.shape
        _, S, _ = keys.shape # S and L are 20
        H = self.n_heads #4

        queries = self.query_projection(queries).view(B, L, H, -1) # [256->256] -> [36,20,4,64]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [256->256] -> [36,20,4,64]
        values = self.value_projection(values).view(B, S, H, -1) # [256->256] -> [36,20,4,64]

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask # None
        )
        if self.mix: # decoder first layer
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1) # (B, 20, )

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with th.no_grad():
            self._mask = th.triu(th.ones(mask_shape, dtype=th.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape # 36,20,4,64
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = th.einsum("blhe,bshe->bhls", queries, keys) # [b,head, 20,20]
        if self.mask_flag: # bhle bhes
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(th.softmax(scale * scores, dim=-1))
        V = th.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor # no need
        self.scale = scale # none
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = th.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, th.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = th.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - th.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[th.arange(B)[:, None, None],
                   th.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = th.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:# Empty
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = th.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[th.arange(B)[:, None, None],
        th.arange(H)[None, :, None],
        index, :] = th.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (th.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[th.arange(B)[:, None, None], th.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape # [36,20,4,64]
        _, L_K, _, _ = keys.shape # [36,20,4,64]

        queries = queries.transpose(2, 1) # [36,4,20,64]
        keys = keys.transpose(2, 1)  # [36,4,20,64]
        values = values.transpose(2, 1)  # [36,4,20,64]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        # y = self.dropout(self.conv2(y).transpose(-1,1))
        #
        # return self.norm3(x+y)
        return  x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)# [Encoderlayer(AttentionLayer(ProbAtten)), Encoderlayer(AttentionLayer(ProbAtten))]
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None # ConvLayer
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:# if no distill go here
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model # d_ff = 1024
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        # y = self.dropout(self.conv2(y).transpose(-1,1))
        #
        # return self.norm2(x+y), attn
        return x, attn


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=th.device('cuda:0')): # 2 e_layers
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention # False

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention  # calculate Q and K attention score, no need find top-k
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers) # 2 e_layers
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1) # 1
            ] if distil else None,
            norm_layer=th.nn.LayerNorm(d_model) # 256
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix), # !!!Mask Flag = True
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=th.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)  # becomes 7

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [batch, seq 20, 256]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # [batch, seq 20, 256] pass inside

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = F.relu(self.projection(dec_out))

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:  # run this
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Exp_Informer():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.args.episode_limit = self.args.episode_limit
        self.timestep_feature = th.linspace(0, self.args.episode_limit+self.args.informer_seq_len-1, steps=self.args.episode_limit+self.args.informer_seq_len-1) # [720 + 20 - 1] because we need add dummy data with zero (seq_len) at the top of the episode_batch_obs data, so that RL agent (informer consume the dummy data and produce predicted obs at step 0) will see the predicted value when we start the env at step 0
        self.args.n_agents = self.args.n_agents
        self.args.model = "informer"
        self.args.padding = 0
        self.args.features = "M"
        self.args.e_layers = 2
        self.args.enc_in = self.args.obs_shape # obs encoder input size
        self.args.dec_in = self.args.obs_shape # obs decoder input size
        self.args.c_out = self.args.obs_shape # obs # output size
        self.seq_len = self.args.informer_seq_len
        self.label_len = self.args.informer_label_len
        self.pred_len = self.args.informer_pred_len
        self.args.factor = 5
        self.args.d_model = 256 # dimension of model
        self.args.n_heads = 4
        self.args.d_layers = 1
        self.args.d_ff = 1024 #dimension of fcn
        self.args.dropout = 0.05
        self.args.attn = "prob"
        self.args.embed = "timeF"
        self.args.freq="s"
        self.args.activation="gelu"
        self.args.output_attention=False
        self.args.distil=False
        self.args.mix=True
        self.device

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict["informer"](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.seq_len,
                self.label_len,
                self.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        return model

    def _get_data(self, flag):
        args = self.args
        timeenc = 0 if args.embed != 'timeF' else 1

    def _select_optimizer(self):
        model_optim = Adam(self.model.parameters(), lr=0.001)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, episode_batch_obs_data, agent_index): # receive data here # [1,24,9,12]
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        seq_len = self.seq_len
        label_len = self.label_len
        pred_len = self.pred_len

        episode_limit = self.args.episode_limit
        max_start = episode_limit - seq_len - pred_len

        for epoch in range(2):  # train informer epoch times
            train_loss = []
            batch_x = []
            batch_y = []
            batch_x_mark = []
            batch_y_mark = []

            for _ in range(episode_limit // seq_len):  # sample data times to form a batch, for loop iteration num = batch_size, now 720/20 is 36
                s_begin = random.randint(0, max_start)
                s_end = s_begin + seq_len
                r_begin = s_end - label_len
                r_end = r_begin + label_len + pred_len

                seq_x = episode_batch_obs_data[0, s_begin:s_end, agent_index, :]
                seq_y = episode_batch_obs_data[0, r_begin:r_end, agent_index, :]
                seq_x_mark = self.timestep_feature[s_begin:s_end]
                seq_y_mark = self.timestep_feature[r_begin:r_end]

                batch_x.append(seq_x)
                batch_y.append(seq_y)
                batch_x_mark.append(seq_x_mark)
                batch_y_mark.append(seq_y_mark)

            batch_x = th.stack(batch_x)  # [batch_size, enc_len, state_dim]
            batch_y = th.stack(batch_y)  # [batch_size, dec_len, state_dim]
            batch_x_mark = th.stack(batch_x_mark)
            batch_y_mark = th.stack(batch_y_mark)

            model_optim.zero_grad()
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred, true)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        # print("Informer Innner Epoch: {} | Training Loss {}".format(epoch, train_loss))

        return self.model, train_loss

    def predict(self, obs_data, env_time_index_data):
        # obs_data  [enc_len, state_dim]
        # env_time_index_data  [enc_len, 1]
        obs_data = obs_data.unsqueeze(0)
         #obs_data need [batch_size, enc_len, state_dim]

        self.model.eval()


        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        s_time_index_begin = env_time_index_data[0]
        s_time_index_begin_end = s_time_index_begin + self.seq_len
        r_time_index_begin = s_time_index_begin_end - self.label_len
        r_time_index_end = r_time_index_begin + self.label_len + self.pred_len

        batch_x = obs_data[:,s_begin:s_end,:]
        batch_y = obs_data[:,r_begin:r_end,:]
        batch_x_mark = self.timestep_feature[s_time_index_begin:s_time_index_begin_end]
        batch_y_mark = self.timestep_feature[r_time_index_begin:r_time_index_end]

        pred, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred = pred[:,:,:].detach().cpu().numpy()
        return pred

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:  # run this
            dec_inp = th.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).float().to(self.device) # [36,1,obs_dim]
        dec_inp = th.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()  # label_len is 19, last value is 0 (pred_data)
         # concate 48 and 24 to become 72.   from index 48 to 72, data value is 0 at initial
        # encoder - decoder
        # run this
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # run this [1,1,12]

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.pred_len:, f_dim:]

        return outputs, batch_y





def run(_config):
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Run and train
    run_sequential(args=args)
    print("Exiting Main")

def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args):

    informer_process_obs_ways = args.informer_process_obs_ways #concat or avg or replace
    informer_obs_duplicate_time = args.informer_pred_len + 1 # if concat, the multiplicaton should be 2 as default. duplicate obs twice

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"] # per agent
    args.avail_actions = env_info["n_actions"] # for example [4], got 4 possible action
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]


    print(args)

    # Default/Base scheme
    if informer_process_obs_ways == "concat" and args.seq2seq == True:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"]*informer_obs_duplicate_time, "group": "agents"}, # Concatenate ways of informer, RL agents receive k-times of obs shape
            "informer_obs": {"vshape": env_info["obs_shape"], "group": "agents"}, # original obs shape feed for informer
            "next_obs": {"vshape": env_info["obs_shape"]*informer_obs_duplicate_time, "group": "agents"},
            "actions": {"vshape": env_info["n_actions_shape"], "group": "agents", "dtype": th.int64},
            "actions_onehot":{"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.uint8},
            "reward": {"vshape": (1,), "group": "agents"} if args.name == "iql" else {"vshape": (1,)}, # has independent reward and global_reward
            "done": {"vshape": (1,), "dtype": th.uint8},
        }
    else:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # Concatenate ways of informer
            "next_obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": env_info["n_actions_shape"], "group": "agents", "dtype": th.int64},
            "actions_onehot": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.uint8},
            "reward": {"vshape": (1,), "group": "agents"} if args.name == "iql" else {"vshape": (1,)},
            # has independent reward and global_reward
            "done": {"vshape": (1,), "dtype": th.uint8},
        }
    groups = {
        "agents": args.n_agents,
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

                                                                 #500              #env seq length
    buffer = ReplayBuffer(scheme, groups, args.batch_size, args.buffer_size, env_info["episode_limit"], args.seq2seq, args.informer_seq_len, args.informer_pred_len, args.on_policy_learning,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, device=args.device)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, None, args)

    if args.use_cuda:
        learner.cuda()

    # Initialized Informer model for agents
    if args.seq2seq == True:
        Informer_agent_models = []
        for agent_num in range(args.n_agents):
            Informer_agent_models.append(Exp_Informer(args))


    if args.checkpoint_path != "": # PASS

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            # logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        # logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    # last_test_T = -args.test_interval - 1 # -10000-1
    # last_log_T = 0
    # model_save_time = 0

    # logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max)) # 2050000

    episode_list = []
    avg_attention_score_2Dlist = []
    single_sample_attention_score_2Dlist=[]
    single_sample_encoded_hidden_states_2Dlist = []
    single_sample_attn_query_2Dlist = []
    single_sample_attn_key_2Dlist = []
    single_sample_attn_logit_2Dlist = []


    episodes_reward_list = []
    episodes_seq2seq_loss_list = []
    episodes_info_result_dic = {}
    while episode < args.t_max: # 300 epochs
        time_start = time.time()

        print("Epoch {} starting".format(episode))
        # Run for a whole episode at a time

        if args.seq2seq == True:
            if episode < 1: # < 4, informer turn off to let the informer train a bit before we deploy
                episode_batch, episode_reward,resultDic = runner.run(test_mode=False,informer_process_obs_ways=informer_process_obs_ways, seq2seq=args.seq2seq) # return seq_batch
            else:# informer turn on
                episode_batch, episode_reward,resultDic = runner.run(Informer_agent_models, test_mode=False,informer_process_obs_ways=informer_process_obs_ways,seq2seq=args.seq2seq) # return seq_batch
            episode_batch_informer_obs_data = episode_batch.transition_data["informer_obs"].clone()
            #Informer start
            time_informer_start = time.time()
            train_losses = []
            train_losses_all_agent_single_value = 0
            for agent_num in range (args.n_agents):
                # call agent's informer
                _,train_loss = Informer_agent_models[agent_num].train(episode_batch_informer_obs_data,agent_num)
                train_losses.append(train_loss)
            train_losses_all_agent_single_value = np.average(train_losses)
            episodes_seq2seq_loss_list.append(train_losses_all_agent_single_value)
            time_informer_end = time.time()
            print("Informer training time for all agents:{}s".format(time_informer_end - time_informer_start))
            print("Informer training loss for all agents(avg):{}".format(train_losses_all_agent_single_value))
            #End Informer
        else:
            episode_batch, episode_reward, resultDic = runner.run(test_mode=False)  # return seq_batch

        buffer.insert_episode_batch(episode_batch.transition_data)

        # if buffer.can_sample(args.batch_size): #32
        for i in range(args.num_epochs): # inner training loop # 1 as value
            episode_sample = buffer.sample(args.batch_size)

            # if episode_sample.device != args.device:
            #     episode_sample.to(args.device)


            avg_attention_score_csv_data,single_sample_attention_score_csv_data, single_sample_encoded_hidden_states, single_sample_attn_query, single_sample_attn_key,single_sample_attn_logit= learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        if ((episode+1) % args.test_nepisode) == 0: # every test_nepisode epoch test, print results to csv file
            runner.run(episode=episode+1,test_mode=True,informer_process_obs_ways=informer_process_obs_ways,seq2seq=args.seq2seq)
            print("Reward for evaluation episode{}:{}".format(episode+1,episode_reward))

        # n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
        #
        #     # logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
        #     # logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
        #     #     time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
        #     # last_time = time.time()
        #
        #     last_test_T = runner.t_env
        #     for _ in range(n_test_runs):
        #         runner.run(test_mode=True)

        # if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
        #     model_save_time = runner.t_env
        #     save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        #     #"results/models/{}".format(unique_token)
        #     os.makedirs(save_path, exist_ok=True)
        #     # logger.console_logger.info("Saving models to {}".format(save_path))
        #
        #     # learner should handle saving/loading -- delegate actor save/load to mac,
        #     # use appropriate filenames to do critics, optimizer states
        #     learner.save_models(save_path)

        # episode += args.batch_size_run
        episode += 1
        episodes_reward_list.append(episode_reward)
        for key,value in resultDic.items():
            if key not in episodes_info_result_dic:
                episodes_info_result_dic[key] = []
            episodes_info_result_dic[key].append(value)

        # if (runner.t_env - last_log_T) >= args.log_interval:
        #     # logger.log_stat("episode", episode, runner.t_env)
        #     # logger.print_recent_stats()
        #     last_log_T = runner.t_env

        end_time = time.time()
        execution_time = end_time - time_start
        print("Episode:",episode)
        print("Program time taken(s):",execution_time,"s")
        print("Reward:",episode_reward)
        num_total_episode_list = list(range(len(episodes_reward_list)))
        with open(f'csv_plot/{args.csv_name}.csv', 'w+', newline='') as f:
            write = csv.writer(f)
            write.writerow(["Epochs"]+num_total_episode_list)
            write.writerow(["Reward"]+episodes_reward_list)
            write.writerow(["seq2seqLoss"] + episodes_seq2seq_loss_list)
            for key, value in episodes_info_result_dic.items(): # record system_accumulated_waiting_times,  system_total_stopped, system_mean_waiting_time, system_mean_speed
                write.writerow([key]+value)


        if avg_attention_score_csv_data and single_sample_attention_score_csv_data is not None:
            # Execute record attention value once in a while
            if ((episode + 1) % args.record_attention_interval) == 0:
                print(args.record_attention_interval)
                episode_list.append(episode)
                avg_attention_score_2Dlist.append(avg_attention_score_csv_data)
                single_sample_attention_score_2Dlist.append(single_sample_attention_score_csv_data)
                single_sample_encoded_hidden_states_2Dlist.append(single_sample_encoded_hidden_states)
                single_sample_attn_query_2Dlist.append(single_sample_attn_query)
                single_sample_attn_key_2Dlist.append(single_sample_attn_key)
                single_sample_attn_logit_2Dlist.append(single_sample_attn_logit)


                with open(f'csv_plot/{args.csv_name}_Attention_score.csv',
                          'w+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(num_total_episode_list)

                    for i in range(len(episode_list)):
                        writer.writerow(["Episode", episode_list[i]])
                        header = [f"Neighbor_{i}" for i in range(len(avg_attention_score_csv_data))] + ["Average Attention Score"]
                        writer.writerow(header)
                        for row in avg_attention_score_2Dlist[i]:
                            writer.writerow(row)
                        header = [f"Neighbor_{i}" for i in range(len(single_sample_attention_score_csv_data))] + ["Single Sample Attention Score"]
                        writer.writerow(header)
                        for row in single_sample_attention_score_2Dlist[i]:
                            writer.writerow(row)
                        header = [f"Neighbor_{i}" for i in range(len(single_sample_attention_score_csv_data))] + ["Single Sample encoded_hidden_states"]
                        writer.writerow(header)
                        for row in single_sample_encoded_hidden_states_2Dlist[i]:
                            writer.writerow(row)
                        header = [f"Neighbor_{i}" for i in range(len(single_sample_attention_score_csv_data))] + ["Single Sample attn_query"]
                        writer.writerow(header)
                        for row in single_sample_attn_query_2Dlist[i]:
                            writer.writerow(row)
                        header = [f"Neighbor_{i}" for i in range(len(single_sample_attention_score_csv_data))] + ["Single Sample attn_key"]
                        writer.writerow(header)
                        for row in single_sample_attn_key_2Dlist[i]:
                            writer.writerow(row)
                        header = [f"Neighbor_{i}" for i in range(len(single_sample_attention_score_csv_data))] + ["Single Sample attn_logit"]
                        writer.writerow(header)
                        for row in single_sample_attn_logit_2Dlist[i]:
                            writer.writerow(row)
    runner.close_env()



    # logger.console_logger.info("Finished Training")




def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
