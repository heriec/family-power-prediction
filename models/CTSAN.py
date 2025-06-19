import torch
import torch.nn as nn


class CrossScaleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_weights = self.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (q.size(-1)**0.5))
        out = torch.bmm(attn_weights, v)
        return self.out_proj(out)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.query.expand(x.size(0), -1, -1)
        K = self.Wk(x)
        V = self.Wv(x)
        attn = torch.softmax(torch.matmul(
            Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5), dim=-1)
        out = torch.matmul(attn, V)
        return out.squeeze(1)


class CTSANModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(CTSANModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.cross_attn = CrossScaleAttention(d_model)
        self.attn_pooling = AttentionPooling(d_model)

        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x_day = x
        x_week = x[:, ::7, :]
        x_month = x[:, ::30, :]
        x_day = self.input_fc(x_day)
        x_week = self.input_fc(x_week)
        x_month = self.input_fc(x_month)

        x_day_enc = self.encoder(x_day)
        x_week_enc = self.encoder(x_week)
        x_month_enc = self.encoder(x_month)

        cross_week = self.cross_attn(x_day_enc, x_week_enc, x_week_enc)
        cross_month = self.cross_attn(x_day_enc, x_month_enc, x_month_enc)

        fused = x_day_enc + cross_week + cross_month
        pooled = self.attn_pooling(fused)

        out = self.output_fc(pooled)
        return out
