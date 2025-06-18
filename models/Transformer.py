import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return out
