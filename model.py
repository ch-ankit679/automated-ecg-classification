import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Swish Activation
# -------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -------------------------
# Conv Block (EXACT MATCH)
# -------------------------
class ConvNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)

        self.normalization_1 = nn.BatchNorm1d(hidden_size)
        self.normalization_2 = nn.BatchNorm1d(hidden_size)
        self.normalization_3 = nn.BatchNorm1d(hidden_size)

        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()

        self.pool = nn.MaxPool1d(2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)

        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)

        x = F.pad(x, (self.kernel_size - 1, 0))
        conv3 = self.conv_3(x)

        x = self.normalization_3(conv1 + conv3)
        x = self.swish_3(x)

        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.pool(x)

        return x

# -------------------------
# RNN (EXACT MATCH)
# -------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hid_size):
        super().__init__()

        self.rnn_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hid_size,
            batch_first=True
        )

    def forward(self, input):
        outputs, hidden = self.rnn_layer(input)
        return outputs, hidden

# -------------------------
# FINAL MODEL (EXACT MATCH)
# -------------------------
class RNNAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer = RNN(input_size=46, hid_size=64)

        self.conv1 = ConvNormPool(1, 64, 5)
        self.conv2 = ConvNormPool(64, 64, 5)

        self.avgpool = nn.AdaptiveMaxPool1d(1)

        self.attn = nn.Linear(64, 64, bias=False)
        self.fc = nn.Linear(64, 5)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)

        x_out, hid_states = self.rnn_layer(x)

        x = torch.cat([hid_states[0], hid_states[1]], dim=0).transpose(0, 1)

        x_attn = torch.tanh(self.attn(x))
        x = x_attn.bmm(x_out)

        x = x.transpose(2, 1)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = F.softmax(self.fc(x), dim=-1)

        return x

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    model = RNNAttentionModel()
    model.load_state_dict(torch.load("model/ecg_model.pth", map_location="cpu"))
    model.eval()
    return model