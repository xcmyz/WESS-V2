from torch import nn


class WESSLoss(nn.Module):
    """WESS Loss"""

    def __init__(self):
        super(WESSLoss, self).__init__()

    def forward(self, mel_output, gate_predicted, mel_target, gate_target):
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_loss = nn.MSELoss()(mel_output, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_predicted, gate_target)

        return mel_loss + gate_loss, mel_loss, gate_loss
