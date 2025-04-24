# coding: utf-8
# Apache-2.0  (PaddlePaddle)

from __future__ import absolute_import, division, print_function
import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F

def _linear_with_decay(in_dim, out_dim, fc_decay):
    regularizer = paddle.regularizer.L2Decay(fc_decay)
    stdv = 1. / math.sqrt(in_dim)
    init = nn.initializer.Uniform(-stdv, stdv)
    w_attr = ParamAttr(regularizer=regularizer, initializer=init)
    b_attr = ParamAttr(regularizer=regularizer, initializer=init)
    return nn.Linear(in_dim, out_dim, weight_attr=w_attr, bias_attr=b_attr)

class CTCHeadTLast(nn.Layer):
    """
    CTC head that can accept (B, T, C)  *or*  (B, C, T).

    Args added:
        channel_last (bool):  If True the incoming tensor is (B, T, C);
                              if False it is (B, C, T).  Default False.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=4e-4,
                 mid_channels=None,
                 return_feats=False,
                 channel_last=False,   # ← new
                 **kwargs):
        super().__init__()
        self.channel_last = channel_last
        self.mid_channels = mid_channels
        self.return_feats = return_feats

        if mid_channels is None:
            self.fc = _linear_with_decay(in_channels, out_channels, fc_decay)
        else:
            self.fc1 = _linear_with_decay(in_channels, mid_channels, fc_decay)
            self.fc2 = _linear_with_decay(mid_channels, out_channels, fc_decay)

    def forward(self, x, **_):
        if x.ndim == 4:
            if self.channel_last:
                x = x.squeeze(1)
            else:
                x = x.squeeze(2)

        if self.channel_last:
            x = x.transpose([0, 2, 1])

        b, c, t = x.shape
        x = x.reshape([b * t, c])

        if self.mid_channels is None:
            logits = self.fc(x)
        else:
            x = self.fc1(x)
            logits = self.fc2(x)

        logits = logits.reshape([b, t, -1])
        if not self.training:
            logits = F.softmax(logits, axis=2)
        return (x, logits) if self.return_feats else logits
