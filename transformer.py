

import mindspore.nn as nn
from mindspore.ops import ReLU
from mindspore.nn.transformer import MultiHeadAttention
from mindspore.nn import Dropout, Dense, LSTM, LayerNorm
from mindspore import Tensor, ops
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, HeUniform
import mindspore
import mindspore.numpy as msnp
import numpy as np


class TransformerEncoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, hidden_size, batch_size):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(batch_size=batch_size*258, hidden_size=d_model,
                                            src_seq_length=250, tgt_seq_length=250,
                                            num_heads=nhead)
        self.sefl_attn2 = MultiHeadAttention(batch_size=batch_size*250, hidden_size=d_model,
                                             src_seq_length=258, tgt_seq_length=258,
                                             num_heads=nhead)
        # Implementation of improved part
        self.lstm = LSTM(d_model, hidden_size, 1, bidirectional=True)
        self.dropout = Dropout(keep_prob=1.0)

        self.init_tensor = initializer(HeUniform(), [d_model, hidden_size*2], mindspore.float16)
        self.linear = Dense(hidden_size*2, d_model, weight_init=self.init_tensor, has_bias=False)
        self.init_weight = msnp.ones((d_model), dtype=mindspore.float16)
        self.init_beta = msnp.zeros((d_model), dtype=mindspore.float16)
        self.norm = LayerNorm([d_model], epsilon=1e-5)

        self.activation = ReLU()
        self.atten_mask = Tensor(np.ones((batch_size*258, 250, 250)), mstype.float16)
        self.atten_mask2 = Tensor(np.ones((batch_size*250, 258, 258)), mstype.float16)
        self.cast = ops.Cast()

    def construct(self, src):
        if src.shape[1] == 250:
            src2, _ = self.self_attn(src, src, src, attention_mask=self.atten_mask)
        else:
            src2, _ = self.sefl_attn2(src, src, src, attention_mask=self.atten_mask2)
        src = src + src2
        src = self.cast(src, mindspore.float32)
        src = self.norm(src)
        src = self.cast(src, mindspore.float16)
        src2 = self.linear(self.activation(self.lstm(src)[0]))
        src = src + src2
        src = self.cast(src, mindspore.float32)
        src = self.norm(src)
        src = self.cast(src, mindspore.float16)
        return src
