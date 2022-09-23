

import math
import mindspore
import mindspore.ops as ops
from mindspore.ops import ReLU, Zeros, Concat
from mindspore import context, Tensor
import mindspore.common.initializer
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeUniform
import mindspore.numpy as msnp
import numpy as np
from transformer import TransformerEncoderLayer

EPS = 1e-8

class Encoder(nn.Cell):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.init_tensor = initializer(HeUniform(), [N, 1, W], mindspore.float16)
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, has_bias=False,
                                  weight_init=self.init_tensor, pad_mode="pad")
        self.expand_dims = ops.ExpandDims()
        self.relu = ReLU()

    def construct(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = self.expand_dims(mixture, 1)  # [B, 1, T]
        mixture_w = self.conv1d_U(mixture)  # [B, N, L]
        mixture_w = self.relu(mixture_w)
        return mixture_w


def big_matrix():
    x = np.zeros((8000, 4000), np.float16)

    for i in range(4000):
        x[2 * i, i] = 1
        x[2 * i + 1, i] = 1
    big_num = Tensor.from_numpy(x)
    return big_num


class Decoder(nn.Cell):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.expand_dims = ops.ExpandDims()
        self.init_weight = initializer(HeUniform(), [W, E], mindspore.float16)
        self.basis_signals = nn.Dense(E, W, has_bias=False, weight_init=self.init_weight)
        self.zeros = ops.Zeros()
        self.concat = ops.Concat(2)
        self.concat3 = ops.Concat(3)
        self.big_num = big_matrix()

    def construct(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [B, E, L]
            est_mask: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        source_w = self.expand_dims(mixture_w, 1) * est_mask  # [B, C, E, L]
        source_w = source_w.transpose((0, 1, 3, 2))
        # S = DV
        est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source = self.overlap_and_add(est_source, self.W//2) # B x C x T
        return est_source

    def overlap_and_add(self, signal, frame_step):
        outer_dimensions = signal.shape[:-2]
        _, frame_length = signal.shape[-2:]

        subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
        a, b = outer_dimensions

        subframe_signal = signal.view((a, b, -1, subframe_length))
        pad = self.zeros((subframe_signal.shape[0], subframe_signal.shape[1],
                          1, subframe_signal.shape[3]), mindspore.float16)

        subframe_signal = self.concat((pad, subframe_signal, pad))
        subframe_signal = subframe_signal.transpose((0, 1, 3, 2))

        subframe_signal_one = subframe_signal[:, :, :, :8000]
        subframe_signal_two = subframe_signal[:, :, :, 8000:16000]
        subframe_signal_three = subframe_signal[:, :, :, 16000:24000]
        subframe_signal_four = subframe_signal[:, :, :, 24000:32000]
        subframe_signal_five = subframe_signal[:, :, :, 32000:40000]
        subframe_signal_six = subframe_signal[:, :, :, 40000:48000]
        subframe_signal_seven = subframe_signal[:, :, :, 48000:56000]
        subframe_signal_eight = subframe_signal[:, :, :, 56000:64000]

        subframe_signal_first_one = ops.matmul(subframe_signal_one, self.big_num)
        subframe_signal_first_two = ops.matmul(subframe_signal_two, self.big_num)
        subframe_signal_first_three = ops.matmul(subframe_signal_three, self.big_num)
        subframe_signal_first_four = ops.matmul(subframe_signal_four, self.big_num)
        subframe_signal_first_five = ops.matmul(subframe_signal_five, self.big_num)
        subframe_signal_first_six = ops.matmul(subframe_signal_six, self.big_num)
        subframe_signal_first_seven = ops.matmul(subframe_signal_seven, self.big_num)
        subframe_signal_first_eight = ops.matmul(subframe_signal_eight, self.big_num)
        result = self.concat3((subframe_signal_first_one, subframe_signal_first_two,
                               subframe_signal_first_three, subframe_signal_first_four,
                               subframe_signal_first_five, subframe_signal_first_six,
                               subframe_signal_first_seven, subframe_signal_first_eight))
        result = result.transpose((0, 1, 3, 2))
        result = result.view((a, b, -1))
        return result


class SingleTransformer(nn.Cell):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, hidden_size, batch_size=2):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=4, hidden_size=hidden_size,
                                                   batch_size=batch_size)

    def construct(self, output_data):
        transformer_output = self.transformer(output_data)
        return transformer_output


# dual-path transformer
class DPT(nn.Cell):
    """
    Deep dual-path transformer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        num_layers: int, number of stacked Transformer layers. Default is 1.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(self, input_size, hidden_size, output_size, batch_size=2, num_layers=1):
        super(DPT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # dual-path transformer
        self.row_transformer = nn.CellList([])
        self.col_transformer = nn.CellList([])
        for _ in range(num_layers):
            self.row_transformer.append(SingleTransformer(input_size, hidden_size, batch_size))
            self.col_transformer.append(SingleTransformer(input_size, hidden_size, batch_size))
        self.prelu = nn.PReLU()
        # self.conv2d = nn.Conv2d(input_size, output_size, 1, weight_init="HeUniform")
        self.init_tensor = initializer(HeUniform(), [output_size, input_size, 1, 1], mindspore.float16)
        self.conv2d = nn.Conv2d(input_size, output_size, 1, weight_init=self.init_tensor)

    def construct(self, input_data):
        # input shape: batch, N, dim1, dim2
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        batch_size, _, dim1, dim2 = input_data.shape
        output_data = input_data
        for i in range(len(self.row_transformer)):
            row_input = output_data.transpose((0, 3, 2, 1)).view((batch_size * dim2, dim1, -1))  # B*dim2, dim1, N
            row_output = self.row_transformer[i](row_input)  # B*dim2, dim1, H

            row_output = row_output.view((batch_size, dim2, dim1, -1)).transpose((0, 3, 2, 1))  # B, N, dim1, dim2

            output_data = row_output

            col_input = output_data.transpose((0, 2, 3, 1)).view((batch_size * dim1, dim2, -1))  # B*dim1, dim2, N
            col_output = self.col_transformer[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view((batch_size, dim1, dim2, -1)).transpose((0, 3, 1, 2))  # B, N, dim1, dim2
            output_data = col_output

        output_data = self.prelu(output_data)
        output_data = self.conv2d(output_data)

        return output_data


class BF_module(nn.Cell):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=6, segment_size=250, batch_size=2):
        super(BF_module, self).__init__()

        # gated output layer

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.batch_size = batch_size

        self.eps = 1e-8

        self.zero = Zeros()
        self.concat = Concat(2)
        self.concat2 = Concat(3)

        # bottleneck
        self.print = ops.Print()

        self.init_tensor = initializer(HeUniform(), [self.feature_dim, self.input_dim, 1], mindspore.float16)
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, weight_init=self.init_tensor)

        # DPT model
        self.DPT = DPT(self.feature_dim, self.hidden_dim, self.feature_dim * self.num_spk, self.batch_size,
                       num_layers=self.layer)

        self.init_tensor2 = initializer(HeUniform(), [self.feature_dim, self.feature_dim, 1], mindspore.float16)
        self.output = nn.SequentialCell(nn.Conv1d(self.feature_dim, self.feature_dim, 1, weight_init=self.init_tensor2),
                                        nn.Tanh())

        self.output_gate = nn.SequentialCell(nn.Conv1d(self.feature_dim, self.feature_dim, 1,
                                                       weight_init=self.init_tensor2),
                                             nn.Sigmoid())


    def construct(self, input_data):
        # input: (B, E, T)
        batch_size, _, _ = input_data.shape

        enc_feature = self.BN(input_data) # (B, E, L)-->(B, N, L)    #error

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K: L is the segment_size

        output_data = self.DPT(enc_segments).view((batch_size * self.num_spk, self.feature_dim, self.segment_size, -1))  # B*nspk, N, L, K

        # overlap-and-add of the outputs  #[4, 64, 31999]
        output_data = self.merge_feature(output_data, enc_rest)  # B*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output_data) * self.output_gate(output_data)  # B*nspk, K, T
        bf_filter = bf_filter.transpose((0, 2, 1)).view((batch_size, self.num_spk, -1, self.feature_dim))  # B, nspk, T, N

        return bf_filter

    def pad_segment(self, input_data, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input_data.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = self.zero((batch_size, dim, rest), mindspore.float16)
            # pad = self.zero((batch_size, dim, rest), mindspore.float32)
            input_data = self.concat((input_data, pad))       #e

        pad_aux = self.zero((batch_size, dim, segment_stride), mindspore.float16)
        # pad_aux = self.zero((batch_size, dim, segment_stride), mindspore.float32)
        input_data = self.concat((pad_aux, input_data, pad_aux))

        return input_data, rest

    def split_feature(self, input_data, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input_data, rest = self.pad_segment(input_data, segment_size)    #e
        batch_size, dim, _ = input_data.shape
        segment_stride = segment_size // 2

        segments1 = input_data[:, :, :-segment_stride].view((batch_size, dim, -1, segment_size))
        segments2 = input_data[:, :, segment_stride:].view((batch_size, dim, -1, segment_size))
        segments = self.concat2((segments1, segments2)).\
                   view((batch_size, dim, -1, segment_size)).transpose((0, 1, 3, 2))

        return segments, rest

    def merge_feature(self, input_data, rest):
        # merge the split features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input_data.shape
        segment_stride = segment_size // 2
        input_data = input_data.transpose((0, 1, 3, 2)).view((batch_size, dim, -1, segment_size * 2)) # B, N, K, L

        input1 = input_data[:, :, :, :segment_size].view((batch_size, dim, -1))[:, :, segment_stride:]
        input2 = input_data[:, :, :, segment_size:].view((batch_size, dim, -1))[:, :, :-segment_stride]

        output_data = input1 + input2
        if rest > 0:
            output_data = output_data[:, :, :-rest]

        return output_data  # B, N, T


# base module for DPTNet_base
class DPTNet_base(nn.Cell):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250, nspk=2, win_len=2, batch_size=2):
        super(DPTNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8
        self.batch_size = batch_size

        self.relu = ReLU()
        self.zeros = Zeros()
        self.concat = Concat(1)

        # waveform encoder
        self.encoder = Encoder(win_len, enc_dim) # [B T]-->[B N L]
        self.group_weight = Tensor(msnp.ones((self.enc_dim), dtype=mindspore.float16))
        self.group_bias = Tensor(msnp.zeros((self.enc_dim), dtype=mindspore.float16))
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=self.eps, affine=True) # [B N L]-->[B N L]
        self.enc_LN_temp = nn.GroupNorm(1, self.enc_dim)
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                   self.num_spk, self.layer, self.segment_size, self.batch_size)
        # [B, N, L] -> [B, E, L]
        self.init_tensor = initializer(HeUniform(), [self.enc_dim, self.feature_dim, 1], mindspore.float16)
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, has_bias=False, weight_init=self.init_tensor)
        self.decoder = Decoder(enc_dim, win_len)
        self.cast = ops.Cast()


    def construct(self, input_data):
        """
        input_data: shape (batch, T)
        """
        # pass to a DPT
        B, _ = input_data.shape
        mixture_w = self.encoder(input_data)  # B, E, L

        mixture_w_t = mixture_w.expand_dims(axis=0).transpose((0, 2, 1, 3))
        # print('mixture_w.shape {}'.format(mixture_w.shape))
        mixture_w_t = self.cast(mixture_w_t, mindspore.float32)
        score_ = self.enc_LN(mixture_w_t) # B, E, L
        score_ = self.cast(score_, mindspore.float16)

        score_ = score_.transpose((0, 2, 1, 3)).squeeze(axis=0)   #e
        score_ = self.separator(score_)
        score_ = score_.view((B*self.num_spk, -1, self.feature_dim)).transpose((0, 2, 1))  # B*nspk, N, T

        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]

        score = score.view((B, self.num_spk, self.enc_dim, -1))  # [B*nspk, E, L] -> [B, nspk, E, L]

        est_mask = self.relu(score)

        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        return est_source


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    model = DPTNet_base(enc_dim=256, feature_dim=64, hidden_dim=128,
                        layer=1, segment_size=250, nspk=2,
                        win_len=2, batch_size=2)
    print(model)
    x1 = np.ones((2, 32000)).astype(np.float16)
    y = mindspore.Tensor.from_numpy(x1)
    output = model(y)
