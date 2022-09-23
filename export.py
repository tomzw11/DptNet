
'''
##############evaluate trained models#################
python export.py
'''

import argparse
import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model import DPTNet_base


parser = argparse.ArgumentParser(
    "Dual-path transformer"
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--enc_dim', default=256, type=int,
                    help='...')
parser.add_argument('--feature_dim', default=64, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--hidden_dim', default=128, type=int,
                    help='...')
parser.add_argument('--layer', default=6, type=int,
                    help='Number of repeats')
parser.add_argument('--segment_size', default=250, type=int,
                    help='segment size')
parser.add_argument('--nspk', default=2, type=int,
                    help='Maximum number of speakers')
parser.add_argument('--win_len', default=2, type=int,
                    help='...')
parser.add_argument('--ckpt', default='DPTNet-100_890.ckpt',
                    help='Location to save epoch models')

def export_DPTNet():
    """ export_dptnet """
    args = parser.parse_args()
    net = DPTNet_base(args.enc_dim, args.feature_dim,
                      args.hidden_dim, args.layer, args.segment_size,
                      args.nspk, args.win_len)
    param_dict = load_checkpoint(args.ckpt)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.random.uniform(0.0, 1.0, size=[1, 32000]).astype(np.float16))
    export(net, input_data, file_name='DPTNet', file_format='MINDIR')
    print("export success")

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2)
    export_DPTNet()
