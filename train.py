

import argparse
import os
from preprocess import preprocess
from data_loader import DatasetGenerator
from lr_sch import dynamic_lr
from network_define import WithLossCell
from model import DPTNet_base
from loss import Loss
from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
import mindspore.dataset as ds



parser = argparse.ArgumentParser(
    "Dual-path transformer"
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default="/home/work/user-job-dir/inputs/data_json/tr",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
# Network architecture
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
# Training config
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--device_num', default=2, type=int,
                    help='device num')
parser.add_argument('--device_id', default=2, type=int,
                    help='device id')
# minibatch
parser.add_argument('--batch_size', default=3, type=int,
                    help='Batch size')
# optimizer
parser.add_argument('--lr', default=5e-6, type=float,
                    help='Init learning rate')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--continue_train', default=0, type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--step_per_epoch', default=7120, type=int,
                    help='...')
parser.add_argument('--ckpt_path', type=str, default="DPTNet-10_890.ckpt",
                    help='Path to model file created by training')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--modelArts', default=0, type=int,
                    help='Continue from checkpoint model')
# modelarts
parser.add_argument('--data_url', default='/home/work/user-job-dir/inputs/data/',
                    help='path to training/inference dataset folder')
parser.add_argument('--train_url', default='/home/work/user-job-dir/model/',
                    help='model folder to save/load')
parser.add_argument('--in_dir', type=str, default=r"/home/work/user-job-dir/inputs/data/",
                    help='Directory path of wsj0 including tr, cv and tt')
parser.add_argument('--out_dir', type=str, default=r"/home/work/user-job-dir/inputs/data_json",
                    help='Directory path to put output files')

def main(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.device_num)
        context.set_auto_parallel_context(parameter_broadcast=True)
        print("Starting traning on multiple devices...")
    else:
        if args.modelArts:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
        else:
            context.set_context(device_id=args.device_id)

    if args.modelArts:
        import moxing as mox
        obs_data_url = args.data_url
        args.data_url = '/home/work/user-job-dir/inputs/data/'
        obs_train_url = args.train_url

        home = os.path.dirname(os.path.realpath(__file__))
        train_dir = os.path.join(home, 'checkpoints') + str(rank_id)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        save_checkpoint_path = train_dir + '/device_' + os.getenv('DEVICE_ID') + '/'
        if not os.path.exists(save_checkpoint_path):
            os.makedirs(save_checkpoint_path)
        save_ckpt = os.path.join(save_checkpoint_path, 'dptnet.ckpt')

        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, args.data_url))

        print("start preprocess on modelArts....")
        preprocess(args)

    print("Start datasetgenerator")
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)

    print("start Generatordataset")
    if is_distributed == 'True':
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False, num_shards=rank_size, shard_id=rank_id)
    else:
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False)
    tr_loader = tr_loader.batch(2)

    print("data loading done")
    # model
    net = DPTNet_base(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                      hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size,
                      nspk=args.nspk, win_len=args.win_len)

    if args.continue_train:
        if args.modelArts:
            home = os.path.dirname(os.path.realpath(__file__))
            ckpt = os.path.join(home, args.ckpt_path)
            params = load_checkpoint(ckpt)
            load_param_into_net(net, params)
        else:
            params = load_checkpoint(args.ckpt_path)
            load_param_into_net(net, params)

    print(net)
    net.set_train()

    lr = dynamic_lr(args.step_per_epoch, args.epochs)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=args.l2)

    my_loss = Loss()
    net_with_loss = WithLossCell(net, my_loss)
    model = Model(net_with_loss, optimizer=optimizer)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(1)
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=5,
                                 keep_checkpoint_max=5)
    if args.modelArts:
        ckpt_cb = ModelCheckpoint(prefix="DPTNet", directory=save_ckpt, config=config_ck)
    else:
        ckpt_cb = ModelCheckpoint(prefix="DPTNet", directory=args.save_folder, config=config_ck)
    cb += [ckpt_cb]

    model.train(epoch=args.epochs, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=False)

    if args.modelArts:
        import moxing as mox
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))

if __name__ == '__main__':
    arg = parser.parse_args()
    print(arg)
    main(arg)
