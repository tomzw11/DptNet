

import argparse
import numpy as np
from model import DPTNet_base
from mir_eval.separation import bss_eval_sources
from data_loader import DatasetGenerator
from loss import Loss
import mindspore.dataset as ds
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net


parser = argparse.ArgumentParser('Evaluate separation performance using DPTNet')
parser.add_argument('--train_dir', type=str, default="/home/work/user-job-dir/inputs/data_json/test",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
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
parser.add_argument('--win_len', default=1, type=int,
                    help='...')
# minibatch
parser.add_argument('--batch_size', default=3, type=int,   #default =3
                    help='Batch size')
# define 2 parameters for running on modelArts
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="DPTNet-10_890.ckpt",
                    help='Path to model file created by training')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')

def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = DPTNet_base(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim, layer=args.layer, segment_size=args.segment_size,
                        nspk=args.nspk, win_len=args.win_len)
    model.set_train(mode=False)

    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(model, params)
    print(model)

    # Load data
    tt_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tt_loader = ds.GeneratorDataset(tt_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tt_loader = tt_loader.batch(batch_size=2)

    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        mixture_lengths_with_list = mixture_lengths.asnumpy().tolist()
        estimate_source = model(padded_mixture)  # [B, C, T]

        my_loss = Loss()
        _, _, estimate_source, reorder_estimate_source = \
            my_loss(padded_source, estimate_source, mixture_lengths)
        # Remove padding and flat
        mixture = remove_pad(padded_mixture, mixture_lengths_with_list)
        source = remove_pad(padded_source, mixture_lengths_with_list)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad(reorder_estimate_source,
                                     mixture_lengths_with_list)
        # for each utterance
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, _, _, _ = bss_eval_sources(src_ref, src_est)
    sdr0, _, _, _ = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calculate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 3:
        C = inputs.shape[1]
    for i, data in enumerate(inputs):
        if dim == 3:    # [B, C, T]
            results.append(data[:, :inputs_lengths[i]].view(C, -1).asnumpy())
        elif dim == 2:  # [B, T]
            results.append(data[:inputs_lengths[i]].view(-1).asnumpy())
    return results

if __name__ == '__main__':
    arg = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=arg.device_target)
    print(arg)
    evaluate(arg)
