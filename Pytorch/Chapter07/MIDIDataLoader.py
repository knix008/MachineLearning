import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data

from MIDI import MidiDataRead

def md_fl_to_pio_rl(md_fl):
    md_d = MidiDataRead(md_fl, dtm=0.3)
    pio_rl = md_d.pio_rl.transpose()
    pio_rl[pio_rl > 0] = 1
    return pio_rl


def pd_pio_rl(pio_rl, mx_l=132333, pd_v=0):
    orig_rol_len = pio_rl.shape[1]
    pdd_rol = np.zeros((88, mx_l))
    pdd_rol[:] = pd_v
    pdd_rol[:, -orig_rol_len:] = pio_rl
    return pdd_rol


class NtGenDataset(data.Dataset):
    def __init__(self, md_pth, mx_seq_ln=1491):
        self.md_pth = md_pth
        md_fnames = os.listdir(md_pth)
        self.mx_seq_ln = mx_seq_ln
        md_fnames_ful = map(lambda fname: os.path.join(md_pth, fname), md_fnames)
        self.md_fnames_ful = list(md_fnames_ful)
        if mx_seq_ln is None:
            self.mx_len_upd()

    def mx_len_upd(self):
        seq_lens = map(
            lambda fname: md_fl_to_pio_rl(fname).shape[1], self.md_fnames_ful
        )
        mx_l = max(seq_lens)
        self.mx_seq_ln = mx_l

    def __len__(self):
        return len(self.md_fnames_ful)

    def __getitem__(self, index):
        md_fname_ful = self.md_fnames_ful[index]
        pio_rl = md_fl_to_pio_rl(md_fname_ful)
        seq_len = pio_rl.shape[1] - 1
        ip_seq = pio_rl[:, :-1]
        gt_seq = pio_rl[:, 1:]
        ip_seq_pad = pd_pio_rl(ip_seq, mx_l=self.mx_seq_ln)
        gt_seq_pad = pd_pio_rl(gt_seq, mx_l=self.mx_seq_ln, pd_v=-100)
        ip_seq_pad = ip_seq_pad.transpose()
        gt_seq_pad = gt_seq_pad.transpose()
        return (
            torch.FloatTensor(ip_seq_pad),
            torch.LongTensor(gt_seq_pad),
            torch.LongTensor([seq_len]),
        )


def pos_proc_seq(btch):
    ip_seqs, op_seqs, lens = btch
    ip_seq_splt_btch = ip_seqs.split(split_size=1)
    op_seq_splt_btch = op_seqs.split(split_size=1)
    btch_splt_lens = lens.split(split_size=1)
    tr_data_tups = zip(ip_seq_splt_btch, op_seq_splt_btch, btch_splt_lens)
    ord_tr_data_tups = sorted(tr_data_tups, key=lambda c: int(c[2]), reverse=True)
    ip_seq_splt_btch, op_seq_splt_btch, btch_splt_lens = zip(*ord_tr_data_tups)
    ord_ip_seq_btch = torch.cat(ip_seq_splt_btch)
    ord_op_seq_btch = torch.cat(op_seq_splt_btch)
    ord_btch_lens = torch.cat(btch_splt_lens)
    ord_ip_seq_btch = ord_ip_seq_btch[:, -ord_btch_lens[0, 0] :, :]
    ord_op_seq_btch = ord_op_seq_btch[:, -ord_btch_lens[0, 0] :, :]
    tps_ip_seq_btch = ord_ip_seq_btch.transpose(0, 1)
    ord_btch_lens_l = list(ord_btch_lens)
    ord_btch_lens_l = map(lambda k: int(k), ord_btch_lens_l)
    return tps_ip_seq_btch, ord_op_seq_btch, list(ord_btch_lens_l)


training_dataset = NtGenDataset('./mozart/train', mx_seq_ln=None)
training_datasetloader = data.DataLoader(training_dataset, batch_size=5,shuffle=True, drop_last=True)

X_train = next(iter(training_datasetloader))
X_train[0].shape

validation_dataset = NtGenDataset('./mozart/valid/', mx_seq_ln=None)
validation_datasetloader = data.DataLoader(validation_dataset, batch_size=3, shuffle=False, drop_last=False)

X_validation = next(iter(validation_datasetloader))
X_validation[0].shape

plt.figure(figsize=(10,7))
plt.title("Matrix representation of a Mozart composition")
plt.imshow(X_validation[0][0][:300].numpy().T);
plt.show()

