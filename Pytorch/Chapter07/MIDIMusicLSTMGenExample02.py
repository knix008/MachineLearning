import sys

sys.dont_write_bytecode = True

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import skimage.io as io
from matplotlib import pyplot as plt

from MIDI import midiwrite
from MIDIDataLoader import NtGenDataset, pos_proc_seq


class MusicLSTM(nn.Module):
    def __init__(self, ip_sz, hd_sz, n_cls, lyrs=2):
        super(MusicLSTM, self).__init__()
        self.ip_sz = ip_sz
        self.hd_sz = hd_sz
        self.n_cls = n_cls
        self.lyrs = lyrs
        self.nts_enc = nn.Linear(in_features=ip_sz, out_features=hd_sz)
        self.bn_layer = nn.BatchNorm1d(hd_sz)
        self.lstm_layer = nn.LSTM(hd_sz, hd_sz, lyrs)
        self.fc_layer = nn.Linear(hd_sz, n_cls)

    def forward(self, ip_seqs, ip_seqs_len, hd=None):
        nts_enc = self.nts_enc(ip_seqs)
        nts_enc_rol = nts_enc.permute(1, 2, 0).contiguous()
        nts_enc_nrm = self.bn_layer(nts_enc_rol)
        nts_enc_nrm_drp = nn.Dropout(0.25)(nts_enc_nrm)
        nts_enc_ful = nts_enc_nrm_drp.permute(2, 0, 1)

        pkd = torch.nn.utils.rnn.pack_padded_sequence(nts_enc_ful, ip_seqs_len)
        op, hd = self.lstm_layer(pkd, hd)

        op, op_l = torch.nn.utils.rnn.pad_packed_sequence(op)

        op_nrm = self.bn_layer(op.permute(1, 2, 0).contiguous())
        op_nrm_drp = nn.Dropout(0.1)(op_nrm)
        lgts = self.fc_layer(op_nrm_drp.permute(2, 0, 1))
        lgts = lgts.transpose(0, 1).contiguous()

        rev_lgts = 1 - lgts

        zero_one_lgts = torch.stack((lgts, rev_lgts), dim=3).contiguous()
        flt_lgts = zero_one_lgts.view(-1, 2)
        return flt_lgts, hd


def load_data_from_midi_files():
    print("> Loading MIDI data from Nottingham dataset")
    # This function is a placeholder for loading MIDI data.
    # The actual implementation would depend on the specific dataset and its structure.
    pass
    training_dataset = NtGenDataset("./data/Nottingham/train", mx_seq_ln=None)
    training_datasetloader = data.DataLoader(
        training_dataset, batch_size=5, shuffle=True, drop_last=True
    )

    validation_dataset = NtGenDataset("./data/Nottingham/valid/", mx_seq_ln=None)
    validation_datasetloader = data.DataLoader(
        validation_dataset, batch_size=3, shuffle=False, drop_last=False
    )

    return training_datasetloader, validation_datasetloader


def visualize_validation_data(validation_datasetloader):
    print("> Visualizing validation data")
    X_validation = next(iter(validation_datasetloader))
    plt.figure(figsize=(10, 7))
    plt.title("Matrix representation of a Nottingham composition")
    plt.imshow(X_validation[0][0][1000:].numpy().T)
    plt.show()


def lstm_model_training(
    lstm_model,
    training_datasetloader,
    validation_datasetloader,
    loss_func,
    evaluate_model,
    lr,
    ep=10,
    val_loss_best=float("inf"),
):
    print("> Training Music LSTM model")
    list_of_losses = []
    list_of_val_losses = []
    model_params = lstm_model.parameters()
    opt = torch.optim.Adam(model_params, lr=lr)
    grad_clip = 1.0
    for curr_ep in range(ep):
        lstm_model.train()
        loss_ep = []
        for batch in training_datasetloader:
            post_proc_b = pos_proc_seq(batch)
            ip_seq_b, op_seq_b, seq_l = post_proc_b
            op_seq_b_v = Variable(op_seq_b.contiguous().view(-1).cpu())
            ip_seq_b_v = Variable(ip_seq_b.cpu())
            opt.zero_grad()
            lgts, _ = lstm_model(ip_seq_b_v, seq_l)
            loss = loss_func(lgts, op_seq_b_v)
            list_of_losses.append(loss.item())
            loss_ep.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), grad_clip)
            opt.step()
            print(f"EPOCH : {curr_ep} , Batch Loss = {loss.item()}")

        tr_ep_cur = sum(loss_ep) / len(training_datasetloader)
        print(f"ep {curr_ep} , train loss = {tr_ep_cur}")

        vl_ep_cur = evaluate_model(lstm_model, loss_func, validation_datasetloader)
        print(f"ep {curr_ep} , val loss = {vl_ep_cur}\n")

        list_of_val_losses.append(vl_ep_cur)

        if vl_ep_cur < val_loss_best:
            torch.save(lstm_model.state_dict(), "best_model.pth")
            val_loss_best = vl_ep_cur
    return lstm_model


def evaluate_model(lstm_model, loss_func, validation_datasetloader):
    print("> Evaluating model on validation dataset")
    lstm_model.eval()
    vl_loss_full = 0.0
    seq_len = 0.0

    for batch in validation_datasetloader:
        post_proc_b = pos_proc_seq(batch)
        ip_seq_b, op_seq_b, seq_l = post_proc_b
        op_seq_b_v = Variable(op_seq_b.contiguous().view(-1).cpu())
        ip_seq_b_v = Variable(ip_seq_b.cpu())
        lgts, _ = lstm_model(ip_seq_b_v, seq_l)
        loss = loss_func(lgts, op_seq_b_v)
        vl_loss_full += loss.item()
        seq_len += sum(seq_l)

    return vl_loss_full / (seq_len * 88)


def generate_music(lstm_model, ln=100, tmp=1, seq_st=None):
    print("> Generating music sequence")
    if seq_st is None:
        seq_ip_cur = torch.zeros(1, 1, 88)
        seq_ip_cur[0, 0, 40] = 1
        seq_ip_cur[0, 0, 50] = 0
        seq_ip_cur[0, 0, 56] = 0
        seq_ip_cur = Variable(seq_ip_cur.cpu())
    else:
        seq_ip_cur = seq_st

    op_seq = [seq_ip_cur.data.squeeze(1)]
    hd = None

    for i in range(ln):
        op, hd = lstm_model(seq_ip_cur, [1], hd)
        probs = nn.functional.softmax(op.div(tmp), dim=1)
        seq_ip_cur = (
            torch.multinomial(probs.data, 1).squeeze().unsqueeze(0).unsqueeze(1)
        )
        seq_ip_cur = Variable(seq_ip_cur.float())
        op_seq.append(seq_ip_cur.data.squeeze(1))

    gen_seq = torch.cat(op_seq, dim=0).cpu().numpy()
    return gen_seq


def main():
    print("> Starting Music LSTM Generation Example")
    training_datasetloader, validation_datasetloader = load_data_from_midi_files()
    visualize_validation_data(validation_datasetloader)
    loss_func = nn.CrossEntropyLoss().cpu()
    lstm_model = MusicLSTM(ip_sz=88, hd_sz=512, n_cls=88).cpu()
    lstm_model = lstm_model_training(
        lstm_model, training_datasetloader, validation_datasetloader, loss_func, evaluate_model, lr=0.01, ep=1
    )
    seq = generate_music(lstm_model, ln=100, tmp=0.8, seq_st=None).transpose()
    io.imshow(seq)
    midiwrite("generated_music.mid", seq.transpose(), dtm=0.25)


if __name__ == "__main__":
    print("> Running Music LSTM Generation Example")
    main()
