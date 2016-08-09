import numpy
from module import *


class Model(object):
    def __init__(self):
        self.lookup = LookupTable(50)
        self.L1 = Linear(250, 300)
        self.N1 = Nonlinear(300)
        self.L2 = Linear(300, 50)
        self.lstm = LSTM(300, 50)
        self.L3 = Linear(50, 4)
        self.softmax = Softmax(4)
        self.target_map = {}
        self.target_id_to_tag = {}

    def forward(self, sentence):
        window_c = []
        window_vectors_c = []
        lin1_c = []
        non1_c = []
        for i in range(2, len(sentence) - 2):
            window = sentence[i - 2: i + 3]
            window_c.append(window)
            window_vectors = [self.lookup.step(w) for w in window]
            window_vectors_c.append(window_vectors)
            lin1 = self.L1.step(numpy.concatenate(window_vectors))
            non1 = self.N1.step(lin1)
            lin1_c.append(lin1)
            non1_c.append(non1)
        lstm_h_c = []
        lstm_c_c = []
        lstm_fg_c = []
        lstm_ig_c = []
        lstm_og_c = []
        lstm_nc_c = []
        lstm_tcell_c = []
        for i in range(len(non1_c)):
            if i == 0:
                last_hidden = numpy.zeros(self.lstm.out_size)
                last_cell = numpy.zeros(self.lstm.out_size)
            else:
                last_hidden = lstm_h_c[i - 1]
                last_cell = lstm_c_c[i - 1]
            l_hidden, l_cell, l_fg, l_ig, l_og, l_nc, l_tcell = self.lstm.step(non1_c[i], last_hidden, last_cell)
            lstm_h_c.append(l_hidden)
            lstm_c_c.append(l_cell)
            lstm_fg_c.append(l_fg)
            lstm_ig_c.append(l_ig)
            lstm_og_c.append(l_og)
            lstm_nc_c.append(l_nc)
            lstm_tcell_c.append(l_tcell)
        lin2_c = [self.L2.step(v) for v in non1_c]
        after_highway = []
        lin3_c = []
        for i in range(len(lstm_h_c)):
            v = numpy.tanh(lstm_h_c[i] + lin2_c[i])
            after_highway.append(v)
            lin3_c.append(self.L3.step(v))
        softmax_c = [self.softmax.step(v) for v in lin3_c]
        return (window_c, window_vectors_c, lin1_c, non1_c, lin2_c,
                lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,
                after_highway, lin3_c, softmax_c)

    def backward(self, window_vectors_c, lin1_c, non1_c, lin2_c,
                 lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,
                 after_highway, lin3_c, softmax_c, targets):
        assert len(softmax_c) == len(targets)
        derrors = [self.softmax.get_gradients(softmax_c[i], self.target_map[targets[i]]) for i in range(len(targets))]
        dlin3W_s = numpy.zeros_like(self.L3.W)
        dlin3b_s = numpy.zeros_like(self.L3.b)
        dhighway_c = []
        for i in range(len(derrors)):
            dhighway, dlin3W, dlin3b = self.L3.back_step(lstm_h_c[i], derrors[i])
            dhighway *= 1 - after_highway[i] ** 2
            dhighway_c.append(dhighway)
            dlin3W_s += dlin3W
            dlin3b_s += dlin3b

        drnn_c = dhighway_c
        dlin2_c = dhighway_c
        dnon1_c = []
        dlin2W_s = numpy.zeros_like(self.L2.W)
        dlin2b_s = numpy.zeros_like(self.L2.b)
        for i in range(len(dlin2_c)):
            dnon1, dlin2W, dlin2b = self.L2.back_step(non1_c[i], dlin2_c[i])
            dnon1_c.append(dnon1)
            dlin2W_s += dlin2W
            dlin2b_s += dlin2b
        drnnWf_s = numpy.zeros_like(self.lstm.Wf)
        drnnWi_s = numpy.zeros_like(self.lstm.Wi)
        drnnWo_s = numpy.zeros_like(self.lstm.Wo)
        drnnWc_s = numpy.zeros_like(self.lstm.Wc)
        drnnBf_s = numpy.zeros_like(self.lstm.Bf)
        drnnBi_s = numpy.zeros_like(self.lstm.Bi)
        drnnBo_s = numpy.zeros_like(self.lstm.Bi)
        drnnBc_s = numpy.zeros_like(self.lstm.Bc)
        dhidden_from_next = numpy.zeros(self.lstm.out_size)
        dcell_from_next = numpy.zeros(self.lstm.out_size)
        for i in range(len(drnn_c) - 1, -1, -1):
            if i == 0:
                last_hidden = numpy.zeros(self.lstm.out_size)
                last_cell = numpy.zeros(self.lstm.out_size)
            else:
                last_hidden = lstm_h_c[i - 1]
                last_cell = lstm_c_c[i - 1]
            d_input, d_hidden, d_last_cell, dWf, dWi, dWo, dWc, dBf, dBi, dBo, dBc = \
                self.lstm.backward(non1_c[i], last_hidden, last_cell,
                                   lstm_fg_c[i], lstm_ig_c[i], lstm_og_c[i], lstm_nc_c[i], lstm_tcell_c[i],
                                   drnn_c[i], dhidden_from_next, dcell_from_next)
            dnon1_c[i] += d_input
            dhidden_from_next = d_hidden
            dcell_from_next = d_last_cell
            drnnWf_s += dWf
            drnnWi_s += dWi
            drnnWo_s += dWo
            drnnWc_s += dWc
            drnnBf_s += dBf
            drnnBi_s += dBi
            drnnBo_s += dBo
            drnnBc_s += dBc

        dlin1_c = []
        for i in range(len(dnon1_c)):
            dlin1_c.append(self.N1.back_step(non1_c[i], dnon1_c[i]))

        dlin1W_s = numpy.zeros_like(self.L1.W)
        dlin1B_s = numpy.zeros_like(self.L1.b)
        dEmbed_c = []
        for i in range(len(dlin1_c)):
            di, dW, dB = self.L1.back_step(window_vectors_c[i], dlin1_c[i])
            dEmbed_c.append(di)
            dlin1W_s += dW
            dlin1B_s += dB

        return dEmbed_c, dlin1W_s, dlin1B_s, dlin2W_s, dlin2b_s, drnnWf_s, drnnWi_s, drnnWo_s, drnnWc_s, drnnBf_s, drnnBi_s, drnnBo_s, drnnBc_s, dlin3W_s, dlin3b_s
