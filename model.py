import numpy


class RNN(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.activation = numpy.tanh
        self.W = numpy.random.uniform(-numpy.sqrt(3./(in_size+out_size)), numpy.sqrt(3./(in_size+out_size)), size=(out_size, in_size))
        self.U = numpy.random.uniform(-numpy.sqrt(1.5/out_size), numpy.sqrt(1.5/ out_size),size=(out_size, out_size))
        self.b = numpy.zeros((out_size,), dtype=numpy.float64)
        self.Wh = numpy.zeros_like(self.W)
        self.Uh = numpy.zeros_like(self.U)
        self.bh = numpy.zeros_like(self.b)

    def step(self, input, last_hidden):
        v = numpy.dot(self.W, input) + self.b + numpy.dot(self.U, last_hidden)
        return self.activation(v)

    '''
        return derivInput, derivHidden, derivW, derivU, derivB
    '''
    def back_step(self, input, last_hidden, hidden, dhidden_from_next, errors):
        dv = (errors +dhidden_from_next) * (numpy.ones(self.out_size) - hidden * hidden)
        di = numpy.dot(dv, self.W)
        dh = numpy.dot(dv, self.U)
        dW = numpy.outer(dv, input)
        dU = numpy.outer(dv, last_hidden)
        return di, dh, dW, dU, dv


def sigmoid(x):
    return numpy.ones_like(x)/(numpy.ones_like(x) + numpy.exp(-x))


class LSTM(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.Wf = numpy.random.uniform(-numpy.sqrt(6./(in_size + out_size * 2)),
                                       numpy.sqrt(6./(in_size + out_size * 2)), size= (out_size, in_size + out_size))
        self.Wi = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wo = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wc = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                   numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Bf = numpy.zeros(out_size)
        self.Bi = numpy.zeros(out_size)
        self.Bo = numpy.zeros(out_size)
        self.Bc = numpy.zeros(out_size)
        self.Wfh = numpy.zeros_like(self.Wf)
        self.Wih = numpy.zeros_like(self.Wi)
        self.Woh = numpy.zeros_like(self.Wo)
        self.Wch = numpy.zeros_like(self.Wc)

    def step(self, input, last_hidden, last_cell):
        real_input = numpy.concatenate((input, last_hidden))
        fg = sigmoid(numpy.dot(self.Wf, real_input) + self.Bf)
        ig = sigmoid(numpy.dot(self.Wi, real_input) + self.Bi)
        og = sigmoid(numpy.dot(self.Wo, real_input) + self.Bo)
        nc = numpy.tanh(numpy.dot(self.Wc, real_input) + self.Bc)
        cell = fg * last_cell + ig * nc
        tcell = numpy.tanh(cell)
        hidden = og * tcell
        return hidden, cell, fg, ig, og, nc, tcell

    def backward(self, input, last_hidden, last_cell, fg, ig, og, nc, tcell, errors_from_next_layer, dhidden_from_next, dcell_from_next):
        errors = errors_from_next_layer + dhidden_from_next
        d_og = errors * tcell
        d_tcell = errors * og
        d_cell = d_tcell * (numpy.ones_like(tcell) - tcell ** 2) + dcell_from_next

        d_fg = d_cell * last_cell
        d_last_cell = d_cell * fg
        d_ig = d_cell * nc
        d_nc = d_cell * ig

        d_fg = d_fg * fg *(numpy.ones_like(fg) - fg)
        d_ig = d_ig * ig *(numpy.ones_like(ig) - ig)
        d_og = d_og * og * (numpy.ones_like(og) - og)
        d_nc = d_nc * nc * (numpy.ones_like(nc) - nc)

        d_real_input = numpy.dot(d_fg, self.Wf) + numpy.dot(d_ig, self.Wi) + numpy.dot(d_og, self.Wo) + numpy.dot(d_nc, self.Wc)
        d_input = d_real_input[:self.in_size]
        d_hidden = d_real_input[self.in_size:]
        real_input = numpy.concatenate((input, last_hidden))
        dWf = numpy.outer(d_fg, real_input)
        dWi = numpy.outer(d_ig, real_input)
        dWo = numpy.outer(d_og, real_input)
        dWc = numpy.outer(d_nc, real_input)
        return d_input, d_hidden, d_last_cell, dWf, dWi, dWo, dWc, d_fg, d_ig, d_og, d_nc

class Linear(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.W = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size)), numpy.sqrt(6. / (in_size + out_size)),
                                 size=(out_size,in_size))
        self.b = numpy.zeros((out_size,), dtype=numpy.float64)
        self.Wh = numpy.zeros_like(self.W)
        self.bh = numpy.zeros_like(self.b)

    def step(self, input):
        v = numpy.dot(self.W, input)
        return v + self.b

    def back_step(self, input, errors):
        assert(errors.shape == self.b.shape)
        dW = numpy.outer(errors, input)
        di = numpy.dot(errors, self.W)
        return di, dW, errors


class Nonlinear(object):
    def __init__(self, size):
        self.size = size

    def step(self, input):
        return numpy.tanh(input)

    def back_step(self, output, errors):
        return errors*(numpy.ones(self.size) - output * output)


class LookupTable(object):
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.table={}

    def add_token(self, token, embed=None):
        if embed is None:
            embed = numpy.random.uniform(-0.1, 0.1, size=self.embed_size)
        self.table[token]=embed

    def has_token(self, token):
        return token in self.table

    def step(self, token):
        return self.table[token]


class Softmax(object):
    def softmax(self, x):
        xmin = numpy.min(x)
        x -= xmin
        y = numpy.exp(-x)
        return y / y.sum()

    def __init__(self, size):
        self.size = size

    def step(self, input):
        return self.softmax(input)

    def get_gradients(self, output, target):
        v = numpy.zeros(self.size)
        v[target] = 1
        return v - output


class Model(object):
    def __init__(self):
        self.lookup = LookupTable(50)
        self.L1 = Linear(250, 300)
        self.N1 = Nonlinear(300)
        self.lstm = LSTM(300,50)
        self.L3 = Linear(50, 4)
        self.softmax = Softmax(4)
        self.target_map = {}
        self.target_id_to_tag = {}

    def forward(self, sentence):
        window_c = []
        window_vectors_c = []
        lin1_c = []
        non1_c = []
        for i in range(2, len(sentence)-2):
            window = sentence[i-2: i+3]
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
        lstm_tcell_c =[]
        for i in range(len(non1_c)):
            if i == 0:
                last_hidden = numpy.zeros(self.lstm.out_size)
                last_cell = numpy.zeros(self.lstm.out_size)
            else:
                last_hidden = lstm_h_c[i-1]
                last_cell = lstm_c_c[i-1]
            l_hidden, l_cell, l_fg, l_ig, l_og, l_nc, l_tcell = self.lstm.step(non1_c[i], last_hidden, last_cell)
            lstm_h_c.append(l_hidden)
            lstm_c_c.append(l_cell)
            lstm_fg_c.append(l_fg)
            lstm_ig_c.append(l_ig)
            lstm_og_c.append(l_og)
            lstm_nc_c.append(l_nc)
            lstm_tcell_c.append(l_tcell)

        lin3_c = [self.L3.step(v) for v in lstm_h_c]
        softmax_c = [self.softmax.step(v) for v in lin3_c]
        return (window_c, window_vectors_c, lin1_c, non1_c,
                lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,
                lin3_c, softmax_c)

    def backward(self, window_vectors_c, lin1_c, non1_c,
                 lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,
                 lin3_c, softmax_c, targets):
        assert len(softmax_c) == len(targets)
        derrors = [self.softmax.get_gradients(softmax_c[i], self.target_map[targets[i]]) for i in range(len(targets))]
        dlin3W_s = numpy.zeros_like(self.L3.W)
        dlin3b_s = numpy.zeros_like(self.L3.b)
        drnn_c = []
        for i in range(len(derrors)):
            drnn, dlin3W, dlin3b = self.L3.back_step(lstm_h_c[i], derrors[i])
            drnn_c.append(drnn)
            dlin3W_s += dlin3W
            dlin3b_s += dlin3b

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
        dnon1_c = []
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
            dnon1_c.insert(0, d_input)
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

        return dEmbed_c, dlin1W_s, dlin1B_s, drnnWf_s, drnnWi_s, drnnWo_s, drnnWc_s, drnnBf_s, drnnBi_s, drnnBo_s, drnnBc_s, dlin3W_s, dlin3b_s
