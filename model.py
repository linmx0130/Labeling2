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
        self.L2 = Linear(300,4)
        self.rnn = RNN(4,4)
        self.L3 = Linear(4, 4)
        self.softmax = Softmax(4)
        self.target_map = {}

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
        lin2_c = [self.L2.step(v) for v in non1_c]
        rnn_c = []
        for i in range(len(lin2_c)):
            if i == 0:
                last_hidden = numpy.zeros(self.rnn.out_size)
            else:
                last_hidden = rnn_c[i-1]
            rnn_out = self.rnn.step(lin2_c[i], last_hidden)
            rnn_c.append(rnn_out)

        lin3_c = [self.L3.step(v) for v in rnn_c]
        softmax_c = [self.softmax.step(v) for v in lin3_c]
        return (window_c, window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c)

    def backward(self, window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c, targets):
        assert len(softmax_c) == len(targets)
        derrors = [self.softmax.get_gradients(softmax_c[i], self.target_map[targets[i]]) for i in range(len(targets))]
        dlin3W_s = numpy.zeros_like(self.L3.W)
        dlin3b_s = numpy.zeros_like(self.L3.b)
        drnn_c = []
        for i in range(len(derrors)):
            drnn, dlin3W, dlin3b = self.L3.back_step(lin2_c[i], derrors[i])
            drnn_c.append(drnn)
            dlin3W_s += dlin3W
            dlin3b_s += dlin3b

        drnnW_s = numpy.zeros_like(self.rnn.W)
        drnnU_s = numpy.zeros_like(self.rnn.U)
        drnnB_s = numpy.zeros_like(self.rnn.b)
        dhidden_from_next = numpy.zeros(self.rnn.out_size)
        dlin2_c = []
        for i in range(len(drnn_c) - 1, -1, -1):
            if i == 0:
                last_hidden = numpy.zeros(self.rnn.out_size)
            else:
                last_hidden = rnn_c[i - 1]
            di, dh, dW, dU, dB = self.rnn.back_step(lin2_c[i], last_hidden, rnn_c[i], dhidden_from_next, drnn_c[i])
            dlin2_c.insert(0, di)
            dhidden_from_next = dh
            drnnW_s += dW
            drnnU_s += dU
            drnnB_s += dB

        dnon1_c = []
        dlin2W_s = numpy.zeros_like(self.L2.W)
        dlin2b_s = numpy.zeros_like(self.L2.b)
        for i in range(len(derrors)):
            dnon_1, dlin2W, dlin2b = self.L2.back_step(non1_c[i], dlin2_c[i])
            dnon1_c.append(dnon_1)
            dlin2W_s += dlin2W
            dlin2b_s += dlin2b

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

        return dEmbed_c, dlin1W_s, dlin1B_s, dlin2W_s, dlin2b_s, drnnW_s, drnnU_s, drnnB_s, dlin3W_s, dlin3b_s
