import numpy
class RNN(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.activation = numpy.tanh
        self.W = numpy.random.uniform(-numpy.sqrt(3. / (in_size + out_size)), numpy.sqrt(3. / (in_size + out_size)),
                                      size=(out_size, in_size))
        self.U = numpy.random.uniform(-numpy.sqrt(1.5 / out_size), numpy.sqrt(1.5 / out_size),
                                      size=(out_size, out_size))
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
        dv = (errors + dhidden_from_next) * (numpy.ones(self.out_size) - hidden * hidden)
        di = numpy.dot(dv, self.W)
        dh = numpy.dot(dv, self.U)
        dW = numpy.outer(dv, input)
        dU = numpy.outer(dv, last_hidden)
        return di, dh, dW, dU, dv


def sigmoid(x):
    return numpy.ones_like(x) / (numpy.ones_like(x) + numpy.exp(-x))


class LSTM(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.Wf = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wi = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wo = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wc = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Bf = numpy.zeros(out_size) - 1
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

    def backward(self, input, last_hidden, last_cell, fg, ig, og, nc, tcell, errors_from_next_layer, dhidden_from_next,
                 dcell_from_next):
        errors = errors_from_next_layer + dhidden_from_next
        d_og = errors * tcell
        d_tcell = errors * og
        d_cell = d_tcell * (numpy.ones_like(tcell) - tcell ** 2) + dcell_from_next

        d_fg = d_cell * last_cell
        d_last_cell = d_cell * fg
        d_ig = d_cell * nc
        d_nc = d_cell * ig

        d_fg = d_fg * fg * (numpy.ones_like(fg) - fg)
        d_ig = d_ig * ig * (numpy.ones_like(ig) - ig)
        d_og = d_og * og * (numpy.ones_like(og) - og)
        d_nc = d_nc * nc * (numpy.ones_like(nc) - nc)

        d_real_input = numpy.dot(d_fg, self.Wf) + numpy.dot(d_ig, self.Wi) + numpy.dot(d_og, self.Wo) + numpy.dot(d_nc,
                                                                                                                  self.Wc)
        d_input = d_real_input[:self.in_size]
        d_hidden = d_real_input[self.in_size:]
        real_input = numpy.concatenate((input, last_hidden))
        dWf = numpy.outer(d_fg, real_input)
        dWi = numpy.outer(d_ig, real_input)
        dWo = numpy.outer(d_og, real_input)
        dWc = numpy.outer(d_nc, real_input)
        return d_input, d_hidden, d_last_cell, dWf, dWi, dWo, dWc, d_fg, d_ig, d_og, d_nc


class GRU(object):
    def __init__(self, in_size, out_size):
        self.Wr = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wz = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Wu = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size * 2)),
                                       numpy.sqrt(6. / (in_size + out_size * 2)), size=(out_size, in_size + out_size))
        self.Br = numpy.zeros(out_size)
        self.Bz = numpy.zeros(out_size)
        self.Bu = numpy.zeros(out_size)
        self.Wrh = (numpy.zeros_like(self.Wr), numpy.zeros_like(self.Wr))
        self.Wzh = (numpy.zeros_like(self.Wz), numpy.zeros_like(self.Wz))
        self.Wuh = (numpy.zeros_like(self.Wu), numpy.zeros_like(self.Wu))

    def step(self, input, last_hidden):
        v = numpy.concatenate((input,last_hidden))
        r = sigmoid(numpy.dot(self.Wr, v) + self.Br)
        z = sigmoid(numpy.dot(self.Wz, v) + self.Bz)
        after_reset = numpy.concatenate((input,last_hidden * r))
        nh = numpy.tanh(numpy.dot(self.Wu, after_reset) + self.Bu)
        h = last_hidden + z * (nh - last_hidden)
        return h, v, r, z, after_reset, nh

    def backward(self, input, last_hidden, hidden, real_input, r, z, after_reset, nh, errors_from_target, errors_from_next ):
        #TODO
        pass


class Linear(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.W = numpy.random.uniform(-numpy.sqrt(6. / (in_size + out_size)), numpy.sqrt(6. / (in_size + out_size)),
                                      size=(out_size, in_size))
        self.b = numpy.zeros((out_size,), dtype=numpy.float64)
        self.Wh = numpy.zeros_like(self.W)
        self.bh = numpy.zeros_like(self.b)

    def step(self, input):
        v = numpy.dot(self.W, input)
        return v + self.b

    def back_step(self, input, errors):
        assert (errors.shape == self.b.shape)
        dW = numpy.outer(errors, input)
        di = numpy.dot(errors, self.W)
        return di, dW, errors


class Nonlinear(object):
    def __init__(self, size):
        self.size = size

    def step(self, input):
        return numpy.tanh(input)

    def back_step(self, output, errors):
        return errors * (numpy.ones(self.size) - output * output)


class LookupTable(object):
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.table = {}

    def add_token(self, token, embed=None):
        if embed is None:
            embed = numpy.random.uniform(-0.1, 0.1, size=self.embed_size)
        self.table[token] = embed

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