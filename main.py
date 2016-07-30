from model import *

def load_data(filename):
    f = open(filename)
    sentences = []
    targets = []
    last_sentence = ["#-2", "#-1"]
    last_target = []
    for l in f.readlines():
        l = l.strip()
        if len(l) == 0:
            last_sentence.append("#1")
            last_sentence.append("#2")
            sentences.append(last_sentence)
            targets.append(last_target)
            last_sentence = ["#-2", "#-1"]
            last_target = []
            continue
        sp = l.split(" ")
        if len(sp)==1:
            print("Error: l=")
            print(l)
        token = sp[0]
        tag = sp[1]
        last_sentence.append(token)
        last_target.append(tag)
    if len(last_target)!=0:
        sentences.append(last_sentence)
        targets.append(last_target)
    return sentences, targets


def sgdUpdate(target, dvalue, learn_rate = 0.01, L2Reg=0.0001):
    target -= learn_rate*(dvalue + target*L2Reg)


def train_forward(m, sentence, target):
    window_c, window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c = m.forward(sentence)
    dEmbed_c, dlin1W, dlin1B, dlin2W, dlin2b, drnnW, drnnU, drnnB, dlin3W, dlin3b= m.backward(window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c, target)
    sgdUpdate(m.L1.W, dlin1W)
    sgdUpdate(m.L1.b, dlin1B)
    sgdUpdate(m.L2.W, dlin2W)
    sgdUpdate(m.L2.b, dlin2b)
    sgdUpdate(m.L3.W, dlin3W)
    sgdUpdate(m.L3.b, dlin3b)
    sgdUpdate(m.rnn.W, drnnW)
    sgdUpdate(m.rnn.U, drnnU)
    sgdUpdate(m.rnn.b, drnnB)

    assert len(dEmbed_c) == len(window_c)
    for i in range(len(window_c)):
        dEmbed = numpy.reshape(dEmbed_c[i], newshape=(5, 50))
        window = window_c[i]
        for j in range(len(window)):
            sgdUpdate(m.lookup.table[window[j]], dEmbed[j])

    correct = 0.
    for i in range(len(softmax_c)):
        if numpy.argmax(softmax_c[i]) == m.target_map[target[i]]:
            correct += 1
    return correct/len(softmax_c)


def init_lookup_table(m, sentences):
    for s in sentences:
        for token in s:
            if not m.lookup.has_token(token):
                m.lookup.add_token(token)


def init_targets_id(m, targets):
    for s in targets:
        for tag in s:
            if tag not in m.target_map:
                m.target_map[tag] = len(m.target_map)


def train_model():
    m = Model()
    sentences, targets = load_data("predict_test_train.utf8")
    init_lookup_table(m,sentences)
    init_targets_id(m, targets)
    data_count = len(sentences)
    for iter_time in range(1000):
        correct_rate = 0
        for i in range(data_count):
            correct_rate += train_forward(m, sentences[i], targets[i])

        correct_rate /= data_count
        print("Iter %d correct rate=%f"%(iter_time, correct_rate))

if __name__=="__main__":
    train_model()