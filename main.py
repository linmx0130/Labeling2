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
        last_sentence.append("#1")
        last_sentence.append("#2")
        sentences.append(last_sentence)
        targets.append(last_target)
    return sentences, targets


def sgdUpdate(target, dvalue, learn_rate = 0.01, L2Reg = 0.0001):
    target -= learn_rate*(dvalue + target*L2Reg)


def adagradUpdate(target, history, dvalue, learn_rate = 0.01, L2Reg = 0.0001):
    dx = dvalue + target * L2Reg
    history += dx **2
    target -= learn_rate * dx / (numpy.sqrt(history) + 1e-7)


def train_forward(m, sentence, target):
    window_c, window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c = m.forward(sentence)
    dEmbed_c, dlin1W, dlin1B, dlin2W, dlin2b, drnnW, drnnU, drnnB, dlin3W, dlin3b= m.backward(window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c, target)
    adagradUpdate(m.L1.W, m.L1.Wh, dlin1W)
    sgdUpdate(m.L1.b, dlin1B)
    adagradUpdate(m.L2.W, m.L2.Wh, dlin2W)
    sgdUpdate(m.L2.b, dlin2b)
    adagradUpdate(m.L3.W, m.L3.Wh, dlin3W)
    sgdUpdate(m.L3.b, dlin3b)
    adagradUpdate(m.rnn.W, m.rnn.Wh, drnnW)
    adagradUpdate(m.rnn.U, m.rnn.Uh, drnnU)
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


def predict_with_model(m, sentence):
    window_c, window_vectors_c, lin1_c, non1_c, lin2_c, rnn_c, lin3_c, softmax_c = m.forward(sentence)
    target = []
    for v in softmax_c:
        ans_id = numpy.argmax(v)
        target.append(m.target_id_to_tag[ans_id])
    return target


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
    numpy.seterr(invalid="raise")
    m = Model()
    sentences, targets = load_data("ctb_seg_train.utf8")
    init_lookup_table(m,sentences)
    init_targets_id(m, targets)
    data_count = len(sentences)
    iter_time = 0
    not_stop = True
    while (not_stop):
        iter_time += 1
        correct_rate = 0
        for i in range(data_count):
            correct_rate += train_forward(m, sentences[i], targets[i])
        correct_rate /= data_count
        print("Iter %d correct rate=%f"%(iter_time, correct_rate))
        if correct_rate>0.99 or iter_time>30:
            not_stop = False
    test_model(m)
    return m


def test_model(m):
    sentences, targets = load_data("ctb_seg_test.utf8")
    total_correct = 0.
    total_tag = 0
    for i in range(len(sentences)):
        m_output = predict_with_model(m, sentences[i])
        total_tag += len(m_output)
        for j in range(len(m_output)):
            if m_output[j] == targets[i][j]:
               total_correct += 1
    print("Test: %f"%(total_correct/total_tag))


if __name__=="__main__":
    m = train_model()