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


def adagradUpdate(target, history, dvalue, learn_rate = 0.02, L2Reg = 0.0001):
    dx = dvalue + target * L2Reg
    history += dx **2
    target -= learn_rate * dx / (numpy.sqrt(history) + 1e-7)


def rmspropUpdate(target, history, dvalue, learn_rate = 0.005, L2Reg = 0.0001, decay_rate = 0.9):
    dx = dvalue + target * L2Reg
    history *= decay_rate
    history += (1-decay_rate) * dx **2
    target -= learn_rate * dx / (numpy.sqrt(history) + 1e-7)


def train_forward(m, sentence, target):
    window_c, window_vectors_c, lin1_c, non1_c, lin2_c,\
        lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,\
        after_highway, lin3_c, softmax_c = m.forward(sentence)
    dEmbed_c, dlin1W, dlin1B, dlin2W, dlin2b, drnnWf, drnnWi, drnnWo, drnnWc, drnnBf, drnnBi, drnnBo, drnnBc, dlin3W, dlin3b= \
        m.backward(window_vectors_c, lin1_c, non1_c, lin2_c, lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,after_highway, lin3_c, softmax_c, target)
    adagradUpdate(m.L1.W, m.L1.Wh, dlin1W)
    adagradUpdate(m.L1.b, m.L1.bh, dlin1B)
    adagradUpdate(m.L3.W, m.L3.Wh, dlin3W)
    adagradUpdate(m.L3.b, m.L3.bh, dlin3b)
    adagradUpdate(m.L2.W, m.L2.Wh, dlin2W)
    adagradUpdate(m.L2.b, m.L2.bh, dlin2b)
    adagradUpdate(m.lstm.Wf, m.lstm.Wfh, drnnWf)
    adagradUpdate(m.lstm.Wi, m.lstm.Wih, drnnWi)
    adagradUpdate(m.lstm.Wo, m.lstm.Woh, drnnWo)
    adagradUpdate(m.lstm.Wc, m.lstm.Wch, drnnWc)
    adagradUpdate(m.lstm.Bf, m.lstm.Bfh, drnnBf)
    adagradUpdate(m.lstm.Bi, m.lstm.Bih, drnnBi)
    adagradUpdate(m.lstm.Bo, m.lstm.Boh, drnnBo)
    adagradUpdate(m.lstm.Bc, m.lstm.Bch, drnnBc)

    assert len(dEmbed_c) == len(window_c)
    for i in range(len(window_c)):
        dEmbed = numpy.reshape(dEmbed_c[i], newshape=(5, m.lookup.embed_size))
        window = window_c[i]
        for j in range(len(window)):
            sgdUpdate(m.lookup.table[window[j]], dEmbed[j])

    correct = 0.
    for i in range(len(softmax_c)):
        if numpy.argmax(softmax_c[i]) == m.target_map[target[i]]:
            correct += 1
    return correct/len(softmax_c)


def predict_with_model(m, sentence):
    window_c, window_vectors_c, lin1_c, non1_c, dlin2_c, \
        lstm_h_c, lstm_c_c, lstm_fg_c, lstm_ig_c, lstm_og_c, lstm_nc_c, lstm_tcell_c,\
        after_highway, lin3_c, softmax_c = m.forward(sentence)
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
                m.target_id_to_tag[len(m.target_map) - 1] = tag


def train_model():
    numpy.seterr(invalid="raise")
    m = Model()
    sentences, targets = load_data("predict_test_train.utf8")
    testset_sentences, testset_targets = load_data("predict_test_train.utf8")
    init_lookup_table(m,sentences)
    init_lookup_table(m, testset_sentences)
    init_targets_id(m, targets)
    data_count = len(sentences)
    iter_time = 0
    not_stop = True
    highest_rate = 0
    while (not_stop):
        iter_time += 1
        correct_rate = 0
        for counter in range(data_count):
            i = numpy.random.randint(0, data_count)
            correct_rate += train_forward(m, sentences[i], targets[i])
        correct_rate /= data_count
        print("Iter %d correct rate=%f"%(iter_time, correct_rate))
        if correct_rate>0.99 and iter_time>30:
            not_stop = False
        current_value = test_model(m, testset_sentences, testset_targets)
        if current_value > highest_rate:
            highest_rate = current_value
            predict_tags(m, testset_sentences, "test_output%d.utf8"%iter_time)
            
    return m


def test_model(m, sentences, targets):
    total_correct = 0.
    total_tag = 0
    for i in range(len(sentences)):
        m_output = predict_with_model(m, sentences[i])
        total_tag += len(m_output)
        for j in range(len(m_output)):
            if m_output[j] == targets[i][j]:
               total_correct += 1
    score = (total_correct/total_tag)
    print("Test: %f"%score)
    return score


def predict_tags(m, sentences, output_file):
    total_correct = 0.
    total_tag = 0
    fout = open(output_file, mode='w', encoding='utf-8')
    for i in range(len(sentences)):
        m_output = predict_with_model(m, sentences[i])
        buffer_sen = []
        for j in range(len(m_output)):
            buffer_sen.append(sentences[i][j+2]+" "+m_output[j])
        for item in buffer_sen:
            print(item, file=fout)
        print("", file=fout)
    fout.close()


if __name__=="__main__":
    m = train_model()
