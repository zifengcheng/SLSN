import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def acc_prf(pred_y, true_y, mask, average='binary'):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(len(mask[i])):
            if mask[i][j] == 1:
                tmp1.append(pred_y[i][j])
                tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1


def load_pair(input_file, max_doc_len=75, max_sen_len=45):
    print('load data_file: {}'.format(input_file))
    pair_id_all = []
    inputFile = open('data_combine/' + input_file)
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # give true pair a index
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')

    return pair_id_all

def load_index(input_file, max_doc_len=75, max_sen_len=45):
    index = np.zeros((2110,76))
    inputFile = open( input_file)
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        for p in pairs:
            index[doc_id][p[1]] = p[0]


        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
    return index


def prf_2nd_step(pair_id_all, pred_y):
    s1, s3 = set(pair_id_all), set(pred_y)
    # print(s3)
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def emotion_step(pair_id_all, pred_y):
    new_cause, y1_cause = [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y:
        y1_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    s1, s2 = set(new_cause), set(y1_cause)
    acc_num = len(s1 & s2)
    p, r = acc_num / (len(s2) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def cause_step(pair_id_all, pred_y):
    new_cause, y1_cause = [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + i % 100)
    for i in pred_y:
        y1_cause.append(int(i / 10000) * 10000 + i % 100)
    s1, s2 = set(new_cause), set(y1_cause)
    acc_num = len(s1 & s2)
    p, r = acc_num / (len(s2) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def emotion_union_step(pair_id_all, pred_y1, pred_y2):
    new_cause, y1_cause, y2_cause = [], [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y1:
        y1_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y2:
        y2_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    s1, s2, s3 = set(new_cause), set(y1_cause), set(y2_cause)
    s3 = s2 | s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def emotion_and_step(pair_id_all, pred_y1, pred_y2):
    new_cause, y1_cause, y2_cause = [], [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y1:
        y1_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y2:
        y2_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    s1, s2, s3 = set(new_cause), set(y1_cause), set(y2_cause)
    s3 = s2 & s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def cause_and_step(pair_id_all, pred_y1, pred_y2):
    new_cause ,y1_cause,y2_cause= [],[],[]
    for i in pair_id_all:
        new_cause.append(int(i/10000)*10000+ i % 100)
    for i in pred_y1:
        y1_cause.append(int(i/10000)*10000+ i % 100)
    for i in pred_y2:
        y2_cause.append(int(i/10000)*10000+ i % 100)
    s1, s2, s3 = set(new_cause), set(y1_cause), set(y2_cause)
    s3 = s2 & s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def cause_union_step(pair_id_all, pred_y1, pred_y2):
    new_cause, y1_cause, y2_cause = [], [], []
    for i in pair_id_all:
        new_cause.append(int(i/10000)*10000+ i % 100)
    for i in pred_y1:
        y1_cause.append(int(i/10000)*10000+ i % 100)
    for i in pred_y2:
        y2_cause.append(int(i/10000)*10000+ i % 100)
    s1, s2, s3 = set(new_cause), set(y1_cause), set(y2_cause)
    s3 = s2 | s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def prf_and_step(pair_id_all, pred_y1, pred_y2):
    s1, s2, s3 = set(pair_id_all), set(pred_y1), set(pred_y2)
    s3 = s2 & s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def prf_union_step(pair_id_all, pred_y1, pred_y2):
    s1, s2, s3 = set(pair_id_all), set(pred_y1), set(pred_y2)
    s3 = s2 | s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


# input:two list to form two sets,then we can solve p,r,f1

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    words = set(words)  # the set of all words
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # key:word value:index
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))  # key:index value:word

    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos
    # two dict, embedding ,pos_embendding


def load_data(input_file, word_idx, max_doc_len=75, max_sen_len=45):
    print('load data_file: {}'.format(input_file))
    y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    doc_id, mask = [], []

    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(int(line[0]))
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)
        mask1 = np.zeros((max_doc_len, 1))
        y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 1)), np.zeros((max_doc_len, 1)), np.zeros(max_doc_len,
                                                                                                          dtype=np.int32), np.zeros(
            (max_doc_len, max_sen_len), dtype=np.int32)
        for i in range(d_len):
            mask1[i] = 1
            if i + 1 in pos:
                y_po[i] = 1
            if i + 1 in cause:
                y_ca[i] = 1
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])

        y_position.append(y_po)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        mask.append(mask1)

    y_position, y_cause, x, sen_len, doc_len = map(np.array, [y_position, y_cause, x, sen_len, doc_len])
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    print(x[0])
    print(y_pairs[0])
    return doc_id, y_position, y_cause, y_pairs, x, sen_len, doc_len, mask
