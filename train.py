import numpy as np
import torch
import argparse
import time
import torch.utils.data as Data
from hierarchical_att_model import HierAttNet
import torch.nn as nn
from utils.prepare_data import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lamda", dest="lam", type=float, metavar='<float>', default=0.6)
parser.add_argument("--belta", dest="belta", type=float, metavar='<float>', default=0.8)
parser.add_argument("--gpu", dest="b", type=str, default='0')
parser.add_argument("--window", dest="win", type=int, default=5)
parser.add_argument("--lr", dest="lr", type=float, default=0.005)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.b


def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    word_idx_rev, word_idx, embedding, embedding_pos = load_w2v(200, 100, 'data/clause_keywords.csv',
                                                                'data/w2v_200.txt')
    e_1, e_2, e_3 = [], [], []
    c_1, c_2, c_3 = [], [], []
    u_1, u_2, u_3 = [], [], []
    i_1, i_2, i_3 = [], [], []
    e_e_1, e_e_2, e_e_3, e_c_1, e_c_2, e_c_3 = [], [], [], [], [], []
    c_e_1, c_e_2, c_e_3, c_c_1, c_c_2, c_c_3 = [], [], [], [], [], []
    u_e_1, u_e_2, u_e_3, u_c_1, u_c_2, u_c_3 = [], [], [], [], [], []
    i_e_1, i_e_2, i_e_3, i_c_1, i_c_2, i_c_3 = [], [], [], [], [], []


    p4, r4, f2 = [], [], []
    p5, r5, f5 = [], [], []
    p6, r6, f6 = [], [], []


    for fold in range(1, 11):
        best_emotion = [-1, -1, -1, -1]
        best_cause = [-1, -1, -1, -1]
        best_pair_emotion = [-1, -1, -1, -1]
        best_pair_cause = [-1, -1, -1, -1]
        best_pair_union = [-1, -1, -1, -1]
        best_pair_inter = [-1, -1, -1, -1]
        best_emo_emo = [-1, -1, -1, -1]
        best_emo_cau = [-1, -1, -1, -1]
        best_cau_emo = [-1, -1, -1, -1]
        best_cau_cau = [-1, -1, -1, -1]
        best_uni_emo = [-1, -1, -1, -1]
        best_uni_cau = [-1, -1, -1, -1]
        best_int_emo = [-1, -1, -1, -1]
        best_int_cau = [-1, -1, -1, -1]
        print('############# fold {} begin ###############'.format(fold))
        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        tr_doc_id, tr_y_position, tr_y_cause, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len, tr_mask = load_data(
            'data_combine/' + train_file_name, word_idx, 75, 30)
        te_doc_id, te_y_position, te_y_cause, te_y_pairs, te_x, te_sen_len, te_doc_len, te_mask = load_data(
            'data_combine/' + test_file_name, word_idx, 75, 30)

        train_pair = load_pair(train_file_name)
        test_pair = load_pair(test_file_name)
        index = load_index('data_combine/all_data_pair.txt')

        print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

        tr_x = torch.LongTensor(tr_x)
        te_x = torch.LongTensor(te_x)
        tr_y_position, tr_y_cause, tr_mask, tr_do = torch.LongTensor(tr_y_position), torch.LongTensor(
            tr_y_cause), torch.LongTensor(tr_mask), torch.LongTensor(tr_doc_id)
        tr_do = tr_do.unsqueeze(1)
        tr_do = tr_do.unsqueeze(1)
        tr_do = tr_do.expand_as(tr_y_position)
        tr_label = torch.cat((tr_y_position, tr_y_cause, tr_mask, tr_do), -1)

        te_y_position, te_y_cause, te_mask, te_do = torch.LongTensor(te_y_position), torch.LongTensor(
            te_y_cause), torch.LongTensor(te_mask), torch.LongTensor(te_doc_id)
        te_do = te_do.unsqueeze(1)
        te_do = te_do.unsqueeze(1)
        te_do = te_do.expand_as(te_y_position)
        te_label = torch.cat((te_y_position, te_y_cause, te_mask, te_do), -1)

        train_data = Data.TensorDataset(tr_x, tr_label)
        test_data = Data.TensorDataset(te_x, te_label)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=250, shuffle=True, num_workers=2)
        embedding = torch.tensor(embedding)
        dis = torch.randn(args.win, 1, 50).cuda()

        model = HierAttNet(100, 100, 32, 2, embedding, 75, 30, args.win, index, dis)
        model.word_att_net.lookup.weight.requires_grad = False
        if torch.cuda.is_available():
            model.cuda()
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.shape)
        criterion1 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5]).cuda())
        criterion2 = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1]).cuda())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        model.train()

        for epoch in range(25):
            result, result_e, result_c = [], [], []
            start_time, step = time.time(), 1
            pos, cau = [-1, -1, -1], [-1, -1, -1]
            for iter, (feature, label) in enumerate(train_loader):
                if torch.cuda.is_available():
                    feature = feature.cuda()  # shape : 32 * 75 * 30
                    label = label.cuda()
                optimizer.zero_grad()
                # model._init_hidden_state()
                predict_1, predict_2, predict_3, lab_3, predict_4, lab_4, pair_1, pair_2 = model(feature, label)
                # print("pair", pair_1)
                if predict_3.shape[2] == 1 and predict_4.shape[2] == 1:
                    loss1 = criterion1(predict_1.permute(0, 2, 1), label[:, :, 0])
                    loss2 = criterion1(predict_2.permute(0, 2, 1), label[:, :, 1])
                    loss = args.lam * args.belta * loss1 + (1 - args.lam) * args.belta * loss2
                elif predict_3.shape[2] == 1:  # no emotion LPS
                    loss1 = criterion1(predict_1.permute(0, 2, 1), label[:, :, 0])
                    loss2 = criterion1(predict_2.permute(0, 2, 1), label[:, :, 1])
                    loss3 = criterion2(predict_4.permute(1, 2, 0), lab_4.squeeze(2))
                    loss = args.lam * args.belta * loss1 + (1 - args.lam) * args.belta * loss2 + (1 - args.lam) * (
                            1 - args.belta) * loss3
                    result_c.extend(pair_2)
                elif predict_4.shape[2] == 1:  # no cause LPS
                    loss1 = criterion1(predict_1.permute(0, 2, 1), label[:, :, 0])
                    loss2 = criterion1(predict_2.permute(0, 2, 1), label[:, :, 1])
                    loss3 = criterion2(predict_3.permute(1, 2, 0), lab_3.squeeze(2))
                    loss = args.lam * args.belta * loss1 + (1 - args.lam) * args.belta * loss2 + args.lam * (
                            1 - args.belta) * loss3
                    result_e.extend(pair_1)
                else:
                    loss1 = criterion1(predict_1.permute(0, 2, 1), label[:, :, 0])
                    loss2 = criterion1(predict_2.permute(0, 2, 1), label[:, :, 1])
                    loss3 = criterion2(predict_3.permute(1, 2, 0), lab_3.squeeze(2))
                    loss4 = criterion2(predict_4.permute(1, 2, 0), lab_4.squeeze(2))
                    loss = args.lam * args.belta * loss1 + (1 - args.lam) * args.belta * loss2 + args.lam * (
                            1 - args.belta) * loss3 + (1 - args.lam) * (1 - args.belta) * loss4
                    result_e.extend(pair_1)
                    result_c.extend(pair_2)
                # re_l2 = ['sent_att_net.fc1.weight','sent_att_net.fc2.weight','sent_att_net.fc4.weight','sent_att_net.fc6.weight']
                # for name,parameters in model.named_parameters():
                #    if name in re_l2 :
                #	loss += torch.norm(parameters) *  1e-6
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                if step % 10 == 0:
                    pred_1 = predict_1.detach().cpu().numpy()
                    pred_1 = pred_1.argmax(-1)
                    labe_1 = label[:, :, 0].detach().cpu().numpy()
                    mas = label[:, :, 2].detach().cpu().numpy()

                    pred_2 = predict_2.detach().cpu().numpy()
                    pred_2 = pred_2.argmax(-1)
                    labe_2 = label[:, :, 1].detach().cpu().numpy()

                    print('epoch :  {}  step {}: train loss {:.4f} '.format(epoch, step, loss))
                    acc, p, r, f1 = acc_prf(pred_1, labe_1, mas)
                    print('position_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                    acc, p, r, f1 = acc_prf(pred_2, labe_2, mas)
                    print('cause_predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                step = step + 1

            test_e, test_c = [], []
            if epoch >= 0:
                model.eval()
                loss_ls = []
                count = 0
                for te_feature, te_label in test_loader:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        model._init_hidden_state(num_sample)
                        predict_1, predict_2, predict_3, lab_3, predict_4, lab_4, pair_1, pair_2 = model(te_feature,
                                                                                                         te_label)
                        test_e.extend(pair_1)
                        test_c.extend(pair_2)
                pred_y_emotion = predict_1.argmax(dim=2).cpu().clone().detach().numpy()
                pred_y_cause = predict_2.argmax(dim=2).cpu().clone().detach().numpy()

            p_test, r_test, f1_test = prf_2nd_step(test_pair, test_e)
            # print("test pair is{}".format(test_e))
            if f1_test > best_pair_emotion[-1]:
                best_pair_emotion[1] = p_test
                best_pair_emotion[2] = r_test
                best_pair_emotion[3] = f1_test
            print('epoch :{},emotion first:test p is {},r is {},f1 is {}'.format(epoch, p_test, r_test, f1_test))

            p_test, r_test, f1_test = prf_2nd_step(test_pair, test_c)
            if f1_test > best_pair_cause[-1]:
                best_pair_cause[1] = p_test
                best_pair_cause[2] = r_test
                best_pair_cause[3] = f1_test
            print('epoch :{},cause first:test p is {},r is {},f1 is {}'.format(epoch, p_test, r_test, f1_test))

            p_test, r_test, f1_test = prf_union_step(test_pair, test_c, test_e)
            if f1_test > best_pair_union[-1]:
                acc, p, r, f1 = acc_prf(pred_y_emotion, te_label[:,:,0].cpu().numpy(), te_label[:,:,2].cpu().numpy())
                print('emotion extraction p is {} r is {} f1 is {}'.format(p,r,f1))
                best_emotion[0] = epoch
                best_emotion[1] = p
                best_emotion[2] = r
                best_emotion[3] = f1  
                acc, p, r, f1 = acc_prf(pred_y_cause, te_label[:,:,1].cpu().numpy(), te_label[:,:,2].cpu().numpy())
                print('cause extraction p is {} r is {} f1 is {}'.format(p,r,f1))
                best_cause[0] = epoch
                best_cause[1] = p
                best_cause[2] = r
                best_cause[3] = f1                
                best_pair_union[1] = p_test
                best_pair_union[2] = r_test
                best_pair_union[3] = f1_test
            print('epoch :{},union:test p is {},r is {},f1 is {}'.format(epoch, p_test, r_test, f1_test))

            p_test, r_test, f1_test = prf_and_step(test_pair, test_c, test_e)
            if f1_test > best_pair_inter[-1]:
                best_pair_inter[1] = p_test
                best_pair_inter[2] = r_test
                best_pair_inter[3] = f1_test
            print('epoch :{},and: test p is {},r is {},f1 is {}'.format(epoch, p_test, r_test, f1_test))
            #print(dis)
            model.train()

        print('this fold best emotion p is {} r is {} f1 is {} best epoch is {}'.format(best_emotion[1], best_emotion[2], best_emotion[3], best_emotion[0]))

        print('this fold best cause p is {} r is {} f1 is {} best epoch is {}'.format(best_cause[1], best_cause[2], best_cause[3], best_cause[0]))

        p5.append(best_emotion[1])
        r5.append(best_emotion[2])
        f5.append(best_emotion[3])

        p6.append(best_cause[1])
        r6.append(best_cause[2])
        f6.append(best_cause[3])

        print('best pair f1: emotion ,p is {},r is {},f1 is {}'.format(best_pair_emotion[1], best_pair_emotion[2],
                                                                       best_pair_emotion[3]))
        print('best pair f1: cause,p is {},r is {},f1 is {}'.format(best_pair_cause[1], best_pair_cause[2],
                                                                    best_pair_cause[3]))
        print('best pair f1: union,p is {},r is {},f1 is {}'.format(best_pair_union[1], best_pair_union[2],
                                                                    best_pair_union[3]))
        print('best pair f1: inter,p is {},r is {},f1 is {}'.format(best_pair_inter[1], best_pair_inter[2],
                                                                    best_pair_inter[3]))

        e_1.append(best_pair_emotion[1])
        e_2.append(best_pair_emotion[2])
        e_3.append(best_pair_emotion[3])
        c_1.append(best_pair_cause[1])
        c_2.append(best_pair_cause[2])
        c_3.append(best_pair_cause[3])

        u_1.append(best_pair_union[1])
        u_2.append(best_pair_union[2])
        u_3.append(best_pair_union[3])
        i_1.append(best_pair_inter[1])
        i_2.append(best_pair_inter[2])
        i_3.append(best_pair_inter[3])

    all_results = [e_1, e_2, e_3, c_1, c_2, c_3, u_1, u_2, u_3, i_1, i_2, i_3]
    e_1, e_2, e_3, c_1, c_2, c_3, u_1, u_2, u_3, i_1, i_2, i_3 = map(lambda x: np.array(x).mean(),
                                                                     all_results)

    print('average resut in 10 fold : emotion p is{} r is {} f1 is {}'.format(e_1, e_2, e_3))
    print('average resut in 10 fold : cause p is{} r is {} f1 is {}'.format(c_1, c_2, c_3))
    print('average resut in 10 fold : union p is{} r is {} f1 is {}'.format(u_1, u_2, u_3))
    print('average resut in 10 fold : inte p is{} r is {} f1 is {}'.format(i_1, i_2, i_3))

    all_results = [p5, r5, f5]
    p4, r4, f1 = map(lambda x: np.array(x).mean(), all_results)
    print('finally emotion result: p is {} r is {} f1 is {}'.format(p4, r4, f1))

    all_results = [p6, r6, f6]
    p4, r4, f1 = map(lambda x: np.array(x).mean(), all_results)
    print('finally cause result: p is {} r is {} f1 is {}'.format(p4, r4, f1))



if __name__ == "__main__":
    train()
