import torch
import torch.nn as nn
import torch.nn.functional as F


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2):
        super(SentAttNet, self).__init__()
        self.lstm1 = nn.LSTM(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * word_hidden_size, sent_hidden_size, bidirectional=True)

        self.lstm3 = nn.LSTM(200, sent_hidden_size, bidirectional=True)
        self.lstm4 = nn.LSTM(200, sent_hidden_size, bidirectional=True)
        self.fc1 = nn.Linear(2 * sent_hidden_size, num_classes)
        self.fc2 = nn.Linear(2 * sent_hidden_size, num_classes)

        self.fc4 = nn.Linear(200, num_classes)

        self.fc6 = nn.Linear(200, num_classes)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bias1 = nn.Parameter(torch.Tensor(1))
        self.bias2 = nn.Parameter(torch.Tensor(1))

    def _create_weights(self, mean=0.0, std=0.05):
        self.bias1.data.normal_(mean, std)
        self.bias2.data.normal_(mean, std)

    def forward(self, input1, input2, label, window_size, index, dis):
        # label:batch * seq_len * 4,input1 :seq_len * batch * ( 2  * hidden_size)
        result_list_1, result_list_2 = [], []
        input1 = self.dropout1(input1)
        f_1, _ = self.lstm1(input1)  # seq_len * batch * ( 2  * hidden_size), predict emotion
        output1 = self.fc1(f_1)  # seq_len * batch *  2
        list_input_1, list_input_2 = [], []
        list_input_emotion, list_input_cause = [], []
        list_label_1, list_label_2 = [], []
        emotion, cause = [], []
        doc_1, doc_2 = [], []

        input2 = self.dropout2(input2)
        f_3, _ = self.lstm2(input2)  # seq_len * batch * ( 2  * hidden_size),predict cause
        output2 = self.fc2(f_3)

        win = int((window_size - 1) / 2)  # radius
        for j in range(output1.shape[1]):  # batch
            for i in range(output1.shape[0]):  # seq
                if output1[i, j, 1] > output1[i, j, 0]:  # is emotion
                    doc_id = label[j, i, 3].tolist()
                    doc_1.append(doc_id)
                    list_input_emotion.append(
                        f_1[i:i + 1, j:j + 1, ])  # seq_len * batch * ( 2  * hidden_size) = 1 * 1 * 200
                    if i >= win and i < output1.shape[0] - win:
                        list_input_1.append(f_3[i - win:i + win + 1, j:j + 1, :])
                    elif i < win:
                        list_input_1.append(
                            torch.cat((torch.zeros(win - i, 1, 200).cuda(), f_3[0:i + win + 1, j:j + 1, :]),
                                      0))
                    else:
                        list_input_1.append(torch.cat(
                            (f_3[i - win:, j:j + 1, :], torch.zeros(win + 1 + i - output1.shape[0], 1, 200).cuda()),
                            0))

                    lab = []
                    for c in range(i - win, i + win + 1):
                        if c >= 0 and c < output1.shape[0] and index[doc_id][c + 1] == i + 1:
                            lab.append(1)
                        else:
                            lab.append(0)
                    lab = torch.Tensor(lab)
                    lab = lab.unsqueeze(0)
                    lab = lab.unsqueeze(2)
                    list_label_1.append(lab.long().cuda())
                    emotion.append(i + 1)

        for j in range(output2.shape[1]):  # batch
            for i in range(output2.shape[0]):  # seq
                if output2[i, j, 1] > output2[i, j, 0]:  # is cause
                    doc_id = label[j, i, 3].tolist()
                    doc_2.append(doc_id)
                    list_input_cause.append(
                        f_3[i:i + 1, j:j + 1, ])  # seq_len * batch * ( 2  * hidden_size) = 1 * 1 * 200
                    if i >= win and i < output2.shape[0] - win:
                        list_input_2.append(f_1[i - win:i + win + 1, j:j + 1, :])
                    elif i < win:
                        list_input_2.append(
                            torch.cat((torch.zeros(win - i, 1, 200).cuda(), f_1[0:i + win + 1, j:j + 1, :]),
                                      0))
                    else:
                        list_input_2.append(torch.cat(
                            (f_1[i - win:, j:j + 1, :], torch.zeros(win + 1 + i - output1.shape[0], 1, 200).cuda()),
                            0))

                    lab = []
                    for c in range(i - win, i + win + 1):
                        if c >= 0 and c < output1.shape[0] and index[doc_id][i + 1] == c + 1:
                            lab.append(1)
                        else:
                            lab.append(0)
                    lab = torch.Tensor(lab)
                    lab = lab.unsqueeze(0)
                    lab = lab.unsqueeze(2)
                    list_label_2.append(lab.long().cuda())
                    cause.append(i + 1)

        if len(list_input_1) == 0 and len(list_input_2) == 0:
            return output1.permute(1, 0, 2), output2.permute(1, 0, 2), torch.Tensor(1, 1, 1), torch.Tensor(
                1), torch.Tensor(1, 1, 1), torch.Tensor(1), result_list_1, result_list_2
        elif len(list_input_2) == 0:  # no cause , has emotion
            f_output = torch.cat(list_input_1, 1)  # seq * batch * 200
            emotion_att = torch.cat(list_input_emotion, 1)  # seq=1 * batch * 200
            weight = torch.bmm(emotion_att.permute(1,0,2),f_output.permute(1,2,0)) # batch * seq
            weight = F.softmax(weight, 2).permute(2,0,1)
            f_output = f_output * weight

            list_label = torch.cat(list_label_1, 0)
            f_output_2, _ = self.lstm3(f_output)
            output3 = self.fc4(f_output_2)
            a = 0
            for j in range(output3.shape[1]):  # batch
                for i in range(output3.shape[0]):  # seq_len
                    if output3[i, j, 1] > output3[i, j, 0]:
                        result_list_1.append(
                            10000 * doc_1[a] + 100 * emotion[a] + emotion[a] + i - (output3.shape[0] - 1) / 2)
                a += 1
            return output1.permute(1, 0, 2), output2.permute(1, 0, 2), output3, list_label, torch.Tensor(1, 1,
                                                                                                         1), torch.Tensor(
                1), result_list_1, result_list_2
        elif len(list_input_1) == 0:  # no emotion , has cause
            f_output = torch.cat(list_input_2, 1)  # seq * batch * ( 2 *  hidden_size )
            cause_att = torch.cat(list_input_cause, 1)

            weight = torch.bmm(cause_att.permute(1,0,2),f_output.permute(1,2,0)) # batch * 1 *seq
            weight = F.softmax(weight, 2).permute(2,0,1)
            f_output = f_output * weight

            list_label = torch.cat(list_label_2, 0)
            f_output_2, _ = self.lstm4(f_output)
            output3 = self.fc6(f_output_2)
            a = 0
            for j in range(output3.shape[1]):  # batch
                for i in range(output3.shape[0]):  # seq_len
                    if output3[i, j, 1] > output3[i, j, 0]:
                        result_list_2.append(
                            10000 * doc_2[a] + cause[a] + 100 * (cause[a] + i - (output3.shape[0] - 1) / 2))
                a += 1
            return output1.permute(1, 0, 2), output2.permute(1, 0, 2), torch.Tensor(1, 1, 1), torch.Tensor(
                1), output3, list_label, result_list_1, result_list_2
        else:
            #print(list_input_1)
            f_output = torch.cat(list_input_1, 1)  # seq * batch * 200
            emotion_att = torch.cat(list_input_emotion, 1)  # seq=1 * batch * 200

            weight = torch.bmm(emotion_att.permute(1,0,2),f_output.permute(1,2,0)) # batch * seq
            weight = F.softmax(weight, 2).permute(2,0,1)
            f_output = f_output * weight

            list_label_1 = torch.cat(list_label_1, 0)
            f_output_2, _ = self.lstm3(f_output)
            output3 = self.fc4(f_output_2)
            a = 0
            for j in range(output3.shape[1]):  # batch
                for i in range(output3.shape[0]):  # seq_len
                    if output3[i, j, 1] > output3[i, j, 0]:
                        result_list_1.append(
                            10000 * doc_1[a] + 100 * emotion[a] + emotion[a] + i - (output3.shape[0] - 1) / 2)
                a += 1

            f_output = torch.cat(list_input_2, 1)  # seq * batch * ( 2 *  hidden_size )
            cause_att = torch.cat(list_input_cause, 1)

            weight = torch.bmm(cause_att.permute(1,0,2),f_output.permute(1,2,0)) # batch * seq
            weight = F.softmax(weight, 2).permute(2,0,1)
            f_output = f_output * weight

            list_label_2 = torch.cat(list_label_2, 0)
            f_output_2, _ = self.lstm4(f_output)
            output4 = self.fc6(f_output_2)
            a = 0
            for j in range(output4.shape[1]):  # batch
                for i in range(output4.shape[0]):  # seq_len
                    if output4[i, j, 1] > output4[i, j, 0]:
                        result_list_2.append(
                            10000 * doc_2[a] + cause[a] + 100 * (cause[a] + i - (output4.shape[0] - 1) / 2))
                a += 1
            return output1.permute(1, 0, 2), output2.permute(1, 0,
                                                             2), output3, list_label_1, output4, list_label_2, result_list_1, result_list_2


if __name__ == "__main__":
    abc = SentAttNet()
