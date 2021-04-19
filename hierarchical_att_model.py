"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from sent_att_model import SentAttNet
from word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, dict,
                 max_sent_length, max_word_length, window_size, index, dis):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(dict, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()
        self.window_size = window_size
        self.index = index
        self.dis = dis

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state1 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state1 = self.sent_hidden_state1.cuda()
            self.sent_hidden_state2 = self.sent_hidden_state2.cuda()

    def forward(self, input, label):

        output_list1 = []
        output_list2 = []
        input = input.permute(1, 0, 2) # 75 * 32 * 30
        for i in input:
            output1, output2 = self.word_att_net(i.permute(1, 0))
            output_list1.append(output1)
            output_list2.append(output2)
        output1 = torch.cat(output_list1, 0)
        output2 = torch.cat(output_list2, 0)
        output1, output2, output3, lab_3, output4, lab_4, result_list_1, result_list_2 = self.sent_att_net(output1,
                                                                                                           output2,
                                                                                                           label,
                                                                                                           self.window_size,
                                                                                                           self.index,
                                                                                                           self.dis)

        return output1, output2, output3, lab_3, output4, lab_4, result_list_1, result_list_2
