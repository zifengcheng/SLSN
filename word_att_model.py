import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WordAttNet(nn.Module):
    def __init__(self, dict, hidden_size=50):
        super(WordAttNet, self).__init__()
        self.lookup = nn.Embedding(num_embeddings=dict.shape[0], embedding_dim=dict.shape[1]).from_pretrained(dict)
        self.lstm1 = nn.LSTM(dict.shape[1], hidden_size, bidirectional=True)
        self.lstm2 = nn.LSTM(dict.shape[1], hidden_size, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.fc4 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, input):  # input shape : 30 * 32
        output = self.lookup(input)
        output1 = self.dropout1(output)
        # print(output1.shape,len.shape)
        # print(len)
        #f_output1, _ = self.lstm1(output1.permute(1, 0, 2).float(), len)
        f_output1, _ = self.lstm1(output1.float())
        weight = F.tanh(self.fc1(f_output1))
        weight = self.fc3(weight)  # seq * batch * 1
        weight = F.softmax(weight, 0)
        weight = weight * f_output1
        output1 = weight.sum(0).unsqueeze(0)

        output = self.dropout2(output)
        f_output2, _ = self.lstm2(output.float())  # feature output and hidden state output
        weight = F.tanh(self.fc2(f_output2))
        weight = self.fc4(weight)  # seq * batch * 1
        weight = F.softmax(weight, 0)
        weight = weight * f_output2
        output2 = weight.sum(0).unsqueeze(0)
        return output1, output2


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
