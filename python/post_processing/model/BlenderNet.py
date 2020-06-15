import torch
import torch.nn as nn


class BlenderNet(nn.Module):
    def __init__(self, num_blends):
        super(BlenderNet, self).__init__()

        self.num_blends = num_blends
        self.fc1 = nn.Linear(num_blends*4, 4)


    def forward(self, input_logits):
        return self.fc1(input_logits)


class TaskIndependentNet(nn.Module):
    def __init__(self, num_blends):
        super(TaskIndependentNet, self).__init__()

        self.num_blends = num_blends

        self.reply_net = nn.Linear(num_blends, 1)
        self.retweet_net = nn.Linear(num_blends, 1)
        self.comment_net = nn.Linear(num_blends, 1)
        self.like_net = nn.Linear(num_blends, 1)

        self.uniform_init()

        self.nets = [self.reply_net, self.retweet_net, self.comment_net, self.like_net]


    def forward(self, input_logits):

        TASKS = ["reply", "retweet", "comment", "like"]

        logits = []

        for i, T in enumerate(TASKS):

            sub_model = self.nets[i]
            sub_inputs = input_logits[:, range(i, input_logits.shape[1], 4)]

            logits.append(sub_model(sub_inputs))

        return torch.cat(logits, dim=1)


    def uniform_init(self):
        with torch.no_grad():
            shape = self.reply_net.weight.shape
            self.reply_net.weight = nn.Parameter(torch.ones(shape)*(1/self.num_blends))
            self.retweet_net.weight = nn.Parameter(torch.ones(shape)*(1/self.num_blends))
            self.comment_net.weight = nn.Parameter(torch.ones(shape)*(1/self.num_blends))
            self.like_net.weight = nn.Parameter(torch.ones(shape)*(1/self.num_blends))
            self.reply_net.bias = nn.Parameter(torch.zeros(1))
            self.retweet_net.bias = nn.Parameter(torch.zeros(1))
            self.comment_net.bias = nn.Parameter(torch.zeros(1))
            self.like_net.bias = nn.Parameter(torch.zeros(1))


class BlenderNetSmall(nn.Module):
    def __init__(self, num_blends):
        super(BlenderNetSmall, self).__init__()

        self.num_blends = num_blends
        #self.fc1 = nn.Linear(num_blends*4, 64)
        #self.fc2 = nn.Linear(64, 4)
        self.fc = nn.Linear(num_blends*4,4)

    def forward(self, input_logits):

        #h = self.fc1(input_logits)
        #logits = self.fc2(h)
        logits = self.fc(input_logits)

        return logits

