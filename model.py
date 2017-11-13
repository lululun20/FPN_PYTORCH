import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import operator
from collections import defaultdict


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)




class ValuePredictionNetwork(torch.nn.Module):
    def __init__(self, num_inputs, action_space, predict_step, branch):
        super(ValuePredictionNetwork, self).__init__()

        self.branch = branch
        self.action_size = action_space.n

        self.data_buffer = defaultdict(list)

        #encoding module
        self.gamma = 0.99

        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
#        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

#        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)



        #transition module 

        self.tran_conv1 = nn.Conv2d(33, 32, 3, stride=1, padding=1)
        self.tran_conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.tran_mask = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.tran_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        #ouput module
        self.out_conv1 = nn.Conv2d(33, 32, 3, stride=1, padding=1)
        self.out_conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.out_hidden = nn.Linear(32 * 5 * 5, 256)
        self.reward_linear = nn.Linear(256, 1)
        
        #value module
        self.value_hidden = nn.Linear(32 * 5 * 5, 256)
        self.value_linear = nn.Linear(256, 1)

        
        


        self.apply(weights_init)
        self.out_hidden.weight.data = normalized_columns_initializer(
            self.out_hidden.weight.data, 0.1)
        self.out_hidden.bias.data.fill_(0)

        self.value_hidden.weight.data = normalized_columns_initializer(
            self.value_hidden.weight.data, 0.1)
        self.value_hidden.bias.data.fill_(0)


        self.value_linear.weight.data = normalized_columns_initializer(
            self.value_linear.weight.data, 0.1)
        self.value_linear.bias.data.fill_(0)
        self.reward_linear.weight.data = normalized_columns_initializer(
            self.reward_linear.weight.data, 0.01)
        self.reward_linear.bias.data.fill_(0)

#        self.lstm.bias_ih.data.fill_(0)
#        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def encode(self, x):
        x = F.elu(self.conv1(x))
#        x = F.elu(self.conv2(x))
#        x = F.elu(self.conv3(x))
        abstract_state = F.elu(self.conv2(x))
        return abstract_state


    def outcome(self, state, action):
        x = self.merge(state, action)
        x = F.elu(self.out_conv1(x))
        x = F.elu(self.out_conv2(x))
        x = x.view(-1, 32 * 5 * 5)
        x = F.elu(self.out_hidden(x))

        return self.reward_linear(x)


    def merge(self, state, action):
        one_action = np.full([1, 1, 5, 5], action)
        inputs = np.concatenate((state.data.numpy(), one_action), axis=1)
        return Variable(torch.FloatTensor(inputs))


    def transit(self, state, action):
        inputs = self.merge(state, action)
        x = F.elu(self.tran_conv1(inputs))
        x = F.elu(self.tran_conv2(x))
        x1 = F.elu(self.tran_conv3(x))

        mask = F.sigmoid(self.tran_mask(x))
        next_state = x1 * mask + state
        return next_state


    def value(self, state):
        state = state.view(-1, 32 * 5 * 5)
        x = F.elu(self.value_hidden(state))
        return self.value_linear(x)
        
    
    def core(self, state, action):
        reward_next = self.outcome(state, action)
        next_state = self.transit(state, action)
        value_next = self.value(next_state)
        return next_state, reward_next, value_next
        
    def plan(self, current_state, action, d):
        value_holder = []
        state_holder = []
        next_state, reward_next, value_next = self.core(current_state, action)

        if d == 1:
            return reward_next.data.numpy()[0, 0] + self.gamma * value_next.data.numpy()[0, 0]

        else:
            current_state = next_state
            for action in range(self.action_size):
                n_state, r_next, v_next = self.core(current_state, action)
                value_holder.append(v_next.data.numpy()[0, 0] + r_next.data.numpy()[0, 0])

            v_max, a_max = torch.topk(torch.FloatTensor(np.array(value_holder)), self.branch)
            sim_value = {}
            for action in a_max:
                Q_a = self.plan(current_state, action, d - 1) 
                sim_value[action] = Q_a
            action = max(sim_value.items(), key=operator.itemgetter(1))[0]
            return reward_next.data.numpy()[0, 0] + self.gamma * (1 / d * value_next.data.numpy()[0, 0] + (d - 1) / d * sim_value[action])



    def predict(self, current_state, actions, t):
        c_state = current_state
        for action in actions:
            t = t + 1
            c_state, reward_next, value_next = self.core(c_state, action)
            self.data_buffer[t].append((value_next, reward_next))
            














