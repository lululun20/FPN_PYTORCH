from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import sys

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




class FramePredictionNetwork(nn.Module):
    def __init__(self):
        super(FramePredictionNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
#        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
#        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
#        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
#        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)


        #self.fc1 = nn.Linear()

        self.apply(weights_init)

        self.fc1 = nn.Linear(32 * 16 * 24, 1280)
        self.fc2 = nn.Linear(1280, 288)

        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.1)


    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
#        x = F.elu(self.conv3(x))
#        x = F.elu(self.conv4(x))
#        x = F.elu(self.conv5(x))
#        x = F.elu(self.conv6(x))



        x = x.view(-1, 32 * 16 * 24)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return x




h_size = 256


retro_step = 3
FPN = FramePredictionNetwork()
FPN.cuda()
num_epochs = 50
data_size = 500
train_data_size = 400
buff = 0
test_data_size = 100
data_path = 'PongFF/'

def read_data(data_path, data_size, train_data_size, test_data_size):
    #read data 
    past_history= []
    episode_next = []
    episode_reward = []
    episode_done = []
    print('reading data')
    avg = []
    for data_index in range(data_size):
        #print 'reading ' + str(data_index) + ' file'
        data_index += buff

        data_path_for_one_ep = data_path + 'episode_' + str(data_index) + '/'
        one_episode_buffer = []
        one_episode_action = []
        one_episode_reward = []
        one_episode_done = []

        action_file = data_path_for_one_ep + 'action.txt'
        reward_file = data_path_for_one_ep + 'reward.txt'
        done_file = data_path_for_one_ep + 'done.txt'

        f = open(action_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_action.append(float(line.strip()))
            
        f = open(reward_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_reward.append(float(line.strip()))

        f = open(done_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_done.append(bool(line.strip()))
        
        episode_len = len(one_episode_action)

        
        for image_index in range(episode_len):
            file_path = data_path_for_one_ep + 'image_' + str(image_index) + '.npy'
            im = np.load(file_path)
            one_episode_buffer.append(im)
        if len(one_episode_buffer) <= retro_step:
            print('discard')
            train_data_size -= 1
            test_data_size -= 1
            continue

        one_past_history = []
        avg_dis = []
        for i in range(retro_step, len(one_episode_buffer)):
            one_past = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(i - retro_step, i)])

            #one_past = np.array(one_episode_buffer[i - 1])
            #print out the difference between each adjacent frame
            res = np.linalg.norm(np.array(one_episode_buffer[i - 1]) - np.array(one_episode_buffer[i - 2]))

            avg_dis.append(res)

            one_action = one_episode_action[i - 1 : i]
            one_action = np.full(one_past[0].shape, one_action)
            #one_action = np.full(one_past.shape, one_action)
            one_moment = np.vstack((one_past, one_action))

            one_past_history.append(one_moment)
            #one_past_history.append(one_past)
 
        next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step, len(one_episode_buffer))])

       #cheat training
#        next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step - 1, len(one_episode_buffer) - 1)])



        rewards = np.vstack(one_episode_reward[retro_step - 1 : -1])
        dones = np.vstack(one_episode_done[retro_step - 1 : -1])
        one_past_history = np.array(one_past_history)
        
        #print 'data_index: ' + str(data_index), np.mean(avg_dis)
        avg.append(np.mean(avg_dis))



        past_history.append(one_past_history)
        episode_next.append(next_observations)
        episode_reward.append(rewards)
        episode_done.append(dones)

    print(np.mean(avg))

    
    #shuffle data

    indices = [i for i in range(len(past_history))]
    np.random.shuffle(indices)
    past_history = [past_history[i] for i in indices]
    episode_next = [episode_next[i] for i in indices]
    episode_reward = [episode_reward[i] for i in indices]
    episode_done = [episode_done[i] for i in indices]

    return past_history, episode_next, episode_reward, episode_done, train_data_size, test_data_size

past_history, episode_next, _, _, train_data_size, test_data_size = read_data(data_path, data_size, train_data_size, test_data_size)



#Training
print('training...')

optimizer = optim.Adam(FPN.parameters(), lr=1e-4)

FPN.train()

loss_func = nn.MSELoss()


               
for epoch in range(num_epochs):
    epoch_loss = []
    epoch_ob_loss = []
    epoch_reward_loss = []
    epoch_done_loss = []
    
    for data_index in range(train_data_size):            
        #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
        #print 'training ' + str(data_index) + ' file'
        sudo_data_index = data_index
                
        optimizer.zero_grad()

        one_piece_past = past_history[sudo_data_index]


        one_piece_past = np.reshape(one_piece_past, (-1, 3, (retro_step + 1) * 4, 24))
        one_true_ob = episode_next[sudo_data_index]

        one_piece_past, one_true_ob = Variable(torch.FloatTensor(one_piece_past).cuda()), Variable(torch.FloatTensor(one_true_ob).cuda())
        predicted_ob = FPN(one_piece_past)

        
            
        ob_loss = loss_func(predicted_ob, one_true_ob)

        np.savetxt('a.txt', predicted_ob.cpu().data.numpy())
        np.savetxt('b.txt', one_true_ob.cpu().data.numpy())
        

#        f = open("test.txt", 'w')
#        f.writelines(predicted_ob.cpu().data.numpy)
#        f.writelines(one_true_ob.cpu().data.numpy)

#        print(predicted_ob.size(), one_true_ob.size(), np.linalg.norm(predicted_ob.cpu().data.numpy() - episode_next[sudo_data_index][0]), np.sqrt(ob_loss.cpu().data.numpy() * 288))
                
        ob_loss.backward()
        optimizer.step()

        epoch_ob_loss.append(ob_loss.data[0])

        

    if epoch % 5 == 0 and epoch != 0:
        o_l = np.mean(epoch_ob_loss)
        print('training on the ' + str(epoch) + ' epoch and the error is :')
        print(np.sqrt(o_l * 288))


        #print out the validation error

        val_epoch_ob_loss = []
        
        for data_index in range(test_data_size):            
            #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
            #print 'training ' + str(data_index) + ' file'
            sudo_data_index = data_index + train_data_size
            

            one_piece_past = past_history[sudo_data_index]        
            one_piece_past = np.reshape(one_piece_past, (-1, 3, (retro_step + 1) * 4, 24))
            one_true_ob = episode_next[sudo_data_index]
            one_piece_past, one_true_ob = Variable(torch.FloatTensor(one_piece_past).cuda()), Variable(torch.FloatTensor(one_true_ob).cuda())
            predicted_ob = FPN(one_piece_past)
            
            ob_loss = loss_func(predicted_ob, one_true_ob)
            
            val_epoch_ob_loss.append(ob_loss.data[0])


        o_l = np.mean(val_epoch_ob_loss)

        print('Validate error is: ')
        print(np.sqrt(o_l * 288))


