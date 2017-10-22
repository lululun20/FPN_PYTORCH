import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import copy
import sys

from envs import create_atari_env
from model import ActorCritic
from model import FramePredictionNetwork


data_size = 2000
num_epochs = 50
batch_size = 40
train_data_size = data_size / 10 * 9
test_data_size = data_size / 10
lstm_hid_size = 512
retro_step = 3

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def predict(rank, args, shared_FPN, retro_step, data_queue, signal_queue, optimizer=None):

    torch.manual_seed(args.seed + rank)


    FPN = FramePredictionNetwork()


    if optimizer is None:
        optimizer = optim.Adam(shared_FPN.parameters(), lr=args.lr)


    FPN.train()
    loss_func == nn.MSELoss()


    avg_ob_loss = []
    avg_img_dis = []

    while True:
        # Sync with the shared model
        retro_buffer = []



        while data_queue.qsize() < data_size:
            sleep(20)


        signal_queue.put(True)
        
        #Get data from Queue
        data = []
        for i in range(data_size):
            data.append(data_queue.get())

        #load shared model
        FPN.load_state_dict(shared_model.state_dict())



        #shuffle data/ don't shuffle the data if it is from LSTM
#        indices = range(data_size)
#        np.random.shuffle(indices)


        done = True

        avg_img_dis = None
        total_img_dis = 0


        #Start iterating
        for epoch in range(num_epochs):

            cx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)
            hx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)


            worker_state_container = [(hx, cx)] * num_workers        
            
            for data_index in range(train_data_size):
            
                total_loss = 0

                sudo_data_index = data_index

                one_data_point = data[sudo_data_index]

                batch_size = one_data_point.shape[0]

                worker_id = one_data_point[1]

                dones = one_data_point[0][:, 2]


                #reload the state of lstm
                if done:

                    cx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)
                    hx = Varaible(torch.zeros(1, lstm_hid_size), volatile=False)

                else:
                    (cx, hx) = worker_state_container[worker_id][-1]
                    cx = Variable(cx.data, volatile=False)
                    hx = Variable(hx.data, volatile=False)
                


                one_piece_past = one_data_point[0][:, 0]
                one_true_ob = one_data_point[0][:, 1]

                #calculate average img distance

                if epoch == 0:
                    prev_img = np.zeros((1, 32, 3, 3))
                    temp_total = 0
                    for i in range(batch_size):
                        if i != batch_size - 1:
                            temp_total += np.linalg.norm(one_piece_past[i, 1] - one_piece_past[i, 0])
                        else:
                            for j in range(1, retro_step):
                                temp_total += np.linalg.norm(one_piece_past[i, j] - one_piece_past[i, j - 1])

                    total_img_dis += temp_total / (batch_size - 1 + retro_step - 1)

                
                #train FPN    
                for i in range(batch_size):
                    one_piece_past_v, one_true_ob_v = Variable(torch.FloatTensor(one_piece_past[i].cuda(), requires_grad=True), Variable(torch.FloatTensor(one_true_ob[i]).cuda()))
                    inputs = one_piece_past_v.unsqueeze(0), (hx, cx)
                    optimizer.zero_grad()
                    predicted_ob, (hx, cx) = FPN(inputs)
                    ob_loss = loss_func(predicted_ob, one_true_ob_v)
                    ob_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm(FPN.parameters(), 30)
                    ensure_shared_grads(FPN, shared_FPN)
                    optimizer.step()
                    total_loss += ob_loss.cpu().data.numpy()
                    done = dones[i]

                if len(worker_state_container) > 0:
                    worker_state_container[worker_id].pop(0)
                worker_state_container[worker_id].append((hx, cx))

            if epoch == 0:
                print('average adjacent feature distance:', total_img_dis / train_data_size)

            
            if epoch % 5 == 0:
                print('training on the ' + str(epoch) + ' epoch and the error is :' np.sqrt((total_loss / batch_size) * 288))
                
                #start validation process
                val_epoch_ob_loss = []
                done = True


                cx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)
                hx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)


                worker_state_container = [(hx, cx)] * num_workers        


                for data_index in range(test_data_size):
                    sudo_data_index = data_index + train_data_size
                    
                    one_data_point = data[sudo_data_index]

                    batch_size = one_data_point.shape[0]

                    worker_id = one_data_point[1]

                    dones = one_data_point[0][:, 2]
                    total_loss = 0

                    #reload the state of lstm
                    if done:
                        cx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)
                        hx = Varaible(torch.zeros(1, lstm_hid_size), volatile=False)

                    else:
                        (cx, hx) = worker_state_container[worker_id][-1]
                        cx = Variable(cx.data, volatile=False)
                        hx = Variable(hx.data, volatile=False)
                


                    one_piece_past = one_data_point[0][:, 0]
                    one_true_ob = one_data_point[0][:, 1]
                    
                    #calculate average img distance

                    if epoch == 0:
                        prev_img = np.zeros((1, 32, 3, 3))
                        temp_total = 0
                        for i in range(batch_size):
                            if i != batch_size - 1:
                                temp_total += np.linalg.norm(one_piece_past[i, 1] - one_piece_past[i, 0])
                            else:
                                for j in range(1, retro_step):
                                    temp_total += np.linalg.norm(one_piece_past[i, j] - one_piece_past[i, j - 1])

                        total_img_dis += temp_total / (batch_size - 1 + retro_step - 1)


                    
                    #calculate validation error
                    for i in range(batch_size):
                        one_piece_past_v, one_true_ob_v = Variable(torch.FloatTensor(one_piece_past[i].cuda(), requires_grad=True), Variable(torch.FloatTensor(one_true_ob[i]).cuda()))
                        inputs = one_piece_past_v.unsqueeze(0), (hx, cx)
                        optimizer.zero_grad()
                        predicted_ob, (hx, cx) = FPN(inputs)
                        ob_loss = loss_func(predicted_ob, one_true_ob_v)
                        total_loss += ob_loss.cpu().data.numpy()
                        done = dones[i]

            
                    if len(worker_state_container) > 0:
                        worker_state_container[worker_id].pop(0)
                    worker_state_container[worker_id].append((hx, cx))



                    if epoch == 0:
                        print('average adjacent feature distance for validation dataset:', total_img_dis / train_data_size)

            
                    if epoch % 5 == 0:
                        print('The validation error ' + str(epoch) + ' epoch and the error is :' np.sqrt((total_loss / batch_size) * 288))




        #when training is done, notify the worker to resume work
        signal_queue.get()

