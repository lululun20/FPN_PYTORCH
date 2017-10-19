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


def predict(rank, args, shared_model, optimizer=None, data_queue, num_workers):

    torch.manual_seed(args.seed + rank)


    FPN = FramePredictionNetwork()


    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)


    FPN.train()
    loss_func == nn.MSELoss()


    avg_ob_loss = []
    avg_img_dis = []

    while True:
        # Sync with the shared model
        retro_buffer = []

        #load shared model
        FPN.load_state_dict(shared_model.state_dict())


        while data_queue.qsize() < data_size:
            sleep(1)


        #Get data from Queue
        data = []
        for i in range(data_size):
            data.append(data_queue.get())

        #shuffle data
        indices = range(data_size)
        np.random.shuffle(indices)


        done = True

        avg_img_dis = None
        total_img_dis = 0

        cx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)
        hx = Variable(torch.zeros(1, lstm_hid_size), volatile=False)



        worker_state_container = [(hx, cx)] * num_workers

        #Start iterating
        for epoch in range(num_epochs):

            cx = Variable(torch.zeros(1, lstm_hid_size))
            hx = Variable(torch.zeros(1, lstm_hid_size))

        
            
            for data_index in range(train_data_size):
            
                total_loss = 0

                sudo_data_index = data_index

                one_data_point = data[sudo_data_index]

                batch_size = one_data_point.shape[0]

                worker_id = one_data_point[1]

                dones = one_data_point[0][:, 2]
                done = dones[0]

                #reload the state of lstm
                if done:
                    FPN.load_state_dict(shared_model.state_dict())
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

                    
                for i in range(batch_size):
                    one_piece_past_v, one_true_ob_v = Variable(torch.FloatTensor(one_piece_past[i].cuda(), requires_grad=True), Variable(torch.FloatTensor(one_true_ob[i]).cuda()))
                    inputs = one_piece_past_v.unsqueeze(0), (hx, cx)
                    optimizer.zero_grad()
                    predicted_ob, (hx, cx) = FPN(inputs)
                    ob_loss = loss_func(predicted_ob, one_true_ob_v)
                    ob_loss.backward(retain_graph=True)
                    optimizer.step()
                    total_loss += ob_loss.cpu().data.numpy()

                if len(worker_state_container) > 0:
                    worker_state_container[worker_id].pop(0)
                worker_state_container[worker_id].append((hx, cx))

            if epoch == 0:
                print('average adjacent feature distance:', total_img_dis / train_data_size)

            
            if epoch % 5 == 0:
                print('training on the ' + str(epoch) + ' epoch and the error is :')
                
        


































                    




#            prev_feature = conv4.data.numpy()

            #create samples for FPN
#            if len(retro_buffer) == retro_step:
#                a = action.numpy()
#                one_action = np.full(retro_buffer[0].shape, a)
#                one_piece = copy.deepcopy(retro_buffer)
#                one_piece.append(one_action)
#                one_piece = np.vstack(one_piece)
#                one_piece = np.reshape(one_piece, (3, (retro_step + 1) * 4, 24))
#                history_buffer.append(one_piece)                                
#                next_ob_buffer.append(conv4.data.numpy())
                
            if len(retro_buffer) == retro_step:
                retro_buffer.pop(0)

            retro_buffer.append(conv4.data.numpy())                


                
            done = done or episode_length >= args.max_episode_length            

            sum_rewards += reward
        
            reward = max(min(reward, 1), -1)
            

            if done:
                episode_length = 0
                state = env.reset()


            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

 #       if rank == 0 and len(img_dis) > 0:
 #           avg_img_dis.append(np.mean(img_dis))
            
 #       if rank == 0 and episode_count % 200 == 0 and len(avg_img_dis) > 0:
 #           print("average image distance: ", np.mean(avg_img_dis))
 #           avg_img_dis = []

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ , _= model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]



        if len(pred_buffer) > 0:

            next_ob_buffer = np.vstack(next_ob_buffer)
            pred_buffer = np.vstack(pred_buffer)

            one_true_ob = Variable(torch.FloatTensor(next_ob_buffer))
            
            predicted_ob = Variable(torch.FloatTensor(pred_buffer), requires_grad=True)

        

            obs_loss = []

            for i in range(10):

                FPN_optimizer.zero_grad()

                ob_loss = FPN_loss_func(predicted_ob, one_true_ob)

                ob_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(FPN.parameters(), 30)

                ensure_shared_grads(FPN, shared_FPN)

                FPN_optimizer.step()

                obs_loss.append(np.sqrt(ob_loss.data.numpy() * 288))
            
#            avg_ob_loss.append(np.mean(obs_loss))
            avg_ob_loss.append(obs_loss[-1])


        optimizer.zero_grad()


        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)


        ensure_shared_grads(model, shared_model)

        

        optimizer.step()

        if rank == 0 and episode_count % 200 == 0:
            print('observation_loss', np.mean(avg_ob_loss))
            avg_ob_loss = []

#        if rank == 0:
#            print('value_loss', value_loss.data.numpy())


