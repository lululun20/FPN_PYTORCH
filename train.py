import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import copy
import sys

from envs import create_atari_env
from model import ValuePredictionNetwork
import random


def ensure_shared_grads(model, shared_model):

    #ensure you are optimizing over shared model

    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, signal_queue, optimizer=None):


    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    #Set up the value and policy network and the model(FPN) network

    model = ValuePredictionNetwork(env.observation_space.shape[0] * args.frame_skip, env.action_space, args.predict_step, args.branch_factor)


    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)


    model.train()

    state = env.reset()
    action_size = env.action_space.n
    burst_states = []



    #tiling states
    for _ in range(args.frame_skip):
        burst_states.append(state)
        
        
    state = np.vstack(burst_states)
    state = torch.from_numpy(state)


    done = True

        

    #length of a one episode
    episode_length = 0

    #number of total episodes played
    episode_count = 0


    epsilon = 1
    annealing_steps = 1000000
    annealing_reduction = (epsilon - 0.05) / annealing_steps

    while True:

        if signal_queue.qsize() > 0:
            sys.exit(str(rank) + 'th thread has been stopped')

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        states = []
        states.append(state)
        actions = []
        rewards = []
                

        episode_count += 1


        for step in range(args.num_steps):

            episode_length += 1
            num = random.uniform(0, 1)

            if num < epsilon:
                action = np.random.choice(action_size, 1)[0]

            else:
                value_holder = []
                for action in range(action_size):
                    plan_value = model.plan(model.encode(Variable(state.unsqueeze(0))), action, args.plan_depth)
                    value_holder.append(plan_value)
                action = np.argmax(value_holder)
            if epsilon > 0.05:
                epsilon -= annealing_reduction

            
            #Perform frame skip
            burst_reward_sum = 0
            burst_state = []

            for _ in range(args.frame_skip):
                state, reward, done, _ = env.step(action)                                        
                burst_state.append(state)
                reward = max(min(reward, 1), -1)
                burst_reward_sum += reward
                done = done or episode_length >= args.max_episode_length            
                if done:
                    break
            
            reward = burst_reward_sum

            
            if done:
                episode_length = 0
                state = env.reset()
                burst_state = []
                while len(burst_state) < args.frame_skip:
                    burst_state.append(state)


            burst_state = np.vstack(burst_state)
            state = torch.from_numpy(burst_state)


            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if done:
                break


        

        R = torch.zeros(1, 1)
        if not done:
            value_holder = []
            #bootstrapping
            for action in range(action_size):
                plan_value = model.plan(model.encode(Variable(state.unsqueeze(0))), action, args.plan_depth)
                value_holder.append(plan_value)
            value = np.max(value_holder)

            R = torch.FloatTensor([value])





        value_loss = 0
        reward_loss = 0

        R = Variable(R)

        #Generate all data points by prediction
        for index, state in enumerate(states[:-1]):
            upper_bound = min(index + args.predict_step, len(actions))
            enc_state = model.encode(Variable(state.unsqueeze(0)))
            model.predict(enc_state, actions[index : upper_bound], index)

                


        R_vector = []
        R_vector.append(R.data.numpy()[0])
        dp_count = 0

        for i in reversed(range(len(rewards))):
            #the target value, the one you want to approximate at time i
            R = args.gamma * R + rewards[i] 
            R_vector.append(R.data.numpy()[0])

            #get all the approximate values            
            for p in model.data_buffer[i + 1]:
                dp_count += 1
                value_loss = value_loss + 0.5 * (R - p[0]).pow(2)
                reward_loss = reward_loss + 100 * (rewards[i] - p[1]).pow(2)



        optimizer.zero_grad()

        if type(value_loss) != int and type(reward_loss) != int:
            (args.value_loss_coef * value_loss + reward_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

            ensure_shared_grads(model, shared_model)
        
            optimizer.step()

            out_v_loss = value_loss.data.numpy()[0, 0] / dp_count
            out_r_loss = reward_loss.data.numpy()[0, 0] / dp_count
        
            if rank == 0 and episode_count % 200 == 0:
                print('v_loss', np.sqrt(out_v_loss / 0.5), 'r_loss', np.sqrt(out_r_loss / 100))

#            print('actions: ', actions)
#            print('rewards: ', rewards)
#            print('Rs: ', R_vector)
#            print('predictions: ', model.data_buffer)
#            print('out_v_loss', out_v_loss)
#            print('out_r_loss', out_r_loss)
#            sys.exit('lalala')        





        model.data_buffer.clear()



#        if rank == 0 and episode_count % 20 == 0:
#            print('value_loss', np.mean(avg_value_loss))
