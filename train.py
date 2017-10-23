import torch
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


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, shared_FPN, retro_step, FPN_optimizer, optimizer=None):




    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    FPN = FramePredictionNetwork()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)


    model.train()
    FPN.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    episode_count = 0

    previous_action = 0
    previous_reward = 0

    pretrain_step = 1000

    worker_step = 0

    while True:
        # Sync with the shared model
        retro_buffer = []
        model.load_state_dict(shared_model.state_dict())
        FPN.load_state_dict(shared_FPN.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []



        sum_rewards = 0

        episode_count += 1

        history_buffer = []
        next_ob_buffer = []
        reward_buffer = []

        


        for step in range(args.num_steps):

            episode_length += 1
            worker_step += 1

            value, logit, (hx, cx), conv4 = model((Variable(state.unsqueeze(0)),
                                            (hx, cx)))



            prob = F.softmax(logit)

            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)


            predicted_value_holder = []
            lstm_state_holder = []

            if len(retro_buffer) == retro_step and worker_step < pretrain_step:
                for a in range(env.action_space.n):
                    a = previous_action
                    one_action = np.full(retro_buffer[0].shape, a)
                    one_piece = copy.deepcopy(retro_buffer)
                    one_piece.append(one_action)
                    one_piece = np.vstack(one_piece)
                    one_piece = np.reshape(one_piece, (3, 16, 24))
                    one_piece = Variable(torch.FloatTensor(one_piece).unsqueeze(0))
                    predicted_cv, predicted_reward = FPN(one_piece)
                    value, temp_state = model.midway((predicted_cv, (hx, cx)))
                    predicted_value_holder.append(value.data.numpy() + predicted_reward.data.numpy())
                    lstm_state_holder.append(temp_state)
                
                action = np.argmax(predicted_value_holder)




            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.numpy())

        
                


            if len(retro_buffer) == retro_step:
                a = previous_action
                one_action = np.full(retro_buffer[0].shape, a)
                one_piece = copy.deepcopy(retro_buffer)
                one_piece.append(one_action)
                one_piece = np.vstack(one_piece)
                one_piece = np.reshape(one_piece, (3, 16, 24))

                history_buffer.append(one_piece)                                
                next_ob_buffer.append(conv4.data.numpy())
                reward_buffer.append(previous_reward)
                
            if len(retro_buffer) == retro_step:
                retro_buffer.pop(0)

            retro_buffer.append(conv4.data.numpy())                


                
            done = done or episode_length >= args.max_episode_length            

            sum_rewards += reward
        
            reward = max(min(reward, 1), -1)

            previous_action = action.numpy()
            previous_reward = reward

            if done:
                if len(retro_buffer) == retro_step:
                    state = torch.from_numpy(state)                
                    _, _, _, conv4 = model((Variable(state.unsqueeze(0)),
                                            (hx, cx)))
                    a = action.numpy()
                    one_action = np.full(retro_buffer[0].shape, a)
                    one_piece = copy.deepcopy(retro_buffer)
                    one_piece.append(one_action)
                    one_piece = np.vstack(one_piece)
                    one_piece = np.reshape(one_piece, (3, 16, 24))
                    history_buffer.append(one_piece)                                
                    next_ob_buffer.append(conv4.data.numpy())
                    reward_buffer.append(previous_reward)
            
                episode_length = 0
                state = env.reset()
                retro_buffer = []



            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break


        R = torch.zeros(1, 1)
        if not done:
            value, _, _ , _= model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        FPN_loss_func = torch.nn.MSELoss()
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



        if len(history_buffer) > 0:

            next_ob_buffer = np.vstack(next_ob_buffer)
            reward_buffer = np.vstack(reward_buffer)

            one_past = Variable(torch.FloatTensor(np.reshape(history_buffer, (-1, 3, (retro_step + 1) * 4, 24))))


            one_true_ob = Variable(torch.FloatTensor(next_ob_buffer))

            one_true_reward = Variable(torch.FloatTensor(reward_buffer))


            predicted_ob, predicted_reward = FPN(one_past)
        
            ob_loss = FPN_loss_func(predicted_ob, one_true_ob)

            reward_loss = FPN_loss_func(predicted_reward, one_true_reward)

            FPN_optimizer.zero_grad()

            total_loss = ob_loss + reward_loss

            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm(FPN.parameters(), args.max_grad_norm)

            ensure_shared_grads(FPN, shared_FPN)

            FPN_optimizer.step()


        optimizer.zero_grad()


        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)


        ensure_shared_grads(model, shared_model)

        

        optimizer.step()

#        if rank == 0:
#            print('observation_loss', np.sqrt(ob_loss.data.numpy() * 288))

