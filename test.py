import time
from collections import deque
import numpy as np
import datetime
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ValuePredictionNetwork


def test(rank, args, shared_model, signal_queue):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    action_size = env.action_space.n

    model = ValuePredictionNetwork(env.observation_space.shape[0], env.action_space, args.predict_step, args.branch_factor)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    reward_holder = []

    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        
        value_holder = []
        for action in range(action_size):
            plan_value = model.plan(model.encode(Variable(state.unsqueeze(0))), action, args.plan_depth)
            value_holder.append(plan_value)
        action = np.argmax(value_holder)



        state, reward, done, _ = env.step(action)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action)
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            if time.time() - start_time > 108000:
                signal_queue.put(True)
                sys.exit('running out of time')
            reward_holder.append(reward_sum)
            if len(reward_holder) == 4:
                reward_holder.pop(0)
            if np.mean(reward_holder) >= 20:
                signal_queue.put(True)
                sys.exit('The test thread has been stopped')

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
