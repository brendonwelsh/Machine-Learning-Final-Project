import dqn
import replay_memory
import prediction_model
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import pixiedust
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import pdb
import config



class ReinforcementLearning:
    '''
    Class that performs reinforcement learning.
    
    '''
    
    def __init__(self):
    
    
        is_ipython = True
        plt.ion()
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor
        self.pm = prediction_model.prediction_model()
        self.state=self.pm.data.norm_data_ls[self.pm.data.ticker_ls.index(config.TICKER)].Close
        self.date=self.pm.data.norm_data_ls[self.pm.data.ticker_ls.index(config.TICKER)].date



        self.policy_net = dqn.DQN(self.pm)
        self.target_net = dqn.DQN(self.pm)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = replay_memory.ReplayMemory(config.REPLAY_MEMORY_CAPACITY)
        self.steps_done = 0
        self.episode_durations = []


    def select_action(self, state):
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * self.steps_done / config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.policy_net(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(0)[1]
        else:
            return torch.LongTensor([random.randrange(3)])

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(1)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        transitions = self.memory.sample(config.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = config.TRANSITION(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        non_final_next_states = non_final_next_states.view(128,10)

        state_batch = Variable(torch.cat(batch.state))
        state_batch = state_batch.view(128,10)
        action_batch = Variable(torch.cat(batch.action))
        action_batch = action_batch.view(128,1)
        reward_batch = Variable(torch.cat(batch.reward))
        reward_batch = reward_batch.view(128,1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(config.BATCH_SIZE).type(torch.Tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def step(self, action, cur_price, next_price, days):
        #Write Logic to determine if at end of time series
        #Write Logic to figure out reward for action(% Profit?)

        a = action == 0
        b = action == 1
        c = action == 2

        if (a):
            #BUY
            if ((next_price - cur_price) > 0):
                reward = 10
            else:
                reward = -10

        if (b):
            #SELL
            if ((next_price - cur_price) > 0):
                reward = -10
            else:
                reward = 10

        if (c):
            if ((next_price - cur_price) > 0):
                reward = 0
            else:
                reward = -5

        if (days > 500):
            done = True
        else:
            done = False

        return reward, done
    
    def do_reinforcement_learning(self):
        for i_episode in range(config.NUM_EPISODES):
            self.state = torch.Tensor(self.pm.data.x_test[1])
            self.state.unsqueeze(0)
            for t in count():
                # Select and perform an action
                action = self.select_action(self.state)
                reward, done = self.step(action[0], self.pm.data.x_test[t][9], self.pm.data.y_test[t], t)
                reward = torch.Tensor([reward])

                next_state = torch.Tensor(self.pm.data.x_test[(t+1)])
                next_state.unsqueeze(0)
                # Store the transition in memory
                self.memory.push(self.state, action, next_state, reward)

                # Move to the next state
                self.state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

            # Update the target network
            if i_episode % config.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        plt.ioff()
        plt.show()

