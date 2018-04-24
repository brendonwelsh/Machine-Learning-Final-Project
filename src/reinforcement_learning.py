import dqn
import replay_memory
import financial_data
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import pdb
from collections import namedtuple


class ReinforcementLearning:
    '''
    Class that performs reinforcement learning.

    '''

    def __init__(self, input_size=10, TICKER='MSFT', BATCH_SIZE=128, GAMMA=0.999, EPS_START=0.9, EPS_END=0.05,
                 EPS_DECAY=200, TARGET_UPDATE=10, REPLAY_MEMORY_CAPACITY=10000, NUM_EPISODES=1, hidden_layer=120, actions=3):

        self.TICKER = TICKER
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.NUM_EPISODES = NUM_EPISODES
        self.fd = financial_data.financial_data(input_size)
        self.date = self.fd.norm_data_ls[self.fd.ticker_ls.index(TICKER)].date
        self.policy_net = dqn.DQN(input_size, hidden_layer, actions)
        self.target_net = dqn.DQN(input_size, hidden_layer, actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = replay_memory.ReplayMemory(REPLAY_MEMORY_CAPACITY)
        self.steps_done = 0
        self.episode_durations = []
        self.actions = actions
        self.input_size = input_size
        self.action_index = ['Buy', 'Sell', 'Hold']
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + \
            (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.policy_net(Variable(state, volatile=True).type(
                torch.FloatTensor)).data.max(0)[1]
        else:
            return torch.LongTensor([random.randrange(self.actions)])

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        TRANSITION = namedtuple(
            'Transition', ('state', 'action', 'next_state', 'reward'))
        batch = TRANSITION(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        non_final_next_states = non_final_next_states.view(128, 10)

        state_batch = Variable(torch.cat(batch.state))
        state_batch = state_batch.view(128, 10)
        action_batch = Variable(torch.cat(batch.action))
        action_batch = action_batch.view(128, 1)
        reward_batch = Variable(torch.cat(batch.reward))
        reward_batch = reward_batch.view(128, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(
            torch.zeros(
                self.BATCH_SIZE).type(
                torch.Tensor))
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(
            expected_state_action_values.data)

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def step(self, action, cur_price, next_price, days, since_buy, price_buy):
        #Write Logic to determine if at end of time series
        #Write Logic to figure out reward for action(% Profit?)

        # action: 1 - Buy, 2 - Sell, 3 - Hold
        # cur_price: current price of the stock (10th value in array)
        # next_price: next price of the stock (y value)
        # days: day count within current episode
        # since_buy: days since the last buy (-1 if no holdings)
        # price_buy: price at the last buy (-1 if no holdings)

        if (action == 0):
            #BUY
            if (price_buy == -1):
                if ((next_price - cur_price) > 0):
                    reward = (next_price - cur_price)*5
                    price_buy = cur_price
                    since_buy = 1
                else:
                    reward = (next_price - cur_price)*5
                    price_buy = cur_price
                    since_buy = 1
            else:
                reward = 0
                price_buy = -1
                since_buy = -1

        elif (action == 1):
            #SELL
            if ((since_buy > 0) and (price_buy > 0)):
                reward = (cur_price - price_buy)*100
                price_buy = -1
                since_buy = -1
            else:
                reward = -1000
                price_buy = -1
                since_buy = -1

        elif (action == 2):
            #HOLD
            if (price_buy == -1):
                if ((next_price - cur_price) > 0):
                    reward = -10
                    price_buy = -1.0
                    since_buy = -1.0
                else:
                    reward = 10
                    price_buy = -1.0
                    since_buy = -1.0
            else:
                if ((next_price - cur_price) > 0):
                    reward = 10*since_buy + (cur_price - price_buy)
                    price_buy = price_buy
                    since_buy = since_buy + 1
                else:
                    reward = -2*since_buy + (cur_price - price_buy)
                    price_buy = price_buy
                    since_buy = since_buy + 1
        

        if (days > 1200):
            done = True
        else:
            done = False

        return reward, done, since_buy, price_buy

    def plot_episode(self, time_series, test_size):
        x, y = self.fd.split_data([time_series])
        state = torch.Tensor(x[0])
        action_ls = []
        x_coord = np.arange(0, test_size)
        for i in range(0, test_size):
            action = self.target_net(Variable(torch.Tensor(x[i])))
            action_ls.append(np.argmax(action.data.numpy()))
        
        loc_buy = np.where(np.array(action_ls)==0)
        loc_sell = np.where(np.array(action_ls)==1)
        loc_hold = np.where(np.array(action_ls)==2)
        
        plt.plot(x_coord, time_series.Close[0:test_size])
        plt.scatter(loc_buy, time_series.Close.values[loc_buy[0]], c = 'g')
        plt.scatter(loc_sell, time_series.Close.values[loc_sell[0]], c = 'r')
        plt.scatter(loc_hold, time_series.Close.values[loc_hold[0]], c = 'y')
        return
    
    
    def do_reinforcement_learning(self):
        for i_episode in range(self.NUM_EPISODES):
            #state = torch.Tensor(self.fd.x_test[0])
            x, y = self.fd.split_data([self.fd.norm_data_ls[self.fd.ticker_ls.index(self.TICKER)]])
            state = torch.Tensor(x[0])
            since_buy = -1
            price_buy = -1
            for t in count():
                
                # Select and perform an action
                action = self.select_action(state)
                #reward, done, since_buy, price_buy = self.step(
                #    action[0], self.fd.x_test[t][self.input_size - 1], self.fd.y_test[t], t, since_buy, price_buy)
                reward, done, since_buy, price_buy = self.step(
                    action[0], x[t][self.input_size - 1], y[t], t, since_buy, price_buy)
                
                reward = torch.Tensor([reward])
                next_state = torch.Tensor(x[t+1])
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    print(f'Episode Done: {i_episode}')
                    break

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
