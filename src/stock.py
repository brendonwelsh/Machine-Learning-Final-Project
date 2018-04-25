import reinforcement_learning
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np

rl = reinforcement_learning.ReinforcementLearning(
    GAMMA=0.5, NUM_EPISODES=3, hidden_layer=256, EPS_START=0.5)
rl.do_reinforcement_learning()
time_series = rl.fd.norm_data_ls[rl.fd.ticker_ls.index(rl.TICKER)]
rl.plot_episode(time_series, 1000)
plt.plot(rl.reward_list)
plt.plot(rl.episode_list)
