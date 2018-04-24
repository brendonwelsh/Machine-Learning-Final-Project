import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, pm=None):
        super(DQN, self).__init__()
        hidden_size = 120  # Random Parameter that can be tuned
        actions = 3  # 3 Different Actions Buy Sell Hold
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(len(pm.data.x_test[1]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, actions)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x