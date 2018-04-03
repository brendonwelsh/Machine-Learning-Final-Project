import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import financial_data as fd
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    def profile(x): return x


class prediction_model:
    '''
    class that holds RNN model
    trained using financial data class
    used to predict stocks prices 
    code based off some of the work by Sean Robertson
    (http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
    '''
    @profile
    def __init__(self, input_size=10, hidden_size=128, output_size=1, learning_rate=0.0005):
        """[constructor for class]

        Keyword Arguments:
            input_size {int} -- [# of past samples used in RNN] (default: {10})
            hidden_size {int} -- [number of hidden units in RNN] (default: {128})
            output_size {int} -- [# of samples predicted] (default: {1})
            learning_rate {float} -- [ML parameter for model] (default: {0.0005})
        """
        self.rnn = RNN(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.rnn.parameters(), learning_rate)
        self.data = fd.financial_data(input_size)
        self.criterion = nn.MSELoss()

    @profile
    def train_data(self):
        """[train data will train using the financial data class on 500+ stocks]
        """
        epoch = 15
        print_every = 10000
        plot_every = 10
        total_loss = 0
        n_iters = len(self.data.x_train)
        all_losses = []
        start = time.time()
        x_train = Variable(torch.Tensor(self.data.x_train))
        y_train = Variable(torch.Tensor(self.data.y_train))
        for j in range(0, epoch):
            for i in range(0, n_iters):
                output, loss = self.train(
                    x_train[i].unsqueeze(0), y_train[i].unsqueeze(0))
                total_loss += loss
                if i % print_every == 0:
                    print('%s (%d %d%%) %.4f' %
                          (self.timeSince(start), i, i / n_iters * 100, loss))
                if i % plot_every == 0:
                    all_losses.append(total_loss / plot_every)
                    total_loss = 0

    @profile
    def test_data(self):
        """[uses test data from financial data class and predicts output with various error tolerances]

        Returns:
            [list] -- [int list of accuracy values]
        """
        x_test = Variable(torch.Tensor(self.data.x_test))
        y_test = Variable(torch.Tensor(self.data.y_test))
        y_res = Variable(torch.zeros(len(x_test)))
        for i in range(0, len(x_test)):
            y_res[i] = self.test(x_test[i].unsqueeze(0))
        errs = [.01, .1, 1, 3, 5, 7, 10]
        p_test = []
        for er in errs:
            p_test.append(
                len(((((torch.abs(y_test-y_res)/y_test))*100) < er).nonzero())/len(x_test))
        return p_test

    @profile
    def train(self, input_tensor, output_tensor):
        """[performs training on neural network passing through data]

        Arguments:
            input_tensor {[Pytorch Tensor 1xinput_size]} -- [input sample for rnn]
            output_tensor {[Pytorch Tensor 1]} -- [output sample from data for input tensor]

        Returns:
            output: [float] -- [result of predicting through RNN]
            lost: [float] -- [lost of data used for training]
        """
        hidden = self.rnn.initHidden()
        loss = 0
        for i in range(input_tensor.size()[0]):
            output, hidden = self.rnn.forward(input_tensor[i], hidden)
            loss += self.criterion(output, output_tensor[i])
            self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss.data[0] / input_tensor.size()[0]

    @profile
    def test(self, input_test):
        """[tests out given input tensor through data to get output]

        Arguments:
            input_test {[Pytorch Tensor 1 x input_size]} -- [input sample for rnn]

        Returns:
            [float] -- [predicted sample of data]
        """
        hidden = self.rnn.initHidden()
        for i in range(input_test.size()[0]):
            output, hidden = self.rnn.forward(input_test[i], hidden)
        return output

    @profile
    def timeSince(self, since):
        """[returns time since input time]

        Arguments:
            since {[float]} -- [time value at begining from time]

        Returns:
            [str] -- [formatted minute second time]
        """
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


class RNN(nn.Module):
    '''
    Define custom RNN module which subclasses
    nn.module
    '''

    def __init__(self, input_size, hidden_size, output_size):
        """[constructor for class]

        Arguments:
            input_size {[uint]} -- [number of inputs used for output prediction]
            hidden_size {[uint]} -- [number of hidden layers in neural network]
            output_size {[uint]} -- [number of samples used for prediction]
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """[passes through RNN to give output and hidden layer]

        Arguments:
            input {[Pytorch Tensor 1 x input_size]} -- [input sample for rnn]
            hidden {[PyTorch Tensor]} -- [past hidden value layer values]

        Returns:
            [output] -- [output sample from prediction]
            [hidden] -- [hidden value numbers]
        """

        combined = torch.cat((input.unsqueeze(0), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        """[initialize the hidden layer values]

        Returns:
            [Pytorch Tensor] -- [zeros hidden value layer]
        """
        return Variable(torch.zeros(1, self.hidden_size))


if __name__ == '__main__':
    mod = prediction_model()
    mod.train_data()
    mod.test_data()
