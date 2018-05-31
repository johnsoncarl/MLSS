import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):

    #Loading the dataset into the train and test tensors
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        # Create a neural network of your choice
        # You can begin with 2 hidden layers and one output layer 
        # With 200 units for each hidden layer and 10 output units
        # insert ReLU activations between the hidden layers
        # and softmax for the output
        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            pass

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer/ try different optimizers
    # here like ADAM AdaGrad Momentum


    # create a loss function use an NLL loss that mimics crossentropy


    # run the main training loop
    # Every iteration over the complete training set is called an epoch
    for epoch in range(epochs):

        train_loss = 0
        # Train over the dataset for each minibatch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            train_loss+=loss
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0]/log_interval))

                train_loss = 0

        # run a test loop
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
            net_out = net(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data[0]
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    create_nn()
