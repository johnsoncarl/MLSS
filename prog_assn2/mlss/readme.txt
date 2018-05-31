Programming Assignment 2 - Classification using Neural Networks
-----------------------------------------------------------------


You have to pick a dress for your special someone. We know you'd rather want to see them without a dress, but let's just maintain decency here and gift them one. Except that you can only look at black and white samples of dresses. You being a noob at making fashion choices and clothing, don't know which type of clothing is which. So, you decide to let 'AI' do the job for you. You decide to train a deep neural network classify the images of clothes into categories to help you out.

We will be using the fashionMNIST dataset here. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

Create a neural network using instructions given in the skeleton code. You are free to play around with the hyperparameters (batch_size, learning rate etc.)once you have a basic version running. Try using different optimizers and regularizations like dropout.

The skeleton code is provided in pytorch_nn_skeleton.py. Try to understand how it is supposed to work, and code up the parts that have been left for you to fill . Record your observations (training / test accuracy / training loss) for changes in hyperparameters, regularization methods that you use or optimizers. Also give reasons for any pattern that you observe among the observations.

We expect you to use the PyTorch Library. Installation instructions are provided here: https://pytorch.org/#pip-install-pytorch.
You are requested to refer to the tutorials to get a better understanding of how to code using pytorch.

Deliverables: Completed code and a small report (<= 1 page) highlighting the observations and conclusions.

Deadline: Thursday 31st May, 11:59 pm.
