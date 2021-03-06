from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
import dlvc.ops as ops
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Layer 1, images are 32x32
        # Input channels = 3, output channels = 32
        # Padding for stride = 1 is calculated by (K-1)/2
        self._cn1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Layer 2, after pooling images are 16x16
        self._cn2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Layer 3, after pooling images are 8x8
        self._cn3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Layer 4, after pooling images are 4x4 - with padding 6x6
        self._cn4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # Layer 4, after pooling images are 2x2 - with padding 4x4
        self._cn5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self._pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 2 is the kernel size

        # 512*1*1 input features, 2 output features
        self._fc1 = nn.Linear(512, 2)

    def forward(self, data):
        data = F.relu(self._cn1(data))
        data = self._pool(data)
        data = F.relu(self._cn2(data))
        data = self._pool(data)
        data = F.relu(self._cn3(data))
        data = self._pool(data)
        data = F.relu(self._cn4(data))
        data = self._pool(data)
        data = F.relu(self._cn5(data))
        # global average
        data = self._global_avg_pool(data)
        data = data.view(-1, 512*1*1)
        data = self._fc1(data)

        return data

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Check performance of CNN.')

    parser.add_argument('--fpath', default=os.path.join(os.getcwd(), "../datasets/cifar-10-batches-py"),
                        help='Path to data set.')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of batch.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--epoch_number', type=int, default=100, help='Epoch number.')

    args = parser.parse_args()

    path = args.fpath
    num_samples_per_batch = args.batch_size
    learning_rate = args.learning_rate

    epochs = args.epoch_number

    trainingSet = PetsDataset(path, Subset.TRAINING)
    validationSet = PetsDataset(path, Subset.VALIDATION)

    num_batches = int(np.ceil(len(trainingSet)/num_samples_per_batch))

    op = ops.chain([
        ops.hwc2chw(),
        ops.add(-127.5),
        ops.mul(1 / 127.5)
    ])

    cnn_net = CNN()
    #weight decay still has to be implemented
    clf = CnnClassifier(cnn_net, (num_samples_per_batch, 3, 32, 32), 2, lr=learning_rate, wd=0.0)

    stored_train_losses = []
    stored_validation_accuracy = []

    v_batch_gen = BatchGenerator(validationSet, num_samples_per_batch, True, op)
    v_iter_gen = iter(v_batch_gen)
    v_batch = next(v_iter_gen)

    for e in range(0, epochs):
        ac = Accuracy()
        t_batch_gen = BatchGenerator(trainingSet, num_samples_per_batch, True, op)
        t_iter_gen = iter(t_batch_gen)

        losses = []

        for b in range(1, num_batches+1):
            t_batch = next(t_iter_gen)
            current_loss = clf.train(t_batch.data, t_batch.label)
            losses.append(np.float(current_loss))
            predictions = clf.predict(v_batch.data)
            predictions = predictions.detach().numpy()
            ac.update(predictions, v_batch.label)

        losses_np = np.asarray(losses)
        mean_loss = np.mean(losses_np)
        var_loss = np.var(losses_np)
        v_accuracy = ac.accuracy()

        stored_train_losses.append(str(mean_loss))
        stored_validation_accuracy.append(str(ac))

        print("epoch" + str(e))
        print("train loss: " + str(mean_loss) + " +- " + str(var_loss))
        print("val acc: " + str(ac))

    with open('train_losses_log', 'w') as file_loss:
        for i in stored_train_losses:
            file_loss.write(i + '\n')

    with open('validation_accuracy_log', 'w') as file_accuracy:
        for i in stored_validation_accuracy:
            file_accuracy.write(i + '\n')
