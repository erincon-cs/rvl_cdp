import sys
import time
import os
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from torch import optim, nn as nn
from torch.utils.data import DataLoader
import torch.distributions as tdist

from torch.autograd import Variable

from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, model, train_dataset, valid_dataset=None, test_dataset=None,
                 summary_path=None, criterion=None):
        self.model = model.train()
        self.training_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.trained = False

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        self.max_score = 0

        # creates writer2 object with auto generated file name if the summary_path is set to None,
        # the dir will be something like 'runs/Aug20-17-20-33'
        if summary_path is not None and not os.path.exists(summary_path):
            print("Path {} does not exist!".format(summary_path))
            os.makedirs(summary_path)

        self.writer = SummaryWriter(summary_path)

    def _data_loop(self, data_loader, nb_epochs, criterion, minibatch_size=129, network_optimizer=None,
                   keep_preds=False):
        total_loss = []
        running_time = 0.0

        y = []
        y_true = []
        mean = torch.tensor([0.0])
        var = torch.tensor([1.0])

        if torch.cuda.is_available():
            self.model.cuda()

        iteration = 0
        running_loss = 0.0

        for i in range(1, nb_epochs + 1):
            print("Epoch: {}".format(i))

            for minibatch_i, samples in enumerate(data_loader):
                iteration += 1
                start = time.time()
                image, labels = samples["image"], samples["label"]

                if torch.cuda.is_available():
                    image = image.cuda()
                    mean = mean.cuda()
                    var = var.cuda()

                if network_optimizer is not None:
                    network_optimizer.zero_grad()
                    self.model.train()

                output = self.model(image)

                if keep_preds:
                    y_true.append(labels.cpu().argmax(dim=1).numpy())

                if torch.cuda.is_available():
                    labels = labels.cuda()

                labels = Variable(labels)
                instance_likelihood = criterion(output, labels.argmax(dim=1))
                self.writer.add_scalar("training instance likelihood", instance_likelihood, iteration)

                loss = len(self.training_dataset) * instance_likelihood

                if self.model.kls is not None:
                    kl = sum(self.model.kls)
                    self.writer.add_scalar("training kl", kl, iteration)

                    loss += kl

                if network_optimizer is not None:
                    loss.backward()
                    network_optimizer.step()

                loss = loss.item()

                total_loss.append(loss)

                running_loss += loss

                end = time.time()
                running_time += (end - start)

                avg_loss = running_loss / (minibatch_size * iteration)

                avg_mb_time = running_time / (minibatch_i + 1)
                sys.stdout.write("\r" + "running loss: {0:.5f}".format(avg_loss) + \
                                 " - average time minibatch: {0:.2f}s".format(avg_mb_time))

                self.writer.add_scalar("training loss", avg_loss, iteration)
                sys.stdout.flush()

                if keep_preds:
                    preds = self.model.predict(image)
                y.append(preds)

                if keep_preds:
                    y = np.hstack(y)
                y_true = np.hstack(y_true)

                if self.valid_dataset and network_optimizer is not None:
                    accuracy, valid_loss = self.evaluate(self.valid_dataset)

                self.writer.add_scalar("valid_loss", valid_loss, i)
                self.writer.add_scalar("valid_accuracy", accuracy, i)

                print("\nEpoch {} loss: {}".format(i, np.mean(total_loss)))

        return total_loss, y, y_true


def fit(self, nb_epochs=10, learning_rate=0.0001,
        num_workers=3, minibatch_size=64, verbose=True):
    self.trained = True

    network_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(self.training_dataset, batch_size=minibatch_size,
                             shuffle=True, num_workers=num_workers)
    training_loss, _, _ = self._data_loop(data_loader, nb_epochs, criterion, minibatch_size, network_optimizer,
                                          keep_preds=False)


def evaluate(self, dataset, data_loader=None, minibatch_size=64):
    if not self.trained:
        raise UserWarning("Model is not trained yet!")

    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=minibatch_size,
                                 shuffle=False, num_workers=3)

    print("Evaluating on {} examples".format(len(data_loader)))

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        self.model.cuda()

    running_time, avg_loss = 0, 0

    with torch.no_grad():
        training_loss, y, y_true = self._data_loop(data_loader, 1, criterion, minibatch_size,
                                                   network_optimizer=None, keep_preds=True)
        accuracy = accuracy_score(y, y_true)

        print("\nAccuracy: {0:.2f}".format(accuracy))

        if accuracy > self.max_score:
            self.max_score = accuracy
            self.model.save("best_model.model")

        return accuracy, avg_loss
