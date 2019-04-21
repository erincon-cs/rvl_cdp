import sys
import time
import os
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score

from torch import optim, nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

_metrics = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score
}


def _get_metric(metric_name: str):
    metric_name = metric_name.lower().strip()

    if metric_name not in _metrics:
        raise ValueError("Metric name {} is not defiend!".format(metric_name))

    return _metrics.get(metric_name)


class Trainer:
    def __init__(self, model, train_dataset, valid_dataset=None, test_dataset=None,
                 summary_path=None, criterion=None, weight_histograms=True, model_path=None):
        self.model = model
        self.training_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.trained = False
        self.summary_path = summary_path
        self.weight_histograms = weight_histograms
        self.iteration = 0
        self.model_path = model_path if model_path else self.model.name

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        self.max_score = 0

        # creates writer2 object with auto generated file name if the summary_path is set to None,
        # the dir will be something like 'runs/Aug20-17-20-33'
        if summary_path is not None and not os.path.exists(summary_path):
            print("Path {} does not exist!".format(summary_path))
            os.makedirs(summary_path)

        self.writer = SummaryWriter(summary_path)

    def _data_loop(self, data_loader, nb_epochs, criterion, minibatch_size=128, network_optimizer=None,
                   keep_preds=False, scale_likelihood=True):
        total_loss = []
        running_time = 0.0

        y = []
        y_true = []
        mean = torch.tensor([0.0])
        var = torch.tensor([2.5])

        if torch.cuda.is_available():
            self.model.cuda()

        running_loss = 0.0

        mode = "training" if network_optimizer is not None else "valid"

        for i in range(1, nb_epochs + 1):
            print("Epoch: {}".format(i))

            for minibatch_i, samples in enumerate(data_loader):
                self.iteration += 1
                start = time.time()
                image, labels = samples["image"], samples["label"]

                if torch.cuda.is_available():
                    image = image.cuda()
                    mean = mean.cuda()
                    var = var.cuda()

                if network_optimizer is not None:
                    network_optimizer.zero_grad()

                output = self.model(image)

                if keep_preds:
                    y_true.append(labels.cpu().argmax(dim=1).numpy())

                if torch.cuda.is_available():
                    labels = labels.cuda()

                labels = Variable(labels)
                instance_likelihood = criterion(output, labels.argmax(dim=1))

                if scale_likelihood:
                    instance_likelihood *= len(self.training_dataset)

                self.writer.add_scalar("{} instance likelihood".format(mode), instance_likelihood, self.iteration)
                loss = instance_likelihood

                if self.model._kl is not None:
                    loss *= len(self.training_dataset)

                if self.model._kl is not None:
                    kl = self.model._kl
                    self.writer.add_scalar("{} kl".format(mode), kl, self.iteration)

                    loss += kl

                if network_optimizer is not None:
                    loss.backward()
                    network_optimizer.step()

                loss = loss.item()

                total_loss.append(loss)

                running_loss += loss

                end = time.time()
                running_time += (end - start)

                avg_loss = running_loss / (self.iteration * minibatch_size)
                avg_mb_time = running_time / (minibatch_i + 1)
                sys.stdout.write("\r" + "running loss: {0:.5f}".format(avg_loss) + \
                                 " - average time minibatch: {0:.2f}s".format(avg_mb_time))

                self.writer.add_scalar("{} loss".format(mode), avg_loss, self.iteration)
                sys.stdout.flush()

                if keep_preds:
                    preds = self.model.predict(image)
                    y.append(preds)

                if self.iteration % 10 == 0 and self.weight_histograms:
                    for name, param in self.model._classifier.named_parameters():
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.iteration)

            if keep_preds:
                y = np.hstack(y)
                y_true = np.hstack(y_true)

            if self.valid_dataset and network_optimizer is not None:
                accuracy, valid_loss = self.evaluate(self.valid_dataset)
                print("\nValid accuracy: {0:.2f} | Valid loss: {0:.2f}".format(accuracy, valid_loss))

                self.writer.add_scalar("valid_loss", valid_loss, self.iteration)
                self.writer.add_scalar("valid_accuracy", accuracy, self.iteration)

                print("\nEpoch {} loss: {}".format(i, np.mean(total_loss)))

        return total_loss, y, y_true

    def fit(self, nb_epochs=10, learning_rate=0.0001,
            num_workers=3, minibatch_size=128, verbose=True):
        self.trained = True

        network_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        data_loader = DataLoader(self.training_dataset, batch_size=minibatch_size,
                                 shuffle=True, num_workers=num_workers)
        self.model.train()
        training_loss, _, _ = self._data_loop(data_loader, nb_epochs, criterion, minibatch_size, network_optimizer,
                                              keep_preds=False)

        self.iteration = 0

    def evaluate(self, dataset, data_loader=None, minibatch_size=128, save=False,
                 metrics=None, metric_check="accuracy"):

        if metrics is None:
            metrics = {"accuracy"}

        if metric_check not in metrics:
            raise ValueError("Metric check {} is not in the defined metrics: {}!".format(metric_check, metrics))

        if not self.trained:
            raise UserWarning("Model is not trained yet!")

        if data_loader is None:
            data_loader = DataLoader(dataset, batch_size=minibatch_size,
                                     shuffle=False, num_workers=3)

        if self.max_score == 0:
            self.model.save(self.model_path)

        print("Evaluating on {} examples\n".format(len(data_loader)))

        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.model.cuda()

        running_time, avg_loss = 0, 0

        with torch.no_grad():
            self.model.eval()
            training_loss, y, y_true = self._data_loop(data_loader, 1, criterion, minibatch_size,
                                                       network_optimizer=None, keep_preds=True)

        metrics_results = {metric_name: _get_metric(metric_name)(y, y_true) for metric_name in metrics}

        max_metric = metrics_results.get(metric_check)

        if max_metric > self.max_score and save:
            self.max_score = max_metric

            self.model.save(self.model_path)

        return metrics_results, avg_loss
