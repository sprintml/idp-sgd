from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from sklearn.metrics import precision_score, recall_score
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import DataLoader


def evaluate(model: Module, loader: DataLoader, cuda: bool) -> Tensor:
    output = []
    with torch.no_grad():
        for data in loader:
            if cuda:
                data = data.cuda()
            data = Variable(data)
            output.append(model(data))
    output = torch.stack(output).detach().cpu()
    output = torch.squeeze(output)
    return output


def accuracy(model: Module, loader: DataLoader, cuda: bool,
             with_idx: bool = False) -> Tuple[float, float]:
    """Evaluate the accuracy of a given model on a given dataset."""
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for batch in loader:
            if with_idx:
                data, target, _ = batch
            else:
                data, target = batch
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            losses.append(F.cross_entropy(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    eval_loss = float(np.mean(losses))
    return eval_loss, 100. * correct / len(loader.dataset)


def accuracy_by_class(model: Module, loader: DataLoader, cuda: bool) -> ndarray:
    """
    Evaluate the class-specific accuracy of a given model on a given dataset.

    Returns:
        A 1-D numpy array of length L = num-classes, containg the accuracy for
        each class.
    """
    model.eval()
    n_classes = len(np.unique(loader.dataset.targets))
    correct = np.zeros(n_classes, dtype=np.int64)
    wrong = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
            target = target.data.cpu().numpy().astype(np.int64)
            for label, pred in zip(target, preds):
                if label == pred:
                    correct[label] += 1
                else:
                    wrong[label] += 1
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    return 100. * correct / (correct + wrong)


def evaluate_precision(model: Module, loader: DataLoader,
                       cuda: bool) -> Tuple[float, ndarray]:
    """Evaluate the precision of a given model on a given dataset."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            predictions.append(output.data.max(1, keepdim=True)[1][0])
            targets.append(target)
    y_pred = torch.stack(predictions).detach().cpu()
    y_true = torch.stack(targets).detach().cpu()
    precision = precision_score(
        y_pred=y_pred,
        y_true=y_true,
        average='micro',
    )
    by_class = precision_score(
        y_pred=y_pred,
        y_true=y_true,
        average=None,
    )
    return precision, by_class


def evaluate_recall(model: Module, loader: DataLoader,
                    cuda: bool) -> Tuple[float, ndarray]:
    """Evaluate the recall of a given model on a given dataset."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            predictions.append(output.data.max(1, keepdim=True)[1][0])
            targets.append(target)
    y_pred = torch.stack(predictions).detach().cpu()
    y_true = torch.stack(targets).detach().cpu()
    recall = recall_score(
        y_pred=y_pred,
        y_true=y_true,
        average='micro',
    )
    by_class = recall_score(
        y_pred=y_pred,
        y_true=y_true,
        average=None,
    )
    return recall, by_class
