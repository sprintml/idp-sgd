import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray

from idp_sgd.training_utils.evaluate import accuracy
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.optim.optimizer import Optimizer

high_noise_multiplier = 1.1
low_noise_multiplier = 0.55
max_grad_norm = 1.0
DELTA = 1e-5


def train_step(model: Module, optimizer: Optimizer, data: Tensor,
               target: Tensor, cuda: bool,
               pp_max_grad_norms: ndarray = None) -> float:
    """Execute one gradient descent step.

    Args:
        model (Module): pytorch module to be trained
        optimizer (Optimizer): pytorch optimizer which updates the model
        data (Tensor): features of the training data in the current batch
        target (Tensor): labels of the training data in the current batch
        cuda (bool): boolean that specifies if the gpu is used
        pp_max_grad_norms (ndarray): clipping thresholds of samples in the
            current mini-batch

    Returns:
        float: loss value of the model on the current batch
    """
    model.train()
    if cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    if pp_max_grad_norms is not None:
        optimizer.step(pp_max_grad_norms=pp_max_grad_norms)
    else:
        optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def weighting_train_epoch(*, loader, cuda, model, budget_index, lr=0.05):
    """train epoch with weighting

    Args:
        loader (_type_): _description_
        cuda (_type_): _description_
        model (_type_): _description_
        budget_index (_type_): _description_
        lr (_type_): _description_

    Returns:
        losses : list of losses
    """
    losses = []
    for batch_id, (data, target, index) in enumerate(loader):
        print('batch_id: ', batch_id)
        # if batch_id == 2000:
        #     break
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.cross_entropy(output, target)
        losses.append(loss.item())
        loss.backward()  # get gradients

        ### here is where the DP happens
        for p in model.parameters():

            # perform clipping (in place)
            torch.nn.utils.clip_grad_norm_(p, max_grad_norm)

            # perform noise addition according to index:
            # print('index: ', index, ' index shape: ', index.shape)
            noise_multiplier = high_noise_multiplier if budget_index[
                index] == 0 else low_noise_multiplier

            noise = torch.normal(mean=0.,
                                 std=noise_multiplier * max_grad_norm,
                                 size=p.shape)
            if cuda:
                noise = noise.cuda()
            p.grad += noise

            # This is what optimizer.step() does - now we do it manually
            prod = torch.mul(p.grad, lr)
            torch.sub(p, prod)  # manual gradient descent
            # p.grad.zero_() # should be reset. But after the subtraction, the gradient is None anyways...
    return losses


def train_epoch_standard(*, loader, cuda, model, optimizer):
    """train epoch without weighting

    Args:
        loader (_type_): _description_
        cuda (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_

    Returns:
        losses : list of losses
    """
    model.train()
    losses = []
    for _, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.cross_entropy(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


def train_epoch(weighting, model, loader, optimizer, cuda, budget_index):
    """Train a given model on a given dataset using a given optimizer for one epoch.
    weighting: boolean: are we using the weighting scheme
    budget_index: array 0 means use low_budget (high noise), 1 means use high_budget (low noise)"""
    model.train()
    if weighting:
        if budget_index is None:
            raise Exception('budget_index is None, but weighting is True')
        losses = weighting_train_epoch(loader=loader,
                                       cuda=cuda,
                                       model=model,
                                       budget_index=budget_index)
    else:
        losses = train_epoch_standard(loader=loader,
                                      cuda=cuda,
                                      model=model,
                                      optimizer=optimizer)

    train_loss = np.mean(losses)
    return train_loss


def train(
        *,
        model,
        weighting=False,
        upsampling=False,
        optimizer,
        cuda,
        budget_index=None,  # TODO - WHAT IS BUDGET INDEX?
        epochs,
        train_loader,
        test_loader,
        privacy_engine,
        seed):
    """train the model with the weighting algorithm
    
    # TODO - IMPROVE THIS DOCSTRING


    Args:
        model (_type_): _description_
        weighting (_type_): _description_
        upsampling (_type_): _description_
        optimizer (_type_): _description_
        cuda (_type_): _description_
        budget_index (_type_): _description_
        epochs (_type_): _description_
        train_loader (_type_): _description_
        test_loader (_type_): _description_
        privacy_engine (_type_): _description_
        seed (_type_): _description_

    Returns:
        res: result dictionary 
    """
    losses = []
    accuracies = []
    epsilons = []
    for epoch in range(epochs):
        train_loss = train_epoch(weighting=weighting,
                                 model=model,
                                 loader=train_loader,
                                 optimizer=optimizer,
                                 cuda=cuda,
                                 budget_index=budget_index)
        _, acc = accuracy(model=model, loader=test_loader, cuda=cuda)
        epsilon = privacy_engine.get_epsilon(DELTA)

        print(
            f'seed: {seed},\tepoch: {epoch},\tloss: {round(float(train_loss), 4)},\taccuracy: {round(acc, 2)},  '
            f'\tepsilon: {round(epsilon, 4)}')

        losses.append(train_loss)
        accuracies.append(acc)
        epsilons.append(epsilon)

    return {
        'weighting': weighting,
        'upsampling': upsampling,
        'seed': seed,
        'epoch': list(np.arange(epochs) + 1),
        'loss': losses,
        'accuracy': accuracies,
        'epsilon': epsilons,
    }
