import argparse
import copy
import getpass
import logging
import math
import numpy as np
import os
import pandas as pd
import time
import torch
from numpy import ndarray
from opacus import PrivacyEngine
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import assign_pp_values
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import List, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from idp_sgd.training_utils import training
from idp_sgd.training_utils.datasets import get_dataset, assign_budgets
from idp_sgd.training_utils.evaluate import accuracy, accuracy_by_class
from idp_sgd.training_utils.models import CIFAR10_CNN
from idp_sgd.training_utils.models import MNIST_CNN
from idp_sgd.training_utils.models import VGG

user = getpass.getuser()

# DEFAULT SETTINGS
n_workers = 6
mode = "run"
DATASET = "MNIST"
save_path = '/mfsnic/projects/idpsgd/'
mode = 'mia'  # debug run mia
dataset = 'MNIST'
individualize = 'clipping'
assign_budget = 'even'
architecture = 'MNIST_CNN'

if mode in ['run', 'mia']:
    ALPHAS = RDPAccountant.DEFAULT_ALPHAS + list(np.arange(
        70, 201, 10)) + list(np.arange(200, 1001, 50)) + list(
            np.arange(1000, 10001, 500))
elif mode == 'debug':
    ALPHAS = RDPAccountant.DEFAULT_ALPHAS
else:
    raise Exception(f'Unknown mode: {mode}.')

parser = argparse.ArgumentParser(description='Baseline Parser')
parser.add_argument(
    '--dname',
    type=str,
    default=dataset,
    help='dataset to be learned',
)
parser.add_argument(
    '--seeds',
    nargs='+',
    type=int,
    # default=[x for x in range(12)],
    default=[0],
    help='keys for reproducible pseudo-randomness',
)
parser.add_argument(
    '--architecture',
    type=str,
    default=architecture,
    help='model architecture to train',
)
parser.add_argument(
    '--accountant',
    type=str,
    default='rdp',
    help='privacy metric to bound DP guarantees',
)
parser.add_argument(
    '--individualize',
    type=str,
    default=individualize,
    help='(i)DP-SGD method ("None", "clipping", "sampling")',
)
parser.add_argument(
    '--log_iteration',
    type=int,
    default=100,
    help='logging frequency',
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.2,
    help='learning rate (parameter of optimizer)',
)
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    help='momentum (parameter of optimizer)',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=30,
    help='number of training epochs',
)
parser.add_argument(
    '--n_workers',
    type=int,
    default=n_workers,
    help='number of CPU cores to pass data to model',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='expected size of mini-batches',
)
parser.add_argument(
    '--max_physical_batch_size',
    type=int,
    default=128,
    help='upper bound on the physical batch size',
)
parser.add_argument(
    '--delta',
    type=float,
    default=1e-5,
    help='target delta in (eps, delta)-DP',
)
parser.add_argument(
    '--budgets',
    nargs='+',
    type=float,
    default=[1., 2., 3.],
    help='target budgets of privacy groups in (eps, delta)-DP '
    'in ascending order',
)
parser.add_argument(
    '--ratios',
    nargs='+',
    type=float,
    default=[0.54, 0.37, 0.09],
    help='relative sizes of privacy groups',
)
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.9,
    help='clipping threshold for DPSGD and default value for '
    'iDP-SGD (of the lowest privacy group for iClipping)',
)
parser.add_argument(
    '--noise_multiplier',
    type=float,
    default=1.396484375,
    help='relation between noise standard deviation and '
    'clipping threshold for DP-SGD',
)
parser.add_argument(
    '--weights',
    nargs='+',
    type=float,
    default=None,
    help='clipping thresholds or sample rates used for iDP-SGD '
    'in ascending order, depending on individualize',
)
parser.add_argument(
    '--adapt_weights_to_budgets',
    type=str,
    default='True',
    choices={'False', 'True'},
    help='if true, the clipping thresholds or sample_rates are '
    'determined such that all given budgets exhaust after '
    'approximately the given number of epochs',
)
parser.add_argument(
    '--use_cuda',
    type=str,
    default='True',
    choices={'False', 'True'},
    help='if use cuda or not',
)
parser.add_argument(
    '--save_path',
    type=str,
    # only use absolute paths!
    default=save_path,
    help='where to save the model, logs, results',
)
parser.add_argument(
    '--mode',
    type=str,
    default=mode,
    choices=['debug', 'run', 'mia'],
    help='the mode of running the code',
)
parser.add_argument('--accuracy_log',
                    type=str,
                    default='accuracy.log',
                    help='The name of the log file to track the accuracy of '
                    'multiple grid search of parameters (batch job array).'
                    ' If set to None, then no log is created.')
parser.add_argument(
    '--assign_budget',
    type=str,
    default=assign_budget,
    choices=['random', 'even', 'per-class'],
    help='The type of budget assignment.',
)
parser.add_argument(
    '--class_budgets',
    nargs='+',
    type=float,
    default=[1.0] * 10,
    help='Per-class budgets for fairness experiments. Only used'
    ' if ´assign_budget´ is ´per-class´.',
)
parser.add_argument('--mia_ndata',
                    type=int,
                    default=25000,
                    help="The number of data points for the mia (membership "
                    "inference attack) models.")
parser.add_argument('--mia_count',
                    type=int,
                    default=1,
                    help="The index of model to create for the membership"
                    " inference attack (mia).")
parser.add_argument(
    '--allow_excess',
    type=str,
    default='False',
    help='If True, training is not terminated due to DP costs.',
)
parser.add_argument(
    '--save_model',
    type=str,
    default='True',
    help='If True, save the model. If False, do not save the model.',
)


def idp_sgd_training(
    private_model: Module, privacy_engine: PrivacyEngine,
    private_loader: DataLoader, test_loader: DataLoader,
    private_optimizer: Optimizer, iterations: int, pp_budgets: ndarray,
    cuda: bool, delta: float, log_iteration: int, args
) -> [
        List[List[float]],
        List[float],
        List[float],
        List[List[float]],
        List[float],
        List[float],
]:
    """Train the given model according to the given hyperparameters."""
    budgets = np.sort(np.unique(pp_budgets))
    accs_train, accs_test, accs_by_class = [], [], []
    losses, times, batch_sizes = [], [], []
    epsilons = [[] for _ in range(privacy_engine.n_groups)]
    private_loader_iter = iter(private_loader)
    last_entry = None
    if privacy_engine.individualize == 'clipping':
        pp_max_grad_norms = assign_pp_values(
            pp_budgets=privacy_engine.pp_budgets,
            values=privacy_engine.weights)
    termination = None
    current_batch_size, i = 0, 0
    start = time.time()
    while termination is None:
        try:
            batch = next(private_loader_iter)
        except StopIteration:
            private_loader_iter = iter(private_loader)
            batch = next(private_loader_iter)
        if privacy_engine.individualize == 'clipping':
            data, target, idx = batch
            batch_pp_max_grad_norms = pp_max_grad_norms[idx]
        else:
            data, target = batch
            batch_pp_max_grad_norms = None
        current_batch_size += len(data)
        scheduler = ReduceLROnPlateau(optimizer=private_optimizer,
                                      mode='min',
                                      factor=0.5,
                                      patience=500)
        loss = training.train_step(model=private_model,
                                   optimizer=private_optimizer,
                                   cuda=cuda,
                                   data=data,
                                   target=target,
                                   pp_max_grad_norms=batch_pp_max_grad_norms)
        scheduler.step(loss)
        # ensure privacy history was changed (whole mini-batch was processed)
        if len(privacy_engine.accountant.history
               ) > 0 and last_entry != privacy_engine.accountant.history[-1]:
            if i == iterations - 1:
                termination = 'Terminate: The maximum of iterations is reached!'
            batch_sizes.append(current_batch_size)
            current_batch_size = 0
            last_entry = copy.copy(privacy_engine.accountant.history[-1])
            dp_eps = privacy_engine.get_epsilon(  # faster, more accurate
                delta=delta,
                optimal=True,
                from_prev_alpha=True)
            if isinstance(dp_eps, float):
                dp_eps = [dp_eps]
            if privacy_engine.individualize is not None:
                exhausted_groups = np.arange(
                    privacy_engine.n_groups)[np.asarray(dp_eps) >= budgets]
            else:
                exhausted_groups = [0] if dp_eps[0] >= budgets[0] else []
            if len(exhausted_groups) > 0 and not args.allow_excess:
                termination = f'Terminate: The budgets of groups ' \
                              f'{exhausted_groups} are exhausted!'
            if i % log_iteration == 0 or termination is not None:
                average_batch_size = float(np.mean(batch_sizes))
                batch_sizes = []
                elapsed_time = (times[-1]
                                if times else 0) + time.time() - start
                start = time.time()
                _, train_acc = accuracy(
                    private_model,
                    private_loader,
                    cuda,
                    with_idx=privacy_engine.individualize == 'clipping')
                _, test_acc = accuracy(private_model, test_loader, cuda)
                acc_by_class = list(
                    accuracy_by_class(private_model, test_loader, cuda))
                losses.append(loss)
                accs_train.append(train_acc)
                accs_test.append(test_acc)
                accs_by_class.append(acc_by_class)
                times.append(elapsed_time)
                for group in range(privacy_engine.n_groups):
                    epsilons[group].append(dp_eps[group])
                dp_costs = [
                    round(epsilons[g][-1], 3)
                    for g in range(privacy_engine.n_groups)
                ]
                dp_costs = dp_costs[0] if len(dp_costs) == 1 else dp_costs
                alphas = [round(accountant.alpha, 2) for accountant in
                          privacy_engine.accountant.accountants] \
                    if privacy_engine.accountant.mechanism() == 'idp' \
                    else round(privacy_engine.accountant.alpha, 2)
                logging.info(
                    f'iteration: {i},   '
                    f'train accuracy: {round(train_acc, 2)},   '
                    f'test accuracy: {round(test_acc, 2)},   '
                    f'loss: {round(loss, 5)},   DP costs: {dp_costs},   '
                    f'average batch-size: {round(average_batch_size)},   '
                    f'best alpha: {alphas},   time: {round(elapsed_time)}')
            i += 1
    logging.info(termination)
    if cuda:
        # attempt to avoid gpu memory errors in experiment sequences
        torch.cuda.empty_cache()
    return epsilons, accs_train, accs_test, accs_by_class, losses, times


def get_mia_train_set_ppbudgets(dataset_name: str, epochs: int, n_workers: int,
                                batch_size: int, seed: int, args, path: str,
                                pp_budgets):
    assert args.mode == 'mia'

    train_set = get_dataset(dataset_name=dataset_name, test=False)

    save_path_idx = os.path.join(path, f'run{args.mia_count}')
    if not os.path.exists(save_path_idx):
        os.makedirs(save_path_idx)

    img = train_set[0][0]
    save_img = os.path.join(save_path_idx, 'img.npy')
    np.save(file=save_img, arr=img.cpu().numpy())

    # select random data points
    n_data = len(train_set)
    indices = np.arange(n_data, dtype=int)
    np.random.seed(seed=args.mia_count)
    idx = np.random.choice(a=indices, size=args.mia_ndata, replace=False)
    np.random.seed(seed=args.seed)

    # Main part: re-map ppbudgets, data items, and labels for mia.
    pp_budgets = pp_budgets[idx]
    train_set.data = train_set.data[idx]
    train_set.targets = np.array(train_set.targets)[idx]

    save_file = os.path.join(save_path_idx, 'assignment.npy')
    np.save(file=save_file, arr=idx)

    n_data = len(train_set)
    logging.info(
        f'seed: {seed},   '
        f'max_iteration: {int(round(epochs * n_data / batch_size))},   '
        f'1 epoch ~= {int(round(n_data / batch_size))} iterations')

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              shuffle=True,
                              pin_memory=True)
    return train_loader, pp_budgets


def initialize_training(dataset_name: str,
                        cuda: bool,
                        epochs: int,
                        n_workers: int,
                        batch_size: int,
                        seed: int,
                        args,
                        shuffle: bool = True):
    """Initialize the training process with the correct dataloader.
    """
    train_set = get_dataset(dataset_name=dataset_name, test=False, args=args)
    test_set = get_dataset(dataset_name=dataset_name, test=True, args=args)
    n_data = len(train_set)
    logging.info(
        f'seed: {seed},   '
        f'max_iteration: {int(round(epochs * n_data / batch_size))},   '
        f'1 epoch ~= {int(round(n_data / batch_size))} iterations')
    device = 'cuda' if cuda else 'cpu'
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              shuffle=shuffle,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             num_workers=n_workers,
                             shuffle=False,
                             pin_memory=True)
    return device, train_loader, test_loader


def make_private(model, optimizer, accountant, individualize, train_loader,
                 max_grad_norm, noise_multiplier, max_physical_batch_size,
                 weights, adapt_weights_to_budgets, epochs, pp_budgets,
                 target_delta=1e-5) \
        -> [Module, PrivacyEngine, Optimizer, DataLoader]:
    """Make model, optimizers, and train loaders private."""
    privacy_engine = PrivacyEngine(accountant=accountant,
                                   individualize=individualize,
                                   weights=weights,
                                   pp_budgets=pp_budgets)
    if adapt_weights_to_budgets:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private_with_epsilon(module=model,
                                       optimizer=optimizer,
                                       data_loader=train_loader,
                                       target_epsilon=min(pp_budgets),
                                       target_delta=target_delta,
                                       epochs=epochs,
                                       max_grad_norm=max_grad_norm,
                                       optimal=True,
                                       max_alpha=10_000)
    else:
        private_model, private_optimizer, private_loader = privacy_engine \
            .make_private(module=model,
                          optimizer=optimizer,
                          data_loader=train_loader,
                          noise_multiplier=noise_multiplier,
                          max_grad_norm=max_grad_norm)

    if individualize == 'clipping':
        group_params = f'individual max_grad_norms={privacy_engine.weights}'
    elif individualize == 'sampling':
        group_params = f'individual sample_rates={privacy_engine.weights}'
    else:
        group_params = 'no individual parameters'
    logging.info(f'Initializing privacy parameters:   '
                 f'max_grad_norm={max_grad_norm},   '
                 f'sample_rate={1 / len(private_loader)},   '
                 f'noise_multiplier {private_optimizer.noise_multiplier},   '
                 f'{group_params}')

    if max_physical_batch_size > 0:
        with BatchMemoryManager(
                data_loader=private_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=private_optimizer) as memory_safe_private_loader:
            return (private_model, privacy_engine, private_optimizer,
                    memory_safe_private_loader)
    else:
        return private_model, privacy_engine, private_optimizer, private_loader


def run_idp_sgd_experiment(
        model: Module, dataset_name: str, optimizer: Optimizer,
        accountant: str, individualize: Optional[str], cuda: bool, epochs: int,
        n_workers: int, batch_size: int, max_physical_batch_size: int,
        delta: float, log_iteration: int, budgets: List[float],
        ratios: List[float], max_grad_norm: float, noise_multiplier: float,
        weights: List[float], adapt_weights_to_budgets: bool, path: str,
        seed: int, mode: Optional[str], args):
    """Train the given model according to the given hyperparameters.

    Args:
        model (Module): pytorch module to be privately trained
        dataset_name (str): name of the dataset
        optimizer (Optimizer): pytorch optimizer which updates the model
        accountant (str): name of the DP metric used to bound privacy costs
        individualize (str): name of individualization method for (i)DP-SGD
        cuda (bool): boolean that specifies if the gpu is used
        epochs (int): expected frequency each data point updates the model
        n_workers (int): number of cpu cores to help passing data to the gpu
        batch_size (int): number of data points used per gradient descent step
        max_physical_batch_size (int): batch_size that fits into memory
        delta (float): parameter of (epsilon, delta)-DP
        log_iteration (int): frequency of logging training results
        budgets (List): list of privacy budgets for each privacy group
        ratios (List): ratios per privacy group relative to # training data
        max_grad_norm (float): clipping threshold of per-sample gradient norms
        noise_multiplier (float): relation of clipping threshold to noise std
        weights (List): clipping thresholds or sample rates per group in same
            order as ratios and budgets
        adapt_weights_to_budgets (bool): if true, the given noise_multipliers
            or sample_rates are ignored and calculated to align with budgets
        path (str): where to save all our results
        seed (int): initializer for pseudo-random number generation
        mode (dict): the dictionary with program parameters
    """
    device, train_loader, test_loader = initialize_training(
        dataset_name=dataset_name,
        cuda=cuda,
        epochs=epochs,
        n_workers=n_workers,
        batch_size=batch_size,
        seed=seed,
        args=args)
    n_data = len(train_loader.dataset)
    if args.assign_budget == 'per-class':
        args.pp_labels = train_loader.dataset.targets
    pp_budgets = assign_budgets(n_data, budgets, ratios, args=args)
    ratios = [sum(pp_budgets == b) / len(pp_budgets) for b in budgets]

    if args.mode == 'mia':
        train_loader, pp_budgets = get_mia_train_set_ppbudgets(
            dataset_name=dataset_name,
            epochs=epochs,
            n_workers=n_workers,
            batch_size=batch_size,
            seed=seed,
            args=args,
            path=path,
            pp_budgets=pp_budgets)

    (private_model, privacy_engine, private_optimizer,
     private_loader) = make_private(
         model=model,
         optimizer=optimizer,
         accountant=accountant,
         individualize=individualize,
         train_loader=train_loader,
         max_grad_norm=max_grad_norm,
         noise_multiplier=noise_multiplier,
         max_physical_batch_size=max_physical_batch_size,
         weights=weights,
         adapt_weights_to_budgets=adapt_weights_to_budgets,
         epochs=epochs,
         pp_budgets=pp_budgets,
         target_delta=1e-5)
    private_model.to(device)
    if mode == 'debug':
        iterations = 1
    else:
        iterations = int(round(epochs * n_data / batch_size))
    (epsilons, accs_train, accs_test, accs_by_class, losses,
     times) = idp_sgd_training(private_model=private_model,
                               privacy_engine=privacy_engine,
                               private_loader=private_loader,
                               test_loader=test_loader,
                               private_optimizer=private_optimizer,
                               iterations=iterations,
                               pp_budgets=pp_budgets,
                               cuda=cuda,
                               delta=delta,
                               log_iteration=log_iteration,
                               args=args)

    save_path_results = os.path.join(path, 'results.csv')
    if args.mode == 'mia':
        save_path_model = os.path.join(
            path, f'run{args.mia_count}/opacus_model.ckpt')
    else:
        save_path_model = os.path.join(path, 'opacus_model.ckpt')
    logged_iterations = list(
        np.arange(len(losses) - 1) * log_iteration) + [iterations]
    res = {
        'iteration':
        logged_iterations,
        'train_accuracy':
        accs_train,
        'test_accuracy':
        accs_test,
        'loss':
        losses,
        'time':
        times,
        'individualize':
        'standard' if individualize is None else individualize,
        'accountant':
        accountant,
        'seed':
        seed,
        'dataset_name':
        dataset_name,
        'batch_size':
        batch_size,
        'ratios':
        str(ratios),
        'budgets':
        str(budgets),
        'linear_budgets':
        str([math.exp(budget) for budget in budgets]),
        'max_grad_norm':
        max_grad_norm,
        'noise_multiplier':
        private_optimizer.noise_multiplier,
        'weights':
        str(privacy_engine.weights),
        'optimizer':
        private_optimizer.__str__().split(' ')[0],
        'momentum':
        private_optimizer.defaults['momentum'],
        'learning_rate':
        private_optimizer.defaults['lr'],
        'epochs':
        epochs,
        'class_budgets': (str(args.class_budgets)
                          if args.assign_budget == 'per-class' else None),
    }
    for group in range(len(budgets)):
        eps_group = epsilons[0] if individualize is None else epsilons[group]
        res[f'epsilon_{group}'] = eps_group
        res[f'linear_epsilon_{group}'] = [math.exp(eps) for eps in eps_group]
    for c in range(len(accs_by_class[0])):
        res[f'accuracy_{c}'] = list(np.asarray(accs_by_class)[:, c])
    pd.DataFrame(res).to_csv(save_path_results,
                             mode='a',
                             index=False,
                             header=not os.path.isfile(save_path_results))
    if args.save_model == 'True':
        privacy_engine.save_checkpoint(path=save_path_model,
                                       module=private_model,
                                       optimizer=private_optimizer)

    print('args.accuracy_log: ', args.accuracy_log)
    if args.accuracy_log != 'None':
        # Logging from many array batch jobs to a single log file.
        last_accuracy = accs_test[-1]
        file_name = save_path_results.split('/')[-2]
        if args.mode == 'mia':
            file_name += f'_mia_{args.mia_count}'
            # TODO: remove the below line after the hyper-parameter tuning
            # path = os.path.join(path, os.pardir)
        else:
            path = os.path.join(path, os.pardir)
        save_path_accuracy = os.path.join(path, args.accuracy_log)
        print('save path accuracy: ', save_path_accuracy)
        with open(save_path_accuracy, 'a') as writer:
            writer.write(f"{file_name},{last_accuracy}\n")


def string_from_array(arr, delimiter="_"):
    return delimiter.join([str(x) for x in arr])


def prepare_path(args):
    """Prepare the path based on the parameters.

    Args:
        args: the program parameters.

    Returns: the full path name based on the parameters
    """
    # 1) the type of the method
    if args.individualize == "None":
        method_name = 'standard'
    else:
        method_name = args.individualize
    path = args.save_path
    path = os.path.join(path, f"{method_name}")

    # 2) the name of the dataset
    path = os.path.join(path, f"{args.dname}")

    # 3) final folder name
    budgets_str = string_from_array(arr=args.budgets)
    ratios_str = string_from_array(arr=args.ratios)
    seeds_str = string_from_array(arr=args.seeds)
    name = f"epochs_{args.epochs}_batch_{args.batch_size}_lr_{args.lr}_" \
           f"max_grad_norm_{args.max_grad_norm}_budgets_{budgets_str}_ratios_" \
           f"{ratios_str}_seeds_{seeds_str}"
    path = os.path.join(path, name)
    return path


def main(args):
    logname = 'training.log'

    args.save_path = prepare_path(args=args)
    args.individualize = args.individualize if args.individualize != 'None' \
        else None
    args.adapt_weights_to_budgets = args.adapt_weights_to_budgets == 'True'
    args.allow_excess = args.allow_excess == 'True'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logging.basicConfig(
        filename=os.path.join(args.save_path, logname),
        filemode='w',
        level=logging.INFO,
        force=True,  # automatically remove the root handlers
    )

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Input args: {args}')

    for s in args.seeds:
        args.seed = s
        torch.manual_seed(s)
        np.random.seed(s)

        if args.individualize == 'None':
            args.individualize = None

        if args.architecture == 'VGG':
            model = VGG(architecture_name='VGG7', dataset_name=args.dname)
        elif args.architecture == 'MNIST_CNN':
            model = MNIST_CNN()
        elif args.architecture == 'CIFAR10_CNN':
            model = CIFAR10_CNN()
        else:
            logging.info('Currently not supported model type')

        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum)
        run_idp_sgd_experiment(
            model=model,
            dataset_name=args.dname,
            optimizer=opt,
            accountant=args.accountant,
            individualize=args.individualize,
            cuda=args.use_cuda,
            epochs=args.epochs,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
            max_physical_batch_size=args.max_physical_batch_size,
            delta=args.delta,
            log_iteration=args.log_iteration,
            budgets=args.budgets,
            ratios=args.ratios,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=args.noise_multiplier,
            weights=args.weights,
            adapt_weights_to_budgets=args.adapt_weights_to_budgets,
            path=args.save_path,
            seed=s,
            mode=args.mode,
            args=args,
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
