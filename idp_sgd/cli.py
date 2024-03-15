import logging
import click
import numpy as np
from idp_sgd.dpsgd_algos import weighted_sampling

from idp_sgd.dpsgd_algos.deprecated_algos import upsampling

from idp_sgd.dpsgd_algos import individual_clipping
from opacus.validators import ModuleValidator
from idp_sgd.training_utils.models import VGG
import torch
import yaml
##
## logging
##
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
format = "%(levelname)s - %(message)s"
stream_handler.setFormatter(
    logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S"))

logging.basicConfig(level=logging.INFO, handlers=[stream_handler])
logger = logging.getLogger()


def open_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@click.group(help="""CLI to run DP-SGD training for data-aware-dp""")
@click.pass_context
def cli(ctx):
    return


# TODO : rename this better
@cli.command(name="weight-opacus", help="""weighting opacus""")
@click.option('--ratio',
              '-r',
              type=float,
              default=[0.5, 0.3, 0.2],
              help="ratio (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--budget',
              '-b',
              type=float,
              default=[1.0, 2.0, 3.0],
              help="buget (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--epochs',
              "-e",
              default=40,
              show_default=True,
              help="number of epochs")
@click.option('--batch-size', default=32, show_default=True, help="batch size")
@click.option('--seed', "-s", default=0, show_default=True, help="seed")
@click.option("--config",
              "-c",
              default=None,
              show_default=True,
              help="config file")
@click.pass_context
def weight_opacus(ctx, ratio, budget, epochs, batch_size, seed, config):
    """ TODO: properly pass in the arguments """
    dname = 'MNIST'
    if config is not None:
        config = open_yaml(config)
        ratio = config.get("ratio", ratio)
        budget = config.get("budget", [5.0, 10.0])
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 128)
        seed = config.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    vgg = VGG(architecture_name='VGG7', dataset_name=dname)
    if not ModuleValidator.is_valid(vgg):
        vgg = ModuleValidator.fix(vgg)
    opt = torch.optim.SGD(vgg.parameters(), lr=0.05, momentum=0)
    weighted_sampling.run_sampling_experiment(
        model=vgg,
        dataset_name=dname,
        optimizer=opt,
        cuda=True,
        epochs=50,
        n_workers=6,
        batch_size=batch_size,
        max_physical_batch_size=0,  # --> BatchMemoryManager is not used
        delta=1e-5,
        log_iteration=100,
        budgets=budget,
        ratios=ratio,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        relative_sample_rates=[1.0, 2.0],
        seed=seed)


@cli.command(name="upsampling-dpsgd", help="""upsampling""")
@click.option('--ratio',
              '-r',
              type=float,
              default=[0.5, 0.3, 0.2],
              help="ratio (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--budget',
              '-b',
              type=float,
              default=[1.0, 2.0, 3.0],
              help="buget (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--epochs',
              "-e",
              default=40,
              show_default=True,
              help="number of epochs")
@click.option('--batch-size', default=32, show_default=True, help="batch size")
@click.option('--seed', "-s", default=0, show_default=True, help="seed")
@click.option("--config",
              "-c",
              default=None,
              show_default=True,
              help="config file")
def upsampling_dpsgd(ctx, ratio, budget, epochs, batch_size, seed, config):
    if config is not None:
        config = open_yaml(config)
        ratio = config.get("ratio", ratio)
        budget = config.get("budget", [5.0, 10.0])
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 128)
        seed = config.get("seed", 0)
    model = VGG(architecture_name="VG7", dataset_name="MNIST")
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0)

    upsampling.main(
        model=model,
        optimizer=optimizer,
        cuda=True,
        epochs=epochs,
        n_workers=0,
        log_iteration=1_000,
        ratios=ratio,
        budgets=budget,
        batch_size=batch_size,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        dataset_name='MNIST',
        seed=seed,
    )


@cli.command(name="clipping-idpsgd", help="""clipping individualized dpsgd""")
@click.option('--ratio',
              '-r',
              type=float,
              default=[0.5, 0.5],
              help="ratio (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--budget',
              '-b',
              type=float,
              default=[1.0, 2.0],
              help="buget (list multiple)",
              show_default=True,
              multiple=True)
@click.option('--epochs',
              "-e",
              default=20,
              show_default=True,
              help="number of epochs")
@click.option('--batch-size', default=256, show_default=True, help="batch size")
@click.option('--seed', "-s", default=0, show_default=True, help="seed")
@click.option("--config",
              "-c",
              default=None,
              show_default=True,
              help="config file")
def clipping_idpsgd(ratio, budget, epochs, batch_size, seed, config):

    if config is not None:
        config = open_yaml(config)
        ratio = config.get("ratio", [0.5, 0.5])
        budget = config.get("budget", [5.0, 10.0])
        epochs = config.get("epochs", 20)
        batch_size = config.get("batch_size", 256)
        seed = config.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dname = 'MNIST'
    for s in [0]:
        torch.manual_seed(s)
        np.random.seed(s)
        vgg = VGG(architecture_name='VGG7', dataset_name=dname)
        if not ModuleValidator.is_valid(vgg):
            vgg = ModuleValidator.fix(vgg)
        individual_clipping.run_clipping_experiment(
            model=vgg,
            dataset_name=dname,
            optimizer=torch.optim.SGD(vgg.parameters(), lr=0.05, momentum=0),
            cuda=True,
            epochs=epochs,
            n_workers=6,
            batch_size=batch_size,
            max_physical_batch_size=0,  # --> BatchMemoryManager not used
            delta=1e-5,
            log_iteration=100,
            budgets=budget,
            ratios=ratio,
            noise_multiplier=1.1,
            max_grad_norms=[0.75, 1.0],
            seed=s)


if __name__ == "__main__":
    cli()
