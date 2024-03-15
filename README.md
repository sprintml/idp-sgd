# Individualized DP-SGD
This is a research library exploring individualization in DP-SGD.

## Development Flow

All code dependencies are managed through Poetry. 
### Installation Using a Virtual Environment 
1. Create a virtual environment `python3.9 -m venv venv`
2. Source this environment `source venv/bin/activate`
3. From the base of the codebase, run `pip install -e .`   

### Installation Using Poetry
`poetry` (version 1.2.1) is used to manage the dependencies.

1. Install Poetry version 1.2.1 (`curl -sSL https://install.python-poetry.org | python3 -`)
2. Navigate to the base of the codebase.
3. Run `poetry shell`
4. Run `poetry install`

### Dependencies management

To add a new dependency run `poetry add <dep-name>`.

## Code structure 
Our main files are located in /idp_sgd/dpsgd_algos/
While our adaptations to the opacus module can be found in /opacus/

We integrated individualization into our custom opacus module. 
Most changes can be found in opacus/opacus/optimizers/optimizer.py, where we extended, for example the optimizer to
operate with per-point individual gradient norms.

The functions that obtain our individualized parameter generation (individual clip norms, sample rates, and noise sclaes)
are located in opacus/opacus/accountants/utils.py.
These are called by the privacy engine within `make_private()` and `make_private_with_epsilon()`.


## Running Experiments

We recommend running experiments using `individual_dp_sgd.py`

```
poetry shell

python ../idp_sgd/dpsgd_algos/individual_dp_sgd.py \
    --save_path $SAVE_DIR \
    --seeds ${seed} \
    --dname "CIFAR10" \ # dataset-name
    --architecture "CIFAR10_CNN" \ # architecture-name
    --individualize "None" \ # individualization : {"None", "sampling", "clipping"}
    --lr $LEARNING_RATE \
    --epochs 30 \
    --batch_size 1024 \
    --budgets 1.0 2.0 3.0 \ # a list of the individualized budgets 
    --ratios 0.54 0.37 0.09 \  # a list of the individualized ratios (must add up to 1.)
    --max_grad_norm ${max_grad_norm} \ # max max-grad-norm
    --mode 'run' # 'run' or 'mia' which is used to conducting MI tests
```

individualize flags: {"None", "sampling", "clipping"}
