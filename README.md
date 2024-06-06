# Have it your way: Individualized Privacy Assignment for DP-SGD

This is a research library implementing the individualization in DP-SGD.

## Abstract
When training a machine learning model with differential privacy, one sets a privacy budget. This uniform budget represents an overall maximal privacy violation that any user is willing to face by contributing their data to the training set. We argue that this approach is limited because different users may have different privacy expectations. Thus, setting a uniform privacy budget across all points may be overly conservative for some users or, conversely, not sufficiently protective for others. In this paper, we capture these preferences through individualized privacy budgets. To demonstrate their practicality, we introduce a variant of Differentially Private Stochastic Gradient Descent (DP-SGD) which supports such individualized budgets. DP-SGD is the canonical approach to training models with differential privacy. We modify its data sampling and gradient noising mechanisms to arrive at our approach, which we call Individualized DP-SGD (IDP-SGD). Because IDP-SGD provides privacy guarantees tailored to the preferences of individual users and their data points, we empirically find it to improve privacy-utility trade-offs.

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

Launching your experiments can be done from the main file `idp_sgd/dpsgd_algos/individual_dp_sgd.py`.

A possible run configuration for our sampling method on SVHN data could be:
```
#!/bin/bash

poetry shell
conda deactivate

name="sampling_svhn"

XXX %Set the path to your poetry virtual environment's python% ../idp_sgd/dpsgd_algos/individual_dp_sgd.py \
--save_path "/%Set your results base folder%${name}/" \
--seeds "42" \
--dname "SVHN" \
--architecture "CIFAR10_CNN" \
--individualize "sampling" \
--lr 0.2 \
--epochs 30 \
--batch_size 1024 \
--budgets 1.0 2.0 3.0 \
--ratios 0.34 0.43 0.23 \
--max_grad_norm 0.9 \
--accuracy_log "${name}.log" \
--mode 'run' \
--assign_budget 'random'
```

Our main files are located in `/idp_sgd/dpsgd_algos/`.
While our adaptations to the opacus module can be found in `/opacus/`.

We integrated individualization into our custom opacus module. 
Most changes can be found in `opacus/opacus/optimizers/optimizer.py`, where we extended, for example the optimizer to
operate with per-point individual gradient norms.

The functions that obtain our individualized parameter generation (individual clip norms, sample rates, and noise sclaes)
are located in `opacus/opacus/accountants/utils.py`.
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

## Citing Us
If you build on our paper or code base, please cite as follows:
```
@article{boenisch2024have,
  title={Have it your way: Individualized Privacy Assignment for DP-SGD},
  author={Boenisch, Franziska and M{\"u}hl, Christopher and Dziedzic, Adam and Rinberg, Roy and Papernot, Nicolas},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
