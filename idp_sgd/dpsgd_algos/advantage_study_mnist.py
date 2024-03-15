import math
import os
from pathlib import Path

# MNIST hyperparameters
# epochs = 80
# max_grad_norm = 0.2
# learning_rate = 0.6

if __name__ == "__main__":
    cwd = Path.cwd()
    path = cwd / 'res/advantage_MNIST/'

    combinations = [
        (seed, individualize, budgets, ratios)
        for individualize, budgets, ratios in [
            ('clipping', '1.0 2.0 3.0', '0.34 0.43 0.23'),
            ('sampling', '1.0 2.0 3.0', '0.34 0.43 0.23'),
            ('clipping', '1.0 2.0 3.0', '0.54 0.37 0.09'),
            ('sampling', '1.0 2.0 3.0', '0.54 0.37 0.09'),
            (None, '1.0', '1.0'),
            (None, '3.0', '1.0'),

            # These are intuitively understandable and comparable DP budgets:
            # (None, f'{math.log(1 + 0.25)}', '1.0'),     # 25% prob. change
            # (None, f'{math.log(1 + 0.5)}', '1.0'),      # 50% prob. change
            # (None, f'{math.log(1 + 1.0)}', '1.0'),      # 100% prob. change
            # (None, f'{math.log(1 + 2.0)}', '1.0'),      # 200% prob. change
            # (None, f'{math.log(1 + 4.0)}', '1.0'),      # 400% prob. change
            # (None, f'{math.log(1 + 8.0)}', '1.0'),      # 800% prob. change
        ] for seed in list(range(10))
    ]

    print(f'Run {len(combinations)} experiments...')
    for i, (seed, individualize, budgets, ratios) in enumerate(combinations):
        print(f'Experiment {i} | {len(combinations)}...')
        os.system(f'python individual_dp_sgd.py '
                  f'--save_path {path} '
                  f'--seeds {seed} '
                  f'--individualize {individualize} '
                  f'--budgets {budgets} '
                  f'--ratios {ratios} '
                  f'--epochs 80 '
                  f'--max_grad_norm 0.2 '
                  f'--lr 0.6 '
                  f'--assign_budget random '
                  f'--batch_size 512 ')
