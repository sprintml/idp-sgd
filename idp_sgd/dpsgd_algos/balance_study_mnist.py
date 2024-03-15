import os
from pathlib import Path

if __name__ == "__main__":
    cwd = Path.cwd()
    path = cwd / 'res/balance_MNIST/'

    noise_multipliers = [1.2, 1.4, 1.6, 1.8, 2.0]

    weights = '0.0019104013715209962 0.0027801523393859865 ' \
              '0.0068237314005126955 0.006976319289611817 ' \
              '0.007128907178710937 0.008648682554138183 ' \
              '0.0098266610579834 0.010119629805053713 ' \
              '0.012963868057861328 0.01832275472302246'

    seeds = range(10)
    j = 0
    print(f'Run {seeds} experiments...')
    for i, noise_multiplier in enumerate(noise_multipliers):
        for s in seeds:
            print(f'Experiment {j} | {len(seeds) * len(noise_multipliers)}...')
            os.system(
                f'python individual_dp_sgd.py '
                f'--save_path {path} '
                f'--seeds {j} '
                f'--individualize sampling '
                f'--budgets 0.5 0.75 2.0 2.05 2.1 2.6 3.0 3.1 4.1 6.1 '
                f'--ratios {"0.1 " * 10} '
                f'--class_budgets 0.75 0.5 2.0 2.6 4.1 2.1 2.05 3.0 3.1 6.1 '
                f'--epochs 80 '
                f'--max_grad_norm 0.2 '
                f'--lr 0.6 '
                f'--assign_budget per-class '
                f'--batch_size 512 '
                f'--adapt_weights_to_budgets False '
                f'--noise_multiplier {noise_multiplier} '
                f'--weights {weights} '
                f'--allow_excess True ')
            j += 1
