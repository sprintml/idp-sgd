import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy import stats

individualize = 'sampling'
# individualize = 'clipping'
target_nr = 4

if individualize == 'sampling':
    # default_path = '/mfsnic/adam/idpsgd/sampling/CIFAR10/epochs_30_batch_1024_lr_0.3_max_grad_norm_0.9_budgets_1.0_2.0_ratios_0.5_0.5'
    default_path = f"/mfsnic/adam/idpsgd/mia_eps_10_20/sampling/CIFAR10/epochs_60_batch_1024_lr_0.5_max_grad_norm_1.5_budgets_10.0_20.0_ratios_0.5_0.5"
elif individualize == 'clipping':
    # default_path = '/mfsnic/adam/idpsgd/clipping/CIFAR10/epochs_30_batch_1024_lr_0.3_max_grad_norm_0.9_budgets_1.0_2.0_ratios_0.5_0.5'
    default_path = '/mfsnic/adam/idpsgd/mia_eps_10_20/clipping/CIFAR10/epochs_60_batch_1024_lr_0.5_max_grad_norm_1.5_budgets_10.0_20.0_ratios_0.5_0.5'
else:
    raise ValueError('No such individualization')

parser = argparse.ArgumentParser(description='MIA Parser')
parser.add_argument('--basefolder', type=str,
                    default=default_path,
                    help='location of the run folders',
                    )
parser.add_argument('--num_shadow_models', type=int,
                    # default=256,
                    default=512,
                    help='how many shadow models should be used for MIA',
                    )
parser.add_argument('--dname', type=str,
                    default='CIFAR10',
                    help='dataset to be learned',
                    )
parser.add_argument('--seed', type=int,
                    default=0,
                    help='keys for reproducible pseudo-randomness',
                    )
parser.add_argument('--score_method', type=str,
                    default='logits',
                    choices=['probs', 'logits'],
                    help='the method used to do the scoring',
                    )


def run_ttest(score1, score2):
    t_value, p_value = stats.ttest_ind(score1, score2)
    print(f't_value: {t_value}, p_value: {p_value}')
    return p_value


def load_assignments_target(args, target_nr) -> np.ndarray:
    assignment_file = os.path.join(args.basefolder, f'target_model{target_nr}/assignment.npy')
    assignment = np.load(file=assignment_file)
    return assignment


def main(args):
    likelihood_file = os.path.join(args.basefolder, f"target{target_nr}/likelihood_scores_{args.score_method}.npy")
    likelihood_scores = np.load(file=likelihood_file)
    num_points = likelihood_scores.shape[0]
    target_assignment = load_assignments_target(args=args, target_nr=target_nr)
    y_true = np.zeros(num_points)
    half = num_points // 2
    y_true[target_assignment] = 1

    fpr0, tpr0, _ = sklearn.metrics.roc_curve(y_true=y_true, y_score=likelihood_scores, pos_label=1)

    y_true_1 = y_true[:half]
    y_true_2 = y_true[half:]
    likelihood_scores1 = likelihood_scores[:half]
    likelihood_scores2 = likelihood_scores[half:]
    run_ttest(likelihood_scores1, likelihood_scores2)
    fpr1, tpr1, _ = sklearn.metrics.roc_curve(y_true=y_true_1, y_score=likelihood_scores1, pos_label=1)
    fpr2, tpr2, _ = sklearn.metrics.roc_curve(y_true=y_true_2, y_score=likelihood_scores2, pos_label=1)

    auc0 = sklearn.metrics.auc(x=fpr0, y=tpr0)
    auc1 = sklearn.metrics.auc(x=fpr1, y=tpr1)
    auc2 = sklearn.metrics.auc(x=fpr2, y=tpr2)

    precision = 3
    auc0 = "{:.3f}".format(round(auc0, precision))
    auc1 = "{:.3f}".format(round(auc1, precision))
    auc2 = "{:.3f}".format(round(auc2, precision))

    plt.plot(fpr0, tpr0, ":", label=f"$\epsilon=[10, 20], AUC={auc0}$")
    plt.plot(fpr1, tpr1, label=f"$\epsilon=10, AUC={auc1}$")
    plt.plot(fpr2, tpr2, label=f"$\epsilon=20, AUC={auc2}$")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.savefig(f"roc_curve_per_group_{args.score_method}.pdf")


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args=args)
