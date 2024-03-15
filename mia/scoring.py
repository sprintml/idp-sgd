import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sklearn.metrics
import torch
from typing import Tuple

from idp_sgd.dpsgd_algos.individual_dp_sgd import initialize_training

individualize = 'sampling'
# individualize = 'clipping'

if individualize == 'sampling':
    # default_path = '/mfsnic/adam/idpsgd/sampling/CIFAR10/epochs_30_batch_1024_lr_0.3_max_grad_norm_0.9_budgets_1.0_2.0_ratios_0.5_0.5'
    default_path = "/mfsnic/adam/idpsgd/mia_eps_10_20/sampling/CIFAR10/epochs_60_batch_1024_lr_0.5_max_grad_norm_1.5_budgets_10.0_20.0_ratios_0.5_0.5"
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
parser.add_argument('--score_method', type=str,
                    default='logits',
                    choices=['probs', 'logits'],
                    help='the method used to do the scoring',
                    )
parser.add_argument('--seed', type=int,
                    default=0,
                    help='keys for reproducible pseudo-randomness',
                    )
parser.add_argument('--individualize', type=str,
                    default=individualize,
                    help='(i)DP-SGD method ("None", "clipping", "sampling")',
                    )


def load_logit_scores(args) -> Tuple[np.ndarray, np.ndarray]:
    shadow_logits_file = os.path.join(args.basefolder, 'shadow_logits.npy')
    shadow_logits = np.load(file=shadow_logits_file)

    target_logits_file = os.path.join(args.basefolder, 'target_logits.npy')
    target_logits = np.load(file=target_logits_file)

    return shadow_logits, target_logits


def load_assignments_shadows(args) -> np.ndarray:
    assignments = []
    for i in range(args.num_shadow_models):
        assignment_file = os.path.join(args.basefolder, f'run{i}/assignment.npy')
        assignment = np.load(file=assignment_file)
        assignments.append(assignment)
    assignments = np.stack(assignments)
    return assignments


def load_assignments_target(args) -> np.ndarray:
    assignment_file = os.path.join(args.basefolder, f'assignment.npy')
    assignment = np.load(file=assignment_file)
    return assignment


def convert_logit_to_prob(logit: np.ndarray, axis=1) -> np.ndarray:
    """Converts logits to probability vectors.
    Args:
      logit: n by c array where n is the number of samples and c is the number of
        classes.
    Returns:
      The probability vectors as n by c array
    """
    prob = logit - np.max(logit, axis=axis, keepdims=True)
    prob = np.array(np.exp(prob), dtype=np.float64)
    prob = prob / np.sum(prob, axis=axis, keepdims=True)
    return prob


def get_labels(args):
    device, train_loader, _ = initialize_training(dataset_name=args.dname,
                                                  cuda=True,
                                                  epochs=-1,
                                                  n_workers=6,
                                                  batch_size=1000,
                                                  seed=args.seed,
                                                  shuffle=False,
                                                  args=args)
    labels = []
    for _, label in train_loader:
        labels.extend(list(label.cpu().numpy()))
    return np.array(labels)


def compute_likelihood(args):
    assignments = load_assignments_shadows(args=args)

    shadow_logits, target_logits = load_logit_scores(args=args)

    if args.score_method == 'probs':
        shadow_logits = convert_logit_to_prob(logit=shadow_logits, axis=2)
        target_logits = convert_logit_to_prob(logit=target_logits, axis=2)

    labels = get_labels(args=args)
    num_points = len(labels)

    print("target model train accuracy: ", np.sum(np.argmax(target_logits[0], axis=1) == labels) / 50000)

    # extract logits for the correct labels
    shadow_logits = np.take_along_axis(shadow_logits, labels[np.newaxis, :, np.newaxis], axis=2)
    target_logits = np.take_along_axis(target_logits, labels[np.newaxis, :, np.newaxis], axis=2)

    shadow_logits = shadow_logits.reshape(shadow_logits.shape[0], -1)
    target_logits = target_logits.reshape(target_logits.shape[0], -1)

    # split to member and non-member lists
    mean_members = np.zeros(num_points)  # logits of members (data points)
    std_members = np.zeros(num_points)
    mean_nonmembers = np.zeros(num_points)  # logits of non-members
    std_nonmembers = np.zeros(num_points)

    all_indices = np.arange(num_points)

    for data_idx in all_indices:
        print('data_idx: ', data_idx)
        member_list = []
        nonmember_list = []
        for model_idx in range(args.num_shadow_models):
            member_indices = assignments[model_idx]
            if data_idx in member_indices:
                member_list.append(shadow_logits[model_idx][data_idx])
            else:
                nonmember_list.append(shadow_logits[model_idx][data_idx])
        mean_member = np.median(member_list)
        std_member = np.std(member_list)
        mean_nonmember = np.median(nonmember_list)
        std_nonmember = np.std(nonmember_list)

        mean_members[data_idx] = mean_member
        std_members[data_idx] = std_member
        mean_nonmembers[data_idx] = mean_nonmember
        std_nonmembers[data_idx] = std_nonmember

    mean_member_file = os.path.join(args.basefolder, f"mean_members_{args.score_method}.npy")
    np.save(arr=mean_members, file=mean_member_file)
    std_member_file = os.path.join(args.basefolder, f"std_members_{args.score_method}.npy")
    np.save(arr=std_members, file=std_member_file)

    mean_nonmember_file = os.path.join(args.basefolder, f"mean_nonmembers_{args.score_method}.npy")
    np.save(arr=mean_nonmembers, file=mean_nonmember_file)
    std_nonmember_file = os.path.join(args.basefolder, f"std_nonmembers_{args.score_method}.npy")
    np.save(arr=std_nonmembers, file=std_nonmember_file)

    mean_member_default = np.nanmean(mean_members)
    mean_members = np.nan_to_num(mean_members, nan=mean_member_default)

    std_member_default = np.nanstd(std_members)
    std_members = np.nan_to_num(std_members, nan=std_member_default)

    mean_nonmember_default = np.nanmean(mean_nonmembers)
    mean_nonmembers = np.nan_to_num(mean_nonmembers, nan=mean_nonmember_default)

    std_nonmember_default = np.nanstd(std_nonmembers)
    std_nonmembers = np.nan_to_num(std_nonmembers, nan=std_nonmember_default)

    pr_in = scipy.stats.norm.logpdf(target_logits, mean_members, std_members + 1e-30)
    pr_out = scipy.stats.norm.logpdf(target_logits, mean_nonmembers, std_nonmembers + 1e-30)

    likelihood_scores = pr_in - pr_out
    likelihood_scores = likelihood_scores.flatten()
    likelihood_file = os.path.join(args.basefolder, f"likelihood_scores_{args.score_method}.npy")
    np.save(arr=likelihood_scores, file=likelihood_file)

    return likelihood_scores


def main(args):
    likelihood_scores = compute_likelihood(args=args)
    target_assignment = load_assignments_target(args=args)
    y_true = np.zeros(likelihood_scores.shape[0])
    y_true[target_assignment] = 1
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=y_true, y_score=likelihood_scores)
    sklearn.metrics.auc(x=fpr, y=tpr)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"roc_curve_{args.score_method}.pdf")


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args=args)
