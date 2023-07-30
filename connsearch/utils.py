import os
import shutil
import statistics as stat
from pathlib import Path
from typing import Tuple

import numpy as np


def print_list_stats(l: list) -> None:
    '''
    Prints some basic stats about a list of numbers, including the count, mean,
        median, min, max, and p(under 50%) and p(above 50%) of the list.
        Additionally, it plots details on the percentiles of values in the list.
    This is useful for getting a quick summary of the permute-testing
        results or data on the different component classifiers.

    :param l: list of numbers
    '''
    if len(l) < 2:
        print(f'List has fewer than two elements: {l}.')
        return
    elif all(x == l[0] for x in l):
        print(f'All values are the same, {len(l)=}.')
        return
    l.sort()
    print(f'Number of items: {len(l)}')
    print(f'Mean item: {stat.mean(l):.3f}')
    print(f'Median item: {stat.median(l):.3f}')
    print(f'Min item: {min(l):.3f}')
    print(f'Max item: {max(l):.3f}')
    p_above = stat.mean([int(i > .50001) for i in l])
    p_below = stat.mean([int(i < .49999) for i in l])
    print(f'p(under 50%): {p_below:.3f} | p(above 50%): {p_above:.3f}')
    p_str = 'Percentile: Accuracy | '
    for p_cutoff in [1.0, .75, .5, .25, .1, .05, .01, .005, .001]:
        idx = min(int(len(l) * (1 - p_cutoff) + .999), len(l) - 1)
        p_str += f'{p_cutoff}: {Colors.BLUE}{l[idx]:.4f}{Colors.ENDC}, '
    print(p_str[:-2])  # the ':-2' crops out the comma and space at the end
    print()


def clear_make_dir(dir_name: str,
                   clear_dir: bool = True) -> None:
    '''
    Creates directory if it does not exist. If it does exist, and clear_dir is
        True, it will delete the directory and all its contents before creating
        the new empty one.

    :param dir_name: str, name of directory to create
    :param clear_dir: boolean, if True, deletes existing directory and contents
    '''
    if clear_dir:
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name, ignore_errors=True)
    if not os.path.isdir(dir_name):
        Path(dir_name).mkdir(parents=True, exist_ok=True)  # os.mkdir(dir_name)


class Colors:
    '''
    Codes for printing colored text to the terminal. Helps things look nice
    Taken from: https://stackoverflow.com/questions/37340049/how-do-i-print-colored-output-to-the-terminal-in-python
    '''
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'


def tril2flat_mappers(tril_idxs: Tuple[np.ndarray, np.ndarray]) -> tuple:
    '''
    Convert the return from np.tril_idxs(...) into a dictionary mapping edges
        to indices and vice versa. This is useful dealing with matrices whose
        bottom triangle has been flattened into a vector.

    :param tril_idxs: tuple containing two equally-lengthed lists.
    :return:
    '''
    edge_to_idx = {}
    for cnt, idx in enumerate(np.transpose(tril_idxs)):
        edge_to_idx[(idx[0], idx[1])] = cnt
    idx_to_edge = {value: key for key, value in edge_to_idx.items()}
    return edge_to_idx, idx_to_edge


def avg_by_subj(X_by_subj: np.ndarray,
                Y_by_subj: np.ndarray,
                cond: int) -> np.ndarray:
    '''
    Get average of X for a given condition (cond). Y_by_subj specifies the cond
        for each example. For this function, X should have already been averaged
        by subject, so there shouldn't be dimensions corresponding to both
        session and example-within-session.
    :param X_by_subj: 4D np.array of shape (subject, example, ROI0, ROI1)
    :param Y_by_subj: 2D np.array of shape (subject, example)
    :param cond: int, the label of the example
    :return:
    '''
    x_dims = len(X_by_subj.shape)
    y_dims = len(Y_by_subj.shape)
    Y_cond = Y_by_subj == cond
    for dim in range(y_dims, x_dims):
        Y_cond = np.expand_dims(Y_cond, dim)
        Y_cond = np.repeat(Y_cond, X_by_subj.shape[dim], axis=dim)
    X_subj_avgs = np.average(X_by_subj, axis=1, weights=Y_cond)
    return X_subj_avgs


def get_t_graph(X: np.ndarray,
                Y: np.ndarray) -> np.ndarray:
    '''
    Generates a t-statistic graph (matrix) for the difference between two
        conditions. This represents a paired t-test applied to every single
        edge. This involves averaging for each condition within subject, then
        taking the difference between the two conditions for each subject.
        Then, for each edge, t = mean(difference)/se(difference)
    :param X: 5D np.array of shape (subject, session, example, ROI0, ROI1)
    :param Y: 3D np.array of shape (subject, session, example)
    :return:
    '''
    Y_by_subj = Y.reshape((Y.shape[0], -1))
    X_by_subj = X.reshape((X.shape[0], -1, X.shape[3], X.shape[4]))
    X0_subj_avg = avg_by_subj(X_by_subj, Y_by_subj, 0)  # shape = (subject, ROI0, ROI1)
    X1_subj_avg = avg_by_subj(X_by_subj, Y_by_subj, 1)  # shape = (subject, ROI0, ROI1)
    M_dif_graph = np.mean(X1_subj_avg - X0_subj_avg, axis=0)
    std_dif_graph = np.std(X1_subj_avg - X0_subj_avg, axis=0, ddof=1)
    se_dif_graph = std_dif_graph / np.sqrt(X0_subj_avg.shape[0])
    # Set the diagonal to 1, so we don't get a warning for dividing by 0 or NaN
    se_dif_graph[np.diag_indices(se_dif_graph.shape[0])] = 1
    t_graph = M_dif_graph / se_dif_graph  # shape = (ROI0, ROI1)
    return t_graph


def format_time(t_secs: float) -> str:
    '''
    Format a time in seconds into a string of "X hours Y minutes Z seconds"
    :param t_secs: float, time in seconds
    :return: string, "X hours Y minutes Z seconds"
    '''
    hours = int(t_secs / 3600)
    minutes = int((t_secs / 60) % 60) if t_secs / 3600 > 0 else int(t_secs / 60)
    seconds = int(t_secs % 60)
    return f"{hours} hours {minutes} minutes {seconds} s"


def get_groups(X: np.ndarray,
               sn_dim: int = 0,
               sess_dim: int = 1,
               ex_dim: int = 2):
    '''
    For the 5D X, see load_data_5D(...), the first three dimensions are
        (subject, session, example). These get reshaped into a single dimension
        when X is returned as a 2D array. group_idxs is a list that maps the
        examples to their associated subject. e.g., the first example is from
        participant 0, so group_idxs[0] = 0.
    This function will, notably, also work for other types of X (e.g., 6D array
        where the 0th dimension correspond to a given component)

    :param X: 5D np array of X, where (subjects, sessions, examples, ROI0, ROI1)
    :param sn_dim: int, specify which dimension corresponds to the subject
    :param sess_dim: int, specify which dimension corresponds to the subject
    :param ex_dim: int, specify which dimension corresponds to the subject
    :return: groups_idx, list
    '''
    if sess_dim is None:
        groups = np.repeat(list(range(X.shape[sn_dim])), X.shape[ex_dim])
    else:
        groups = np.repeat(list(range(X.shape[sn_dim])),
                           X.shape[sess_dim] * X.shape[ex_dim])
    return groups
