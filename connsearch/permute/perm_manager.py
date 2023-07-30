import os
import pickle
import statistics as stat
import time
import warnings
from copy import deepcopy
from glob import glob
from typing import Union, Callable

import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from tqdm import tqdm

from connsearch.components import get_connectivity_comps, get_proximal_comps
from connsearch.connsearcher import ConnSearcher
from connsearch.permute.shuffle import shuffle_X_within_session, \
    shuffle_X_within_subject, shuffle_X_within_half, shuffle_Y_fully, \
    scramble_X_fully
from connsearch.utils import print_list_stats, format_time


class Permutation_Manager:
    '''
    This class is used to create permute-testing distributions for a given
        classifier and dataset. It is designed to be used with ConnSearcher.
        Permutation-testing, notably, takes the highest accuracy achieved by
        any component classifier for a given shuffled dataset. Thus, this is a
        form of FWE correction.
    You should run permute-testing before you conduct your analyses.
    The permute-testing results are cached, and then later, you can
        retrieve them without running the permute-testing again.
    '''

    def __init__(self, X: Union[None, np.ndarray] = None,
                 Y: Union[None, np.ndarray] = None,
                 coords: Union[list, np.ndarray] = None,
                 clf: Union[None, BaseEstimator, ClassifierMixin] = None,
                 n_folds: int = 5,
                 n_repeats: int = 10,
                 n_perm: int = 1000,
                 comp_size: int = 8,
                 component_func: Callable = get_connectivity_comps,
                 load_existing: bool = True,
                 perm_strategy: str = 'within_subject',
                 wu_analysis: bool = False,
                 cache_dir: str = 'permutation_saves',
                 connsearcher: Union[None, ConnSearcher] = None,
                 desc: str = '',
                 seed: int = 0) -> None:
        '''
        There are very many settings that need to be taken into account when
            identifying the p-value associated with a classifier's accuracy.

        :param X: X, dataset to be shuffled then used to train/test classifiers.
                     takes in the 5D dataset returned by load_data_5D. Note that
                     X can also be set by passing a ConnSearcher object (see
                     below)
        :param Y: Y, labels for X. As with X, it can be set with a ConnSearcher
                     object
        :param coords: (#ROIs, 3) array of coordinates for each ROI. Only used
                       if you are defining components based on proximity
        :param clf: sklearn untrained classifier (e.g., SVM)
        :param n_folds: int, number of folds for cross-validation
        :param n_repeats: int, number of times to repeat cross-validation
        :param n_perm: int, number of permutations to run
        :param comp_size: int, number of ROIs in each component
        :param component_func: function used to define components
        :param load_existing: bool, if True: loads & continues existing results
                    (e.g., if n_perms has not been reached). if False: restarts
        :param perm_strategy: str, how to shuffle the data. options:
                    'within_subject', 'scramble_X', 'within_session',
                    'shuffle_Y', 'within_half'. within_subject will likely be
                    best for datasets where subjects contribute multiple
                    examples. The manuscript only uses within_subject.
        :param wu_analysis: bool, if True: does the Wu approach, else ConnSearch
                          Use func_components=get_none_components if doing Wu
        :param cache_dir: str, directory to save/load results
        :param connsearcher: ConnSearcher object. Rather than passing the above
            settings, most can simply be extracted from this objected.
        :param desc: str, extra string added near the end of the cache filename.
                     Some users may find this useful
        :param seed: int, random seed. Good to use for reproducibility
        '''
        np.random.seed(seed)

        assert (X is not None and Y is not None) or connsearcher is not None, \
            'Must specify either X and Y or connsearcher'
        if (X is not None or Y is not None) and connsearcher is not None:
            warnings.warn('You provided X/Y and a ConnSearcher object. You '
                          'should only provide one. The ConnSearcher object '
                          'will be ignored.')

        if connsearcher is None:  # Can set the settings manually
            self.X = X
            self.Y = Y
            self.coords = coords
            self.clf = clf
            self.comp_size = comp_size
            self.n_splits = n_folds
            self.n_repeats = n_repeats
            self.wu_analysis = wu_analysis
        else:  # Or settings can be loaded from a passed ConnSearcher object
            self.X = connsearcher.X
            self.Y = connsearcher.Y
            self.coords = connsearcher.coords
            self.clf = connsearcher.clf
            self.comp_size = connsearcher.comp_size
            self.n_splits = connsearcher.n_splits
            self.n_repeats = connsearcher.n_repeats
            self.wu_analysis = connsearcher.wu_analysis
        self.n_perm = n_perm
        self.desc = desc
        self.component_func = component_func
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.perm_strategy = perm_strategy
        scheme2func = {'scramble_X': scramble_X_fully,  # Data shuffle functions
                       'within_session': shuffle_X_within_session,
                       'within_subject': shuffle_X_within_subject,
                       'shuffle_Y': shuffle_Y_fully,
                       'within_half': shuffle_X_within_half}
        self.perm_func = scheme2func[self.perm_strategy]
        self.init_cache_str()  # Sets up the filename for the cache

        self.max_scores = []  # Highest accuracy of any component classifier
        #   for a given shuffled dataset. list length will
        #   be the number of analyzed shuffled datasets.
        self.all_scores = []  # All accuracies of all component classifiers
        #   for a given shuffled dataset. Basically a 2D
        #   array shaped (#completed, #components)
        self.perm_times = []  # Time to run each permute. list length will
        #   be the number of analyzed shuffled datasets.
        #   Recorded to help estimate time to completion.
        self.load_existing = load_existing
        self.init_load()  # Load existing permute-testing results if they
        #    exist. Useful if you have, say, you have run 500
        #    permutations so far and want 1000 in total.

    def init_cache_str(self):
        '''
        The permute-testing results will be cached. This function sets up
            the filename for that cache. The filename specifies the settings.
        '''
        cache_str = self.desc
        X = self.X
        x_str = '' if X is None else \
            f'x{X.shape[0]}-{X.shape[1]}-{X.shape[2]}-{X.shape[3]}_'
        y_str = '' if self.Y is None else \
            'y' + ''.join([str(y) for y in np.unique(self.Y)]) + '_'
        set_size_str = '_NoSet' if self.comp_size is None else \
            f'set{self.comp_size}'
        split_str = '' if self.n_splits is None else f'spl{self.n_splits}_'
        repeat_str = '' if self.n_splits is None else f'rep{self.n_repeats}'
        comp_str = 'prox_' if self.component_func == get_proximal_comps else ''
        wu_str = 'wu_' if self.wu_analysis else ''
        cache_str += f'{x_str}{y_str}{set_size_str}{split_str}' \
                     f'{repeat_str}{comp_str}{wu_str}_' \
                     f'{self.perm_strategy}'
        self.cache_str = cache_str

    def init_load(self):
        '''
        Loads existing permute-testing results for a given setting, if
            self.continue_existing = True.
        :return:
        '''
        if self.load_existing:
            _, fp_loaded = self.find_and_load()
            if fp_loaded is None: return
            n_done = len(self.max_scores)
            n_left = self.n_perm - n_done
            avg_time = np.mean(self.perm_times)
            std_time = np.std(self.perm_times)
            print('Loaded existing permute results_data.\n'
                  f'\tfp = {fp_loaded}\n'
                  f'\tn perms complete: {n_done}\n'
                  f'\tseconds/perm: {avg_time:.2f} s +/- {std_time:.2f} s\n'
                  f'\t{n_left} left to go ({format_time(n_left * avg_time)})')
            if n_left == 0:
                print('All permutations done!')

    def find_and_load(self):
        '''
        Loads the cached permute-testing results. In creating a
            Permutation_Manager, users will specify how many permutations it
            should have (n_perms). This function will load whatever is cached
            work even if not all n_perms are done. For example, n_perms=1000
            but only 500 are done and cached.
        Uses the self.cache_str specified in self.setup_cache_str()
        :return:
        '''

        def get_glob_str_n(glob_str: str):
            '''
            Gets ts the cached file with the largest number of permute
            simulations. This number is in the filename.
            '''
            fp_piece = os.path.join(self.cache_dir, f'{self.cache_str}_n')
            glob_str_pruned = glob_str.replace(fp_piece, '')
            glob_str_pruned = glob_str_pruned.replace('.pkl', '')
            return int(glob_str_pruned)

        glob_l_str = glob(os.path.join(self.cache_dir,
                                       f'{self.cache_str}_n*.pkl'))
        if len(glob_l_str) == 0:
            return None, None
        else:
            glob_str_high_n = max(glob_l_str, key=get_glob_str_n)
            with open(glob_str_high_n, 'rb') as f:
                perm_data = pickle.load(f)
            self.max_scores = perm_data['max_scores']
            self.all_scores = perm_data['all_scores']
            self.perm_times = perm_data['perm_times']
            return perm_data, glob_str_high_n

    def run_permutations(self):
        n_done = len(self.max_scores)
        for i in tqdm(range(n_done, self.n_perm), desc='Permuting'):
            t_st = time.perf_counter()
            X_perm, Y_perm = self.perm_func(deepcopy(self.X),
                                            deepcopy(self.Y))
            components = self.component_func(X=X_perm,
                                             coords=self.coords,
                                             comp_size=self.comp_size)

            cs = ConnSearcher(X_perm, Y_perm, None, components,
                              None, self.n_splits, self.n_repeats,
                              clone(self.clf),  # not sure if clone is needed
                              wu_analysis=self.wu_analysis)
            scores = cs.do_group_level_analysis(verbose=False)
            scores = sorted(scores, reverse=True)
            self.all_scores.append(scores)

            print(f'{self.perm_strategy}: median = {stat.median(scores)} '
                  f'[Note: should average out around 0.5 across all perms]')
            max_score = max(scores)
            self.max_scores.append(max_score)
            self.all_scores.append(scores)
            t_end = time.perf_counter()
            self.perm_times.append(t_end - t_st)

            print(f'Permutation sim {i} took {t_end - t_st:.2f} seconds\n')
            print_list_stats(self.max_scores)
            perm_data = {'max_scores': self.max_scores,
                         'all_scores': self.all_scores,
                         'perm_times': self.perm_times}
            cache_fn = f'{self.cache_str}_n{i + 1}.pkl'
            print(f'Saving permute results to: {cache_fn}')
            with open(os.path.join(self.cache_dir, cache_fn), 'wb') as f:
                pickle.dump(perm_data, f)

            old_cache_fn = f'{self.cache_str}_n{i}.pkl'
            if i > 0 and os.path.isfile(os.path.join(self.cache_dir,
                                                     old_cache_fn)):
                # Deletes cached data from previous loop
                os.remove(os.path.join(self.cache_dir, old_cache_fn))

    def get_acc_pval_funcs(self,
                           min_perms: int = 100,
                           try_nearby: bool = True):
        '''
        Returns functions for converting accuracies to p-values and vice versa.
            In creating a Permutation_Manager, users will specify how many
            permutations it should have (n_perms). This function will try to
            work even if not all n_perms are done. For example, n_perms=1000
            but only 500 are done and cached. If the number done is greater than
            min_perms, then this function will still give the suitable returns
            based on just those 500 permutations. See self.find_and_load().

        Additionally, our preliminary tests showed that component size had
            relatively little bearing on the permute-testing results. Hence,
            if the precise file matching the desired parameters isn't found,
            this function will search for cached results with different
            component sizes, if try_nearby=True. This is useful if you want to
            quickly change comp_size for your analysis but don't want to re-run
            permute-testing, which is slow.

        :param min_perms: minimum number of permutations required done,
            otherwise, this function will error.
        :param try_nearby: if True, will search for cached results with
            alternative component_sizes (other self.comp_size)
        :return: acc2p, a function mapping accuracies to p-values
                        can handle either floats or arrays as inputs
                        and returns floats or arrays, respectively.
                 p2acc, same premise as acc2p but converts p-value to accuracy
                 int, number of permutations completed
        '''

        perm_data, _ = self.find_and_load()  # Loads permute-testing results
        # of the available cached file with the highest number of
        # permutations done.
        comp_size_true = self.comp_size
        self.comp_size = 3
        while (perm_data is None or len(perm_data['max_scores']) < min_perms) \
                and try_nearby:  # Search for cached results with other comp_size
            self.comp_size += 1
            if self.comp_size > 400:  # The loop doesn't search infinitely
                break
            perm_data, _ = self.find_and_load()
        if perm_data is None:
            raise ValueError(f'No permute data found: {self.cache_str}')
        elif len(perm_data['max_scores']) < min_perms:
            raise ValueError(f'Not enough permutations done '
                             f'({len(perm_data["max_scores"])}). '
                             f'Need at least {min_perms}.')
        self.comp_size = comp_size_true  # reset to originally specified size
        self.max_scores = sorted(perm_data['max_scores'])
        n_scores = len(self.max_scores)
        print_list_stats(self.max_scores)

        def acc2p(acc: float):
            '''
            Function that transforms accuracy to p-value.
            Finds the index of the first score in max_scores that is greater
                than acc. Then returns 1 - that index / n_scores as the p-value.
            :param acc: float, between 0 and 1, or an array of floats
            :return:
            '''
            # self.max_scores must be sorted
            i = np.searchsorted(self.max_scores, acc)
            p = 1 - (i - 1) / n_scores
            if p == 0:
                p = 1 / (n_scores * 2)
            return p

        def p2acc(p: float):
            '''Function that transforms p-value to accuracy.'''
            i = int((1 - p) * n_scores)
            return self.max_scores[i]

        # Note that acc2p and p2acc are functions. Users do not call their
        #   permutation_manager manager to get p-values from accuracy. Rather,
        #   they call the returned acc2p function. The same applies vice versa.
        return acc2p, p2acc, len(self.max_scores)
