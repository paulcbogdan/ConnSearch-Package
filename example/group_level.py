import pathlib
import sys
sys.path.append(f'{pathlib.Path(__file__).parent.parent}')

import os

from sklearn.svm import SVC

from connsearch.connsearcher import ConnSearcher
from connsearch.components import get_components
from connsearch.report.plots import plot_components, \
    plot_ConnSearch_ROI_scores
from connsearch.report.tables import prepare_components_table
from connsearch.permute.perm_manager import Permutation_Manager
from connsearch.utils import clear_make_dir, Colors
from example.generate_dataset import generate_dataset


def run_group_ConnSearch(dir_results,
                         n_splits=5,
                         n_repeats=10,
                         acc_thresh=None,
                         override_existing=True,
                         proximal=False,
                         wu_analysis=False,
                         no_components=False,
                         comp_size=16,
                         make_component_plots=True,
                         make_table=True,
                         make_ROI_plots=True,
                         n_perm=21,
                         n_edges_show_effect=100):
    '''
    Runs a ConnSearch group-level analysis. This function (a) prepares a
        directory to save results and plots, (b) loads data, (c) defines
        components, (d) creates the ConnSearcher object, (e) loads the
        permute-testing accuracy threshold via Permutation_Manager, (f)
        runs the ConnSearcher object's group-level analysis, (g) plots every
        significant component, Figure 3, (h) generates a table detailing
        significant components, Table 1, and (i) creates the ROI plots, showing
        the accuracy of each ROI's corresponding component, Figure 5.

    Many of these steps are optional. Additionally, instead of being used for
        a ConnSearch analysis, the ConnSearcher object can also be used for the
        analytic technique described in Wu et al. (2021; Cerebral Cortex).
        However, this feature was not used for the final manuscript.

    :param dir_results: str, directory where results and plots will be saved
    :param n_splits: int, number of cross-validation folds.
    :param n_repeats: int, number of cross-validation repeats. The type of cross
        validation used is repeated k-fold group stratified cross-validation.
    :param acc_thresh: float None, or 0. If None, the accuracy threshold for
        significance is determined via permute-testing. If 0, all components
        are considered significant (i.e., all components' results are saved)
    :param override_existing: bool, if True, the directory where results are
        saved will be overwritten
    :param proximal: bool, if True, components are defined based on proximity
        to a Core ROI. Not used for the final manuscript beyond a footnote.
    :param wu_analysis: bool, if True, the Wu et al. (2021) approach or a
        modified version, which uses only edges within the component, is used.
        Not used for the manuscript.
    :param no_components: bool, if True, does the original Wu method, where
        classifiers are fit based on every single connection to an ROI.
        Not used for the manuscript.
    :param comp_size: int, number of ROIs in the component
    :param N: int, number of subjects
    :param make_component_plots: bool, if True, plots significant components
    :param make_table: bool, if True, generates table of significant components
    :param make_ROI_plots: bool, if True, generates ROI plots
    :param n_edges_show_effect: int, number of edges in generated dataset to
        that will show an effect (the effect is Cohen's d = 1)
    :return:
    '''
    assert not no_components or wu_analysis, \
        f'Can\'t do ConnSearch if: {no_components=}'
    clear_make_dir(dir_results, clear_dir=override_existing)  # create directory

    X, Y, coords = generate_dataset(n_edges_show_effect=n_edges_show_effect)
    # Define components. The manuscript only defined components based on
    #   connectivity but alludes to the possibility of defining components
    #   based on proximity to a Core ROI (if proximal=True). We also implemented
    #   the technique by Wu et al. (2021; Cerebral Cortex), which is used by
    #   setting no_components=True.
    components = get_components(proximal=proximal, no_components=no_components,
                                X=X, comp_size=comp_size, coords=coords)

    # Defining ConnSearcher. This involves passing it a sklearn-style classifier
    #   which will be trained/tested for each component. We used an SVM
    #   See the ConnSearcher object.
    clf = SVC(kernel='linear')
    CS = ConnSearcher(X, Y, coords, components, dir_results, n_splits=n_splits,
                      n_repeats=n_repeats, clf=clf, wu_analysis=wu_analysis)

    # Run group-level analysis. acc_thresh determines which component results
    #   are saved. If acc_thresh=None, then components are saved if their
    #   accuracy surpasses the threshold found by permute-testing (p < .05).
    # If acc_thresh is a float (0.0-1.0), components will be saved if their
    #   accuracy surpasses that threshold. If acc_thresh=0, then all components
    #   are saved (useful for plotting all component results, e.g., Figure X)
    if acc_thresh is None:
        # See permute.Permutation_Manager. This class manages the
        #   permute-testing. The actual permute-testing, which takes
        #   over an hour usually, is done elsewhere (see run_permutations.py).
        #   PM object simply retrieves those cached permute-testing results.
        #   The retrieved results account for the current dataset size,
        #   cross-validation, etc. settings. These settings are retrieved from
        #   the ConnSearcher object (CS).
        PM = Permutation_Manager(connsearcher=CS, n_perm=n_perm,
                                 cache_dir=f'{ROOT_DIR}/example/'
                                           f'permutation_saves')
        # Based on permute-testing, the p-value associated with a given
        #   accuracy can be calculated. PM creates a function acc2p, which
        #   takes an accuracy and returns the associated p-value. The opposite
        #   is done by p2acc. An error will be raised if permute-testing
        #   results are not found for the current ConnSearcher settings.
        #   Note that the p-values correspond to FWE corrected p-values.
        acc2p, p2acc, _ = PM.get_acc_pval_funcs(min_perms=n_perm)
        print(f'\nAccuracy threshold: '
              f'{Colors.GREEN}{p2acc(0.05):.1%}{Colors.ENDC}\n')
        CS.do_group_level_analysis(acc2p=acc2p)  # Run group-level analysis
        # save components where p < .05
    else:  # You can also specify a threshold directly. This is viable if you
        #   want quick results and do not want to wait for permute-tests.
        CS.do_group_level_analysis(acc_thresh=acc_thresh)

    # At this point, dir_results will be populated with .pkl files. Each
    #   .pkl file contains the results for a single saved component.
    # The upcoming functions plot the results and/or make a table of results.
    # Each of these functions below operate by loading .pkls from dir_results,
    #    then generating a .csv or .png(s). These are saved nearby dir_results.
    if make_table:
        fn_csv = f'ConnSearch_Group.csv'
        dir_table = pathlib.Path(dir_results).parent
        fp_csv = os.path.join(dir_table, fn_csv)
        prepare_components_table(fp_csv, dir_results)

    if make_component_plots or make_ROI_plots:
        dir_pics = os.path.join(pathlib.Path(dir_results).parent, 'pics')
        # create pics directory
        clear_make_dir(dir_pics, clear_dir=override_existing)

        if make_component_plots:
            plot_components(dir_results, dir_pics)

        if make_ROI_plots:
            fp_ROI_plots = os.path.join(dir_pics, 'ROI_plots.png')
            plot_ConnSearch_ROI_scores(dir_results, fp_ROI_plots,
                                       group_level=False, avg_ROIs=True,
                                       vmin=.55, vmax=.61)
            # If avg_ROIs=True, the plots assign ROIs scores as the average of
            #   all the components they contributed to.
            # If avg_ROIs=False, assigns ROIs scores based solely on the
            #   component for which they are the Core ROI.



if __name__ == '__main__':
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    DIR_RESULTS = fr'{ROOT_DIR}/example/group_level_results/out_data'
    run_group_ConnSearch(DIR_RESULTS, make_ROI_plots=False)

    DIR_RESULTS_ALL = fr'{ROOT_DIR}/example/group_results_all/out_data'
    run_group_ConnSearch(DIR_RESULTS_ALL, n_repeats=1,
                         make_component_plots=False, make_table=False,
                         make_ROI_plots=True, n_edges_show_effect=500,
                         acc_thresh=0)

    # ConnSearch is fairly similar to the Method by Wu et al. (2021).
    # Wu, J., Eickhoff, S. B., Hoffstaedter, F., Patil, K. R., Schwender, H.,
    #   Yeo, B. T., & Genon, S. (2021). A connectivity-based psychometric
    #   prediction framework for brain–behavior relationship studies.
    #   Cerebral Cortex, 31(8), 3732-3751.
    # Their method was actually implemented via the ConnSearcher object.
    #   Just set no_components=True and wu_analysis=True to use it.
    #   We implemented this to serve as a comparison to ConnSearch, although
    #   these results did not make it into the manuscript (in the end, the
    #   method only yielded a single significant result for our N=50 dataset).
    DIR_RESULTS = fr'{ROOT_DIR}/example/Wu_group_results/out_data'
    run_group_ConnSearch(DIR_RESULTS, make_ROI_plots=False, no_components=True,
                         wu_analysis=True, acc_thresh=.60)
