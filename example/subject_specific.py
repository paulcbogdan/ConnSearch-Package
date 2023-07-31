import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import generate_dataset

import os

from connsearch.connsearcher import ConnSearcher
from connsearch.components import get_components
from connsearch.report.plots import plot_components, \
    plot_ConnSearch_ROI_scores
from connsearch.report.tables import prepare_components_table
from connsearch.utils import clear_make_dir
from example.generate_dataset import generate_dataset

def run_subject_ConnSearch(dir_results,
                           alpha_thresh=.05,
                           override_existing=True,
                           fwe=True,
                           proximal=False,
                           wu_analysis=False,
                           no_components=False,
                           comp_size=16,
                           subtract_mean=False,
                           make_component_plots=True,
                           make_table=True,
                           make_ROI_plots=True,
                           n_edges_show_effect=2000):
    '''
    Runs a ConnSearch subject-specific analysis. This function (a) prepares a
        directory to save results and plots, (b) loads data, (c) defines
        components, (d) creates the ConnSearcher object, (e)
        runs the ConnSearcher object's subject-specific analysis, (g) plots
        every significant component, Figure 7 top, (h) generates a table
        detailing significant components, Table 3, and (i) creates the ROI
        plots, showing the accuracy of each ROI's corresponding component,
        Figure 7 bottom.

    :param dir_results: str, directory where results and plots will be saved
    :param alpha_thresh: float, corrected p-value threshold for significance
    :param override_existing: bool, if True, the directory where results are
        saved will be overwritten
    :param fwe: bool, if True, uses Holm-Sidak (FWE) correction. If False,
        uses Benjamani-Hochberg (FDR) correction.
    :param proximal: bool, if True, components are defined based on proximity
        to a Core ROI. Not used for the final manuscript.
    :param wu_analysis:bool, if True, the Wu et al. (2021) approach or a
        modified version, which uses only edges within the component, is used.
        Not used for the manuscript.
    :param no_components: bool, if True, does the original Wu method, where
        classifiers are fit based on every single connection to an ROI.
        Not used for the manuscript.
    :param comp_size: int, number of ROIs in the component
    :param N: int, number of subjects
    :param dataset_num: int, which dataset to use. The manuscript needed this to
        select amongst the five 50-participant datasets.
    :param subtract_mean: bool, if True, subtracts the mean group-level 2-back
        vs. 0-back connectivity from each edge before subject-specific analysis.
        See Supplemental Methods 1.3.
    :param make_component_plots: bool, if True, plots significant components
    :param make_table: bool, if True, generates table of significant components
    :param make_ROI_plots: bool, if True, generates ROI plots
    :param n_edges_show_effect: int, number of edges in generated dataset to
        that will show an effect (the effect is Cohen's d = 1).
    :return:
    '''

    assert not no_components or wu_analysis, \
        f'Can\'t do ConnSearch if: {no_components=}'
    clear_make_dir(dir_results, clear_dir=override_existing) # create directory

    X, Y, coords = generate_dataset(n_edges_show_effect=n_edges_show_effect)

    # Define components. The manuscript only defined components based on
    #   connectivity but alludes to the possibility of defining components
    #   based on proximity to a Core ROI (if proximal=True). We also implemented
    #   the technique by Wu et al. (2021; Cerebral Cortex), which is used by
    #   setting no_components=True.
    components = get_components(proximal=proximal, no_components=no_components,
                                X=X, comp_size=comp_size, coords=coords)

    CS = ConnSearcher(X, Y, coords, components, dir_results,
                      wu_analysis=wu_analysis)
    CS.do_subject_specific_analysis(alpha=alpha_thresh, FWE=fwe,
                                    subtract_mean=subtract_mean)

    if make_table:
        fn_csv = f'ConnSearch_subject-specific.csv'
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
                                       group_level=False, avg_ROIs=False,
                                       vmin=1.7, vmax=2.7)
            # If avg_ROIs=True, the plots assign ROIs scores as the average of
            #   all the components they contributed to.
            # If avg_ROIs=False, assigns ROIs scores based solely on the
            #   component for which they are the Core ROI.


if __name__ == '__main__':
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    DIR_RESULTS = fr'{ROOT_DIR}/example/subject_specific_results/out_data'
    run_subject_ConnSearch(DIR_RESULTS, make_ROI_plots=False)

    DIR_RESULTS = \
        fr'{ROOT_DIR}/example/subject_specific_results_all/out_data'
    run_subject_ConnSearch(DIR_RESULTS, subtract_mean=False,
                           make_component_plots=False, make_table=False,
                           make_ROI_plots=True, alpha_thresh=1.)