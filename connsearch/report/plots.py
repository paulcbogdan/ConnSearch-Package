import os
import pickle
from collections import defaultdict
from typing import Union, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
from nichord import convert_matrix, plot_chord, plot_glassbrain, combine_imgs, \
    get_idx_to_label
from nilearn import plotting
from tqdm import tqdm

from connsearch.utils import clear_make_dir


def plot_components(dir_results: str,
                    dir_pics: str,
                    clear_dir: bool=True,
                    subject_specific: bool=False) -> None:
    '''
    Plots the results saved in dir_results. ConnSearch saves each component
        result as a separate .pkl file in dir_results, and the present
        function loops over all .pkl files in dir_results, loads them, then
        plots them. This function saves the plots in dir_pics.
    The results will be plotted in multiple styles.
    This function can also be used to plot the significant "Wu-components"
        (vector of edges connected to a single ROI).

    :param dir_results: Directory containing the .pkl results files
    :param dir_pics: Directory to save the plots
    :param clear_dir: Whether to clear dir_pics before plotting
    :param subject_specific: bool, if True sets -1 and 1 as vmin/vmax for colors
    :return:
    '''

    clear_make_dir(dir_pics, clear_dir=clear_dir)
    clear_make_dir(os.path.join(dir_pics, 'glass'), clear_dir=clear_dir)
    clear_make_dir(os.path.join(dir_pics, 'chord'), clear_dir=clear_dir)

    chord_params = {}
    if subject_specific:
        chord_params['vmin'] = -1
        chord_params['vmax'] = 1
        glass_params = {'vmin': -1, 'vmax': 1}
    else:
        glass_params = {}

    chord_black_params = {'black_BG': True}

    fns = os.listdir(dir_results)
    print(f'Total number of results to plot: {len(fns)}')
    for fn in tqdm(fns, 'Plotting component results'):
        if fn[-4:] != '.pkl':  # skips any other files that may mistakenly enter
            continue
        fp = os.path.join(dir_results, fn)
        with open(fp, 'rb') as f:
            result = pickle.load(f)
            # Result is a dict. It is built and saved via connsearcher.ConnSearcher.save_component
        title = f'Component Core ROI: {result["component"][0]}'

        # These plot the loaded result
        plot_result(result, dir_pics, fn, chord_params=chord_params,
                    glass_params=glass_params, title=title)
        plot_result(result, dir_pics, fn.replace('.pkl', '_black.pkl'),
                    chord_params=chord_black_params, title=title)

        sum_chord_params = {'alphas': 0.9,  # For averaging network-pair edges
                            'plot_count': True,
                            'norm_thickness': True,
                            'plot_abs_sum': True}
        plot_result(result, dir_pics, fn.replace('.pkl', '_sum.pkl'),
                    chord_params=sum_chord_params, title=title)


def plot_ConnSearch_ROI_scores(dir_results: str,
                               fp_out: str,
                               group_level: bool = False,
                               vmin: Union[None, float, int]=None,
                               vmax: Union[None, float, int]=None,
                               avg_ROIs: bool=True) -> None:
    '''
    Plots the results of all the ConnSearch component models. Each ROI is a dot
        on a glass brain, where its color indicates the score (accuracy or
        t-value) of the component for which the ROI is the Core ROI.
    :param dir_results: str, directory where results will be loaded from
    :param fp_out: str, filepath to save the plot
    :param group_level: if True, sets the default vmin/vmax for group-level
        plotting. if False, sets the default vmin/vmax for subject-level.
    :param vmin: float, minimum value for the colorbar
    :param vmax: float, maximum value for the colorbar
    :param avg_ROIs: bool, if True, assigns ROIs scores as the average of
        all the components they contributed to. if False, assigns ROIs
        scores based solely on the component for which they are the Core ROI.
    :return:
    '''
    fns = os.listdir(dir_results)
    node_vals = []
    node_coords = []
    i2coord = {}
    i2accs = defaultdict(list)

    for fn in fns:  # Each component result is saved in its own .pkl file
        if '.pkl' not in fn:
            continue
        fp = os.path.join(dir_results, fn)
        with open(fp, 'rb') as f:
            result = pickle.load(f)
        score = result['score']
        print(f'{score=}')
        core_i = result['component'][0]
        core_roi_coord = result['coords_component'][0]
        i2coord[core_i] = core_roi_coord
        for j in result['component']:
            i2accs[j].append(score)
        node_vals.append(score)
        node_coords.append(core_roi_coord)

    if avg_ROIs:
        node_vals_avg = []
        node_coords_avg = []
        for i in i2coord:
            node_vals_avg.append(np.mean(i2accs[i]))
            node_coords_avg.append(i2coord[i])
        node_vals = node_vals_avg
        node_coords = node_coords_avg
    else:
        node_vals = np.array(node_vals)
        node_coords = np.array(node_coords)

    if group_level:
        if vmax is None: vmax = .65
        if vmin is None: vmin = .555
    else:
        if vmax is None: vmax = 5.5
        if vmin is None: vmin = 2.5
    plot_ROI_scores(node_vals, node_coords, vmin=vmin, vmax=vmax,
                    fp_out=fp_out, title='', dpi=600)


def plot_result(result: dict,
                dir_out: str,
                fn_out: str,
                chord_params: Union[None, dict] = None,
                glass_params: Union[None, dict] = None,
                title: Union[None, str] = None,
                cmap: Union[None, str, matplotlib.colors.Colormap] = None):
    '''
    Plots a given component. Used to generate the single-component figures
        in the associated manuscript.
    Plotting is slow and will often take more time than running ConnSearch.

    :param result: dict, contains the necessary info to plot a given result
    :param dir_out: str, specifies where the plot should be saved
    :param fn_out: str, specifies the filename of the plot
    :param chord_params: parameters for the chord diagram
    :param glass_params: parameters for the glass brain plot
    :param title: title for the glass + chord combined plot
    :param cmap: colormap to use for the plots
    '''
    if chord_params is None: chord_params = {}
    if glass_params is None: glass_params = {}
    print(f'Plotting: {fn_out=}')
    edges = result['edges']
    idx_to_label, network_order, network_colors = \
        get_network_info(result['coords_all'])

    if result['edge_weights'] is None:
        _, edge_weights = convert_matrix(result['adj'])  # flattens matrix
    else:
        edge_weights = result['edge_weights']  # list of weights, length = #edges

    if ('code' in result) and ('wu' in result['code']):
        is_wu = True # appearance of plots changed slightly for Wu results analysis.
    else:
        is_wu = False

    dir_chord = os.path.join(dir_out, 'chord')
    fp_chord = os.path.join(dir_chord, fn_out.replace('.pkl', '_chord.png'))
    dir_glass = os.path.join(dir_out, 'glass')
    fp_glass = os.path.join(dir_glass, fn_out.replace('.pkl', '_glass.png'))
    fp_combined = os.path.join(dir_out, fn_out.replace('.pkl', '_combined.png'))

    # See NiChord (https://github.com/paulcbogdan/NiChord) for plotting details
    if cmap is None:
        cmap = 'turbo' if not all(w == 1 for w in edge_weights) else 'Greys'
    plot_chord(idx_to_label, edges, edge_weights=edge_weights,
               fp_chord=fp_chord,
               arc_setting=False if is_wu else True,  # Helps Wu plots look nicer
               network_order=network_order, coords=result['coords_all'],
               network_colors=network_colors, cmap=cmap,
               **chord_params)

    plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                    result['coords_all'], network_colors=network_colors,
                    cmap=cmap,
                    **glass_params)
    combine_imgs(fp_glass, fp_chord, fp_combined, title=title)

    fp_combined = fp_combined[:-4] + '_1' + fp_combined[-4:]
    combine_imgs(fp_glass, fp_chord, fp_combined, title=title, only1glass=True,
                 fontsize=75 if (len(title) > 50) else 82)
    print(f'\tPlotted: {fp_combined=}')
    plt.close()


def get_network_info(coords: Union[list, np.ndarray] = None) -> \
        Tuple[dict, list, dict]:
    '''
    Returns info about the ROI network labels and some settings for plotting

    :param coords: list of ROI coordinates. shape (#ROIs, 3)
    :return:
        idx_to_label, dict mapping each ROI's index to a network label
        network_order, list of network labels in order to be listed on diagrams
        network_colors, dict mapping each network label to a color
    '''
    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']
    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    idx_to_label = get_idx_to_label(coords, atlas='yeo',
                                    search_closest=True)
    labels = set(idx_to_label.values())
    do_pop = []
    for network in network_colors:
        if network not in labels:
            do_pop.append(network)
    for network in do_pop:
        del network_colors[network]
        network_order.remove(network)

    return idx_to_label, network_order, network_colors


def plot_ROI_scores(node_vals: list,
                    coords: Union[list, np.ndarray],
                    vmin: Union[None, float, int] = None,
                    vmax: Union[None, float, int] = None,
                    fp_out: Union[None, str] = None,
                    title: Union[None, str] = None,
                    show: bool = False,
                    dpi: int = 600) -> None:
    '''
    Used for generating those plots where each ROI is a dot on a glass brain,
        where the dot's color indicates its score. Score can be the accuracy of
        its corresponding components (ConnSearch) or the number of highly
        predictive edges it is connected to (RFE, NCFS, CPM). This is a
        general function used for plotting the results of all the methods.

    :param node_vals: list of floats, the score for each ROI
    :param coords: list of tuples, the coordinates of each ROI
    :param vmin: float, minimum value for the colorbar
    :param vmax: float, maximum value for the colorbar
    :param fp_out: str, filepath to save the plot
    :param title: str, title of the plot
    :param show: bool, whether to show the plot immediately via
        matplotlib.pyplot.show(). Regardless of this setting, the plot is saved.
    :param dpi: int, resolution of the plot
    :return:
    '''
    if show: dpi = dpi * 2.5 / 6
    print(f'Plotting ROI scores glass brains: {fp_out=}')

    fig = plt.figure(figsize=(8, 2.79), dpi=dpi)

    node_vals_, node_coords_ = zip(*sorted(zip(node_vals, coords),  # plots
                                           key=lambda x: x[0]))  # highest on top

    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                      plt.cm.CMRmap(np.linspace(0.05, 1, 1000)))

    # plots all three glass brain angles (sagittal, coronal, axial)
    plotting.plot_markers(node_vals_, node_coords_,
                          node_cmap=mymap, alpha=1,
                          node_vmin=vmin, node_vmax=vmax,
                          black_bg=True,
                          node_size=3.25,
                          title=title,
                          figure=fig)
    plt.savefig(fp_out)
    if show: plt.show()

    # plots only the sagittal view of the glass brain
    fig = plt.figure(figsize=(4, 4 / (2.6 / 2.3) * .71), dpi=dpi)
    plotting.plot_markers(node_vals_, node_coords_,
                          node_cmap=mymap, alpha=1,
                          node_vmin=vmin, node_vmax=vmax,
                          black_bg=True,
                          node_size=1.75,
                          title=title,
                          display_mode='x',
                          figure=fig)

    if show: plt.show()
    fp_x = fp_out[:-4] + '_x.png'
    plt.savefig(fp_x)
    plt.clf()
