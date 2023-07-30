from copy import deepcopy
from typing import Union

import numpy as np
from scipy.spatial import distance


def get_components(X: Union[None, np.ndarray] = None,
                   comp_size: int = 16,
                   coords: Union[list, np.ndarray] = None,
                   proximal: bool = False,
                   no_components: bool = False):
    '''
    Calls the component function below. Usually returns 2D numpy arrays, shaped
        (#ROIs, comp_size). Each row i contains the indices for the component
        built using Core ROI_i.

    :param X: 5D numpy array representing the dataset, shape =
        (subjects, sessions, examples, ROI0, ROI1).
    :param comp_size: int, size of components
    :param coords: list of 3-element tuples, representing each ROI's coordinate.
                   Only used if proximal=True.
    :param proximal: bool, if True, defines components based on proximity
        to a Core ROI. If False, defines components based on connectivity.
        The manuscript only uses components based on connectivity.
    :param no_components: bool, if True, returns None.
    :return:
    '''
    if no_components or comp_size is None:
        return None
    elif proximal:
        return get_proximal_comps(X=X, comp_size=comp_size, coords=coords)
    else:
        return get_connectivity_comps(X=X, comp_size=comp_size, coords=coords)


def get_connectivity_comps(X: np.ndarray = None,
                           comp_size: int = 16,
                           coords: Union[None, list, np.ndarray] = None) \
        -> np.ndarray:
    '''
    This function defines components based on connectivity, as reported in the
        manuscript.

    :param X: 5D numpy array representing the dataset, shape =
              (subjects, sessions, examples, ROI0, ROI1).
    :param comp_size: int, size of components
    :param coords: required for consistency with other the component function
                   below. Not used.
    :return: 2D numpy array, shaped (#ROIs, comp_size). Each row i contains the
             indices for the component built using Core ROI_i.
    '''
    components = []
    avg_graph = X.mean(axis=(0, 1, 2))
    for i in range(avg_graph.shape[0]):
        connectivities_i = [(i, 1)]  # a node's connectivity with itself is set 1
        for j in range(avg_graph.shape[0]):
            avg = avg_graph[i][j]  # taking the abs(...) has virtually no impact
            connectivities_i.append((j, avg))
        connectivities_i.sort(key=lambda x: x[1])
        connectivities_i.reverse()
        component_i = [j[0] for j in connectivities_i[0:comp_size]]
        components.append(component_i)
    components = np.array(components)
    return components


def get_proximal_comps(X: np.ndarray = None,
                       comp_size: int = 16,
                       coords: Union[None, list, np.ndarray] = None) \
        -> np.ndarray:
    '''
    Not used for the manuscript. This function defines components based on
        proximity to a given Core ROI.
    Permutation_Manager requires that X, size, and coords can all be taken as
        arguments.

    :param X: required argument for consistency with function above. Not used
    :param comp_size: int, size of components
    :param coords: list of 3-element tuples, representing each ROI's coordinate
    :return: 2D numpy array, shaped (#ROIs, comp_size). Each row i contains the
        indices for the component built using Core ROI_i.
    '''
    components = []
    for i, coord_i in enumerate(deepcopy(coords)):
        distances_i = [(i, 0)]  # a node's "distance" to itself is set to 0
        for j, coord_j in enumerate(deepcopy(coords)):
            if i == j:
                continue
            distances_i.append((j, distance.euclidean(coord_i, coord_j)))
        distances_i.sort(key=lambda x: x[1])
        component_i = [j[0] for j in distances_i[0:comp_size]]
        components.append(component_i)
    components = np.array(components)
    return components


def get_none_components(X: np.ndarray = None,
                        comp_size: int = 16,
                        coords: Union[None, list, np.ndarray] = None) -> None:
    '''
    When components are None, this causes ConnSearcher to carry out the analysis
        by Wu et al. (2021).
    This function exists because Permutation_Manager can be used for the Wu
        approach. Running Permutation_Manager requires a function to define the
        components.

    :param X: required argument for consistency with the other functions
    :param comp_size: required argument for consistency with the other functions
    :param coords: required argument for consistency with the other functions
    :return: None
    '''
    return None
