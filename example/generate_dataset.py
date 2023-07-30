import numpy as np
from nilearn.datasets import fetch_coords_power_2011

def generate_dataset(n_subjects=50, n_sessions=2, n_rois=264,
                     np_seed=0, n_edges_show_effect=100):
    # Create a binary classification dataset with random values
    #   This represents the connectivity matrices (264x264) from 50 participants
    #   who each completed two sessions, and each session yielded two examples.
    np.random.seed(np_seed)
    n_examples_per_session = 2 # Kept at 2 to easily create Y = 0 and 1 examples
    X = np.random.normal(loc=0.0, scale=1.0, size=(n_subjects,
                                                   n_sessions,
                                                   n_examples_per_session,
                                                   n_rois,
                                                   n_rois))
    # Note that some edges may get selected twice, but this will rarely occur
    #   given that there are 264*264 = 69696 edges (including the diagonal).
    # Sadly, np.random.choice, which allows sampling without replacement doesn't
    #   let us sample 2D vectors, (i, j), which would be needed.
    edges = np.random.random_integers(0, n_rois - 1,
                                      size=(2, n_edges_show_effect))
    X[:, :, 0, edges[0], edges[1]] += 1.0
    diags = np.diag_indices(264)
    X[:, :, :, diags[0], diags[1]] = 0. # set diagonal of every matrix to zero

    Y = np.empty((n_subjects, n_sessions, n_examples_per_session))
    Y[:, :, 0] = 0
    Y[:, :, 1] = 1

    coords_record = fetch_coords_power_2011(legacy_format=False).rois
    coords = coords_record[['x', 'y', 'z']].values
    return X, Y, coords