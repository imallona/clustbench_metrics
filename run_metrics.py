#!/usr/bin/env python

"""
Omnibenchmark-izes Markek Gagolewski's https://github.com/gagolews/clustering-benchmarks/blob/0e751cc9dfc30f332ea3e3aac2b95ada8fbc266a/clustbench/score.py#L30

Expects 2D clustering result arrays, e.g. partitions with 5 columns. Being k the true num of clusters, both 
  input and outputs will have 5 columns, k-2, k-1, k, k+1 and k+2
"""

import argparse
import os, sys
import numpy as np
import gzip
import genieclust
import clustbench
from clustbench.load_results import labels_list_to_dict

## partition-comparison (predicted vs true)
VALID_METRICS = ['normalized_clustering_accuracy'
                 'adjusted_fm_score',
                 'adjusted_mi_score',
                 'adjusted_rand_score',
                 'compare_partitions',
                 'fm_score',
                 'mi_score',
                 'normalized_clustering_accuracy',
                 'normalized_confusion_matrix',
                 'normalized_mi_score',
                 'normalized_pivoted_accuracy',
                 'pair_sets_index',
                 'rand_score']

## works for true labels only
def load_true_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")

    return(data)

## works for true labels only
def load_predicted_labels(data_file):
    data = np.loadtxt(data_file, ndmin=2, dtype=str,delimiter=",")
    header = data[0,]
    values = data[1:,].astype(int)
    
    
    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    return([header, values])

# ## adapted from https://github.com/gagolews/clustering-benchmarks/blob/0e751cc9dfc30f332ea3e3aac2b95ada8fbc266a/clustbench/score.py#L30
# ## author M Gagolewski

# def get_scores_across_ks(
#     labels,
#     results,
#     metric=genieclust.compare_partitions.normalized_clustering_accuracy,
#     compute_max=True,
#     warn_if_missing=True
# ):
#     """
#     Computes a similarity score between the reference and the predicted partition

#     Takes into account that there can be more than one ground truth partition
#     and ignores the noise points (as explained in the Methodology section
#     of the clustering benchmark framework's website).

#     If ``labels`` is a single label vector, it will be wrapped inside
#     a list. If ``results`` is not a dictionary,
#     `labels_list_to_dict` will be called first.


#     Parameters
#     ----------

#     labels
#         A vector-like object or a list thereof.

#     results
#         A dictionary of clustering results, where
#         ``results[K]`` gives a K-partition.

#     metric : function
#         An external cluster validity measure; defaults to
#         ``genieclust.compare_partitions.normalized_clustering_accuracy``.
#         It will be called like ``metric(y_true, y_pred)``.

#     compute_max : bool
#         Whether to apply ``max`` on the particular similarity scores.

#     warn_if_missing : bool
#         Warn if some ``results[K]`` is required, but missing.

#     Returns
#     -------

#     score : float or array thereof
#         The computed similarity scores. Ultimately, it is a vector of
#         ``metric(y_true[y_true>0], results[max(y_true)][y_true>0])``
#         over all ``y_true`` in ``labels``
#         or the maximum thereof if ``compute_max`` is ``True``.
#     """

#     scores = []
    
#     y_true = labels
#     y_pred = results
    
#     if np.min(y_pred) < 1 or np.max(y_pred) > k:
#         raise ValueError("`results[k]` is not between 1 and k=%d, min is %d and max is %d." % (k, np.min(y_pred), np.max(y_pred)))
    
#     scores.append(metric(y_true[y_true > 0], y_pred[y_true > 0]))

#     if compute_max and len(scores) > 0:
#         return np.nanmax(scores)
#     else:
#         return np.array(scores)


## adapted from https://github.com/gagolews/clustering-benchmarks/blob/0e751cc9dfc30f332ea3e3aac2b95ada8fbc266a/clustbench/score.py#L30
## author M Gagolewski
def get_single_score(
    labels,
    results,
    metric=genieclust.compare_partitions.normalized_clustering_accuracy,
    compute_max=True,
    warn_if_missing=True
):
    """
    Computes a similarity score between the reference and the predicted partition

    Takes into account that there can be more than one ground truth partition
    and ignores the noise points (as explained in the Methodology section
    of the clustering benchmark framework's website).

    If ``labels`` is a single label vector, it will be wrapped inside
    a list. If ``results`` is not a dictionary,
    `labels_list_to_dict` will be called first.


    Parameters
    ----------

    labels
        A vector-like object or a list thereof.

    results
        A dictionary of clustering results, where
        ``results[K]`` gives a K-partition.

    metric : function
        An external cluster validity measure; defaults to
        ``genieclust.compare_partitions.normalized_clustering_accuracy``.
        It will be called like ``metric(y_true, y_pred)``.

    compute_max : bool
        Whether to apply ``max`` on the particular similarity scores.

    warn_if_missing : bool
        Warn if some ``results[K]`` is required, but missing.

    Returns
    -------

    score : float or array thereof
        The computed similarity scores. Ultimately, it is a vector of
        ``metric(y_true[y_true>0], results[max(y_true)][y_true>0])``
        over all ``y_true`` in ``labels``
        or the maximum thereof if ``compute_max`` is ``True``.
    """

    # labels = list(np.array(labels, ndmin=2))

    # if type(results) is not dict:
    #     results = labels_list_to_dict(results)

    scores = []

    k = int(max(labels))
    y_true = labels
    y_pred = results
    
    if np.min(y_pred) < 1 or np.max(y_pred) > k + 2:
        raise ValueError("`results[k]` is not between 1 and k=%d + 2, min is %d and max is %d." % (k, np.min(y_pred), np.max(y_pred)))
    
    scores.append(metric(y_true[y_true > 0], y_pred[y_true > 0]))

    if compute_max and len(scores) > 0:
        return np.nanmax(scores)
    else:
        return np.array(scores)
    
def main():
    parser = argparse.ArgumentParser(description='clustbench fastcluster runner')

    parser.add_argument('--clustering.predicted', type=str,
                        help='gz-compressed textfile containing the clustering result labels.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='gz-compressed textfile containing the true labels.', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.')
    parser.add_argument('--name', type=str, help='name of the dataset', default='clustbench')
    parser.add_argument('--metric', type=str,
                        help='metric',
                        required = True)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    truth = load_true_labels(getattr(args, 'data.true_labels'))
    ks, predicted = load_predicted_labels(getattr(args, 'clustering.predicted'))
    
    name = args.name

    
    if args.metric == 'normalized_clustering_accuracy':
        metric = genieclust.compare_partitions.normalized_clustering_accuracy
    elif args.metric == 'adjusted_fm_score':
        metric = genieclust.compare_partitions.adjusted_fm_score
    elif args.metric not in VALID_METRICS:
        raise ValueError("Invalid metric.")
    else:
        raise ValueError('Valid metric, but not implemented')
    
    res = []
    for i in range(len(ks)):
        k = ks[i][2:] # to remove the 'k=' prepend
        ## the str is to coerce numpy float64 to something easier to write to disk
        res.append(str(get_single_score(results = predicted[:,i,],
                                       labels = truth,
                                       metric = metric,
                                       compute_max=True,
                                       warn_if_missing=True)))
        
    # np.savetxt(os.path.join(args.output_dir, f"{name}.scores.gz"), [scores], delimiter=",")
    ## np.savetext-free alternative
    with gzip.open(os.path.join(args.output_dir, f"{name}.scores.gz"), 'wb') as fh:
        fh.write(','.join(ks).encode())
        fh.write('\n'.encode())
        fh.write(','.join(res).encode())
        fh.write('\n'.encode())
        

if __name__ == "__main__":
    main()
