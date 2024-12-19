#!/usr/bin/env python

"""
Omnibenchmark-izes Markek Gagolewski's https://github.com/gagolews/clustering-benchmarks/blob/0e751cc9dfc30f332ea3e3aac2b95ada8fbc266a/clustbench/score.py#L30
"""

import argparse
import os, sys
import numpy as np
import genieclust
import clustbench

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

## works for both predicted and true labels
def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")

    return(data)

def main():
    parser = argparse.ArgumentParser(description='clustbench fastcluster runner')

    parser.add_argument('--clustering.predicted', type=str,
                        help='gz-compressed textfile containing the clustering result labels.', required = True)
    parser.add_argument('--data.true_labels', type=int,
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

    
    truth = load_labels(getattr(args, 'data.true_labels'))
    predicted = load_labels(getattr(args, 'clustering.predicted'))
    name = args.name
    
    if args.metric == 'normalized_clustering_accuracy':
        metric = genieclust.compare_partitions.normalized_clustering_accuracy
    elif args.metric == 'adjusted_fm_score':
        metric = genieclust.compare_partitions.adjusted_fm_score
    elif args.metric not in VALID_METRICS:
        raise ValueError("Invalid metric.")
    else:
        raise ValueError('Valid metric, but not implemented')
    
    
    scores = get_score(results = predicted,
                       predicted = truth,
                       metric = metric,
                       compute_max=True,
                       warn_if_missing=True) 
   
    print(scores)
    np.savetxt(os.path.join(args.output_dir, f"{name}.scores.gz"), scores, delimiter=",")
 

if __name__ == "__main__":
    main()
