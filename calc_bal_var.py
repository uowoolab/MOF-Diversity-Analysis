#!/usr/bin/env python3

import numpy as np
import math
from math import pi
import pandas as pd
from scipy.stats import entropy
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def cluster(data_, features, epsilon):
    """ Cluster the data using HDBSCAN """
    vals = data_[features].values 
    clusterer = hdbscan.HDBSCAN(
    min_samples=1,
    min_cluster_size=2, cluster_selection_method='eom',
    cluster_selection_epsilon=epsilon)
    labels = clusterer.fit_predict(vals)
    counter = -1
    # Separate each unclustered point into its own bin
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = counter
            counter -= 1
    outlier_score = clusterer.outlier_scores_
    data_['cluster_id'] = labels
    data_['outlier'] = outlier_score
    data_['prob'] = clusterer.probabilities_
    return data_

def get_balance(data_, subset):
    # Calculate the Pielou evenness of the subset
    vals = data_.loc[data_["dataset"] == subset]['cluster_id']
    bin_ids, counts = np.unique(vals, return_counts=True)
    num_bins = len(counts)
    dist_max_ent = np.ones(num_bins)
    dist_min_ent = np.zeros(num_bins)
    dist_min_ent[0] = 1
    norm_f = 1 - np.exp(-entropy(dist_min_ent, qk=dist_max_ent))
    s = 1 - np.exp(-entropy(counts, qk=dist_max_ent))
    balance_subset = (1 - (s / norm_f))
    # Calculate the Pielou evenness of the whole set
    vals = data_['cluster_id']
    bin_ids, counts = np.unique(vals, return_counts=True)
    num_bins = len(counts)
    dist_max_ent = np.ones(num_bins)
    dist_min_ent = np.zeros(num_bins)
    dist_min_ent[0] = 1
    norm_f = 1 - np.exp(-entropy(dist_min_ent, qk=dist_max_ent))
    s = 1 - np.exp(-entropy(counts, qk=dist_max_ent))
    balance_set = (1 - (s / norm_f))

    return balance_subset, balance_set

def get_variety(data_, subset):
    # Calculate the variety of the subset and set
    variety_subset = (len(set(data_.loc[data_['dataset'] == subset]['cluster_id'])) /
                      len(set(data_['cluster_id'])))
    variety_set = (len(set(data_['cluster_id'])) / len(set(data_['cluster_id'])))

    return variety_subset, variety_set


if __name__ == '__main__':
    
    fulldescription = """
        Calculate the balance and variety of a dataset using a set of descriptors.
        Since this code relies on comparing a subset to a superset, the user must give
        the name of the subset label (which must be under a column named "dataset"). Thus,
        your csv file should have only the columns for your descriptors, a data label
        (e.g., filename), and a column called "dataset" which indicates whether the point
        corresponds to a point in the subset or superset. The value of epsilon is defined
	in the HDBSCAN documentation. One method of tuning this parameter is by inspecting
	the members of each cluster.
        """

    parser = argparse.ArgumentParser(description=fulldescription)
    parser.add_argument('csv_file', type=str,
                        help='CSV file containing the descriptors to use')
    parser.add_argument('label_column', type=str,
                        help='The name of the csv column with the labels for the data (e.g., filename)')
    parser.add_argument('subset', type=str,
                        help='The name of the subset in the "dataset" column')
    parser.add_argument('epsilon', type=float,
                        help='The epsilon value used in HDBSCAN clustering (cutoff).')

    args = parser.parse_args()
    csv_file = args.csv_file
    label_column = args.label_column
    subset = args.subset
    epsilon = args.epsilon

    data = pd.read_csv(csv_file)

    # Assume descriptors are all columns except the cluster and dataset labels
    descriptors = data.drop(['filename', 'dataset'], axis=1).columns

    # Get balance
    new_data = cluster(data_=data, features=descriptors, epsilon=epsilon)
    new_data.to_csv('data_with_clusters.csv')
    balance_subset, balance_set = get_balance(data_=new_data, subset=subset)

    # Get variety
    variety_subset, variety_set = get_variety(data_=new_data, subset=subset)
    print("\nBalance of subset: {}\nBalance of set: {}\n".format(balance_subset, balance_set))
    print("Variety of subset: {}\nVariety of set: {}\n".format(variety_subset, variety_set))

    with open('diversity_metrics.out', 'w') as f:
        f.write("Balance of subset: {}\nBalance of set: {}\n".format(balance_subset, balance_set))
        f.write("Variety of subset: {}\nVariety of subset: {}\n".format(variety_subset, variety_set))
        f.close()



