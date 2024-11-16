import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import argparse
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd

def replacement_function(data):
    Y = np.zeros_like(data)

# Dictionary to track replacements
    replacement_dict = {}
    current_max = -1

# Iterate through each element in X
    for i, value in enumerate(data):
#       print (i,value)
        if value < 0:
            Y[i] = value
        else:
            if value in replacement_dict:
        # If the value is already in the replacement dictionary, use its replacement
                Y[i] = replacement_dict[value]
            else:
        # If the value is not in the dictionary, create a new replacement
                current_max += 1
                replacement_dict[value] = current_max
                Y[i] = current_max
    return Y 

# Display the result
class Clustering:
    def __init__(self):
        pass        
    def KMeans(self, data, n_clusters):
        km = KMeans(n_clusters=n_clusters, init= 'random', max_iter=300, tol=1e-04, random_state=0)
        clusters = km.fit(data)
        labels = clusters.labels_
        new_labels = replacement_function(labels)
        return new_labels
    def AgglomerativeClustering(self, data, n_clusters):
        km = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = km.fit(data)
        labels = clusters.labels_
        new_labels = replacement_function(labels)
        return new_labels
    def GaussianMixture(self, data, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        gmm.fit(data)
        cluster_labels = gmm.predict(data)
        new_labels = replacement_function(cluster_labels)
        return (new_labels)
    def BirchClustering(self, data, n_clusters):
        km = Birch(n_clusters=n_clusters,threshold=0.01)
        clusters = km.fit(data)
        labels = clusters.labels_
        new_labels = replacement_function(labels)
        return new_labels
    def DBSCAN_total(self, data, min_samples_list, eps_list):
        results = []
        for min_samples in min_samples_list:
            for eps in eps_list:
                km = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                clusters = km.fit(data)
                labels = clusters.labels_
                clustered_mask = labels != -1
                data_clustering = data[clustered_mask]
                data_clusters = labels[clustered_mask]

                max_cluster = np.max(labels)+1

                if (max_cluster>1 and max_cluster<len(data_clustering)):
                    score=silhouette_score(data_clustering, data_clusters)
                    fraction = len(data_clustering) / len(data)
                    score=score*fraction
                else:
                    score=0
                    fraction=0
                results.append({'eps': eps, 'min_samples': min_samples, 'score': score, 'fraction': fraction})
        df_results = pd.DataFrame(results)
        max_score = df_results['score'].max()
        best_params = df_results[df_results['score'] == max_score]
        best_combination = best_params.nsmallest(1, 'eps')
        best_eps = best_combination['eps'].values[0]
        best_min_samples = best_combination['min_samples'].values[0]
        km = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean')
        clusters = km.fit(data)
        labels = clusters.labels_
        new_labels = replacement_function(labels)
         
        return new_labels,max_score

    def DBSCAN(self, data, eps, min_samples):
        
        km = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        clusters = km.fit(data)
        labels = clusters.labels_
        new_labels = replacement_function(labels)
        return new_labels
