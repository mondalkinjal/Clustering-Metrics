import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Clustering import Clustering
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
Clustering = Clustering()

def scoring(properties, cluster_labels, name, parameter, silly_cutoff, algorithm):
    combined_data=properties
    property_clusters = algorithm(combined_data,np.max(cluster_labels)+1)
    property_score = silhouette_score(combined_data, property_clusters)
    unique_elements = np.unique(property_clusters)
    positions = defaultdict(list)
    k_values = {}
    major_elements = {}
    for i, e in enumerate(property_clusters):
        positions[e].append(i)
    for e in unique_elements:
        fe = len(positions[e]) / len(property_clusters)
        elements_at_positions = cluster_labels[positions[e]]
        count = Counter(elements_at_positions)
        me, count_me = count.most_common(1)[0]
        fme = count_me / len(positions[e])
        k = fe * fme
        k_values[e] = k
        major_elements[e] = me
    inverse_major_elements = defaultdict(list)
    for e, me in major_elements.items():
        inverse_major_elements[me].append(e)
    for me, es in inverse_major_elements.items():
        if len(es) > 1:
            highest_k = max(k_values[e] for e in es)
            for e in es:
                if k_values[e] != highest_k:
                    k_values[e] = 0
    kvalue_score = sum(k_values.values())
    property_clustered_metric = property_score*kvalue_score
    label_clustered_metric = 0
    max_label = np.max(cluster_labels)+1
    for l in range(0,max_label+1):
        positions = np.where(cluster_labels == l)
        len_occur = np.sum(cluster_labels == l)
        frac_positions = len_occur/len(cluster_labels)
        combined_data = properties[positions[0],:]
        if (combined_data.shape[0]>0):
                    
            def silly(k,combined_data):
                if (combined_data.shape[0]>k):
                    final_labels=algorithm(combined_data, k)
                    are_all_elements_same = np.all(final_labels == final_labels[0])
                    if are_all_elements_same:
                        return 0
                    else:
                        sc=silhouette_score(combined_data, final_labels)
                        return sc
                else:
                    return 0
            no=[]
            score=[]
            iterator= [2,3,4,5,6]
            for j in iterator:
                sc=silly(j,combined_data)
                score.append(sc)
                no.append(j)
            max_element=max(score)
            max_index=score.index(max_element)
            iterator_val=iterator[max_index]
            if (max_element>silly_cutoff):
                labels=algorithm(combined_data, iterator_val)
                max_occ_count = np.bincount(labels).max()
                total_elements = len(labels)
                frac_occ = (max_occ_count/total_elements)
                part_score=frac_positions*frac_occ
                label_clustered_metric=label_clustered_metric+part_score
            else:
                m_index=1
                label_clustered_metric=label_clustered_metric+frac_positions
    
    return property_clustered_metric,label_clustered_metric,property_clusters   

