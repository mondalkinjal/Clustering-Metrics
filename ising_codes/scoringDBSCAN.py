import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Clustering import Clustering
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
Clustering = Clustering()

def scoring(properties, cluster_labels,property_clusters,property_score,min_samples_list_total,eps_list_total, algorithm_total):
    silly_cutoff=0.30
    mask = property_clusters != -1
    length_actual_ground_truth=np.sum(mask)
    unique_elements = np.unique(property_clusters)
    unique_elements = unique_elements[unique_elements != -1]
    positions = defaultdict(list)
    k_values = {}
    major_elements = {}
    for i, e in enumerate(property_clusters):
        if e != -1:
            positions[e].append(i)
    
    for e in unique_elements:
        fe = len(positions[e]) / length_actual_ground_truth
        elements_at_positions = cluster_labels[positions[e]]
        count = Counter(elements_at_positions)
        most_common_elements = count.most_common(2)
        if most_common_elements[0][0] == -1 and len(most_common_elements) > 1:
        # If -1 is the most common, use the second most common element
            me, count_me = most_common_elements[1]
        else:
        # Otherwise, use the most common element as usual
            me, count_me = most_common_elements[0]


#        me, count_me = count.most_common(1)[0]
        if me != -1:
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
    max_label = np.max(cluster_labels)
    if (max_label==0):
        property_clustered_metric = 0

    if (max_label>0):
        for l in range(0,max_label+1):
            positions = np.where(cluster_labels == l)
            len_occur = np.sum(cluster_labels == l)
            frac_positions = len_occur/len(cluster_labels)
            combined_data = properties[positions[0],:]
            if (combined_data.shape[0]>0):
            

                labels,max_element = algorithm_total(combined_data,min_samples_list_total,eps_list_total)
                shifted_labels = labels + 1
                counts = np.bincount(shifted_labels)
                max_occ_count = counts.max()
                max_label = counts.argmax()
                if (max_element>silly_cutoff):
                    total_elements = len(shifted_labels)
                    frac_occ = (max_occ_count/total_elements)
                    part_score=frac_positions*frac_occ
                    label_clustered_metric=label_clustered_metric+part_score
                                   
                else:
                    m_index=1
                    label_clustered_metric=label_clustered_metric+frac_positions
                    
    return property_clustered_metric,label_clustered_metric,property_clusters   

