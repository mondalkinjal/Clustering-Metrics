import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoring import scoring 
from sklearn.decomposition import PCA
import MDAnalysis as mda
import matplotlib.colors
import matplotlib.ticker as ticker


u=mda.Universe("protein_ligand.gro","final_production.xtc")
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.001
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
Clustering = Clustering()

backbone_rmsd=np.load("backbone_rmsd.npy")
backbone_rmsd=backbone_rmsd.reshape(-1,1)
backbone_rmsd_plot=backbone_rmsd.flatten()
ligand_rmsd=np.load("ligand_rmsd.npy")
ligand_rmsd=ligand_rmsd.reshape(-1,1)
ligand_rmsd_plot=ligand_rmsd.flatten()
distances_array=np.load("distances_array.npy")

Clustering_Algorithms={'KMeans':Clustering.KMeans, 'AgglomerativeClustering':Clustering.AgglomerativeClustering, 'GaussianMixture':Clustering.GaussianMixture, 'BirchClustering':Clustering.BirchClustering}


n_clusters=[2,3,4,5,6]

w1, w2 = 0.5, 0.5

backbone_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
backbone_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
ligand_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
ligand_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}

final_protein_rep = np.load("representation_protein.npy")

x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    y_ax=0
    for n in n_clusters:
        cluster_labels=algorithm(final_protein_rep,n)

        backbone_rmsd_property,backbone_rmsd_label,backbone_rmsd_clusters=scoring(backbone_rmsd,cluster_labels,name,n,0.20,algorithm)
        ligand_rmsd_property,ligand_rmsd_label,ligand_rmsd_clusters=scoring(ligand_rmsd,cluster_labels,name,n,0.20,algorithm)
        distance_property,distance_label,dist_clusters=scoring(distances_array,cluster_labels,name,n,0.20,algorithm)

        backbone_rmsd_property_scores[name].append(backbone_rmsd_property)
        ligand_rmsd_property_scores[name].append(ligand_rmsd_property)
        distances_property_scores[name].append(distance_property)

        backbone_rmsd_label_scores[name].append(backbone_rmsd_label)
        ligand_rmsd_label_scores[name].append(ligand_rmsd_label)
        distances_label_scores[name].append(distance_label)

        
        y_ax=y_ax+1
    x_ax=x_ax+1

def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)
def weighted_average_dicts(d_1, d_2, w1, w2):
    # Ensure that w1 + w2 = 1
    if not (w1 + w2 == 1):
        raise ValueError("Weights w1 and w2 must sum to 1")

    # Initialize the result dictionary
    weighted_avg_dict = {}

    # Iterate over the keys and compute the weighted average
    for key in d_1:
        multiplied_d_1 = [x * w1 for x in d_1[key]]
        multiplied_d_2 = [x * w2 for x in d_2[key]]
        weighted_score = [x + y for x, y in zip(multiplied_d_1,multiplied_d_2)]
        weighted_avg_dict[key] = weighted_score

    return weighted_avg_dict


result_average_property = weighted_average_dicts(ligand_rmsd_property_scores, distances_property_scores, w1, w2)
result_average_label = weighted_average_dicts(ligand_rmsd_label_scores, distances_label_scores, w1, w2)




plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

def plot_bars(ax, scores, title):
    bar_width = 0.2
    bar_positions = np.arange(len(n_clusters))
    for i, alg in enumerate(Clustering_Algorithms):
        ax.bar(bar_positions + i * bar_width, scores[alg], bar_width, label=alg)
    ax.set_title(title.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('Clusters', fontweight='bold')
    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_xticks(bar_positions + bar_width * 1.5)
    ax.set_xticklabels(n_clusters)
    ax.legend()
    ymin, ymax = get_min_max(scores)
    ax.set_ylim(ymin-0.05, ymax+0.05)

plot_bars(axes[0], result_average_property, 'A')
plot_bars(axes[1], result_average_label, 'B')

plt.tight_layout()
plt.savefig("scores_overall_property_label.jpg")

