import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoring import scoring 
import MDAnalysis as mda
u=mda.Universe("protein_folding.pdb")
dcd_files = [f'pnas2012-1yrf-WT-345K-protein-{i:03d}.dcd' for i in range(0, 199)]
u.load_new(dcd_files)
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.000200000
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
print (times.shape)
times = times[99::100]
print (times.shape)
Clustering = Clustering()
rmsd_array=np.load("final_rmsd.npy")
print (rmsd_array.shape)
end_to_end_distances=np.load("end_to_end_distances.npy")
Clustering_Algorithms={'KMeans':Clustering.KMeans, 'Agglomerative Clustering':Clustering.AgglomerativeClustering, 'Gaussian Mixture':Clustering.GaussianMixture, 'Birch Clustering':Clustering.BirchClustering}
n_clusters=[2,3,4,5,6]

rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
dist_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
dist_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}

final_protein_rep = np.load("representation_protein.npy")
#plt.rcParams.update({'font.size': 30})
fig_actual_rmsd, axes_actual_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_dist, axes_actual_dist = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_rmsd, axes_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_dist, axes_dist = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')

subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    y_ax=0
    for n in n_clusters:
        cluster_labels=algorithm(final_protein_rep,n)
        rmsd_property,rmsd_label,rmsd_clusters=scoring(rmsd_array,cluster_labels,name,n,0.20,algorithm)
        dist_property,dist_label,dist_clusters=scoring(end_to_end_distances,cluster_labels,name,n,0.20,algorithm)
        rmsd_property_scores[name].append(rmsd_property)
        rmsd_label_scores[name].append(rmsd_label)
        dist_property_scores[name].append(dist_property)
        dist_label_scores[name].append(dist_label)

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

w1 = 0.3
w2 = 0.7

result_average_property = weighted_average_dicts(rmsd_property_scores, dist_property_scores, w1, w2)
result_average_label = weighted_average_dicts(rmsd_label_scores, dist_label_scores, w1, w2)

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
plt.savefig("overall_scores_rmsd_dist.jpg")

