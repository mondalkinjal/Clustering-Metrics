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
times = times[99::100]
Clustering = Clustering()
rmsd_array=np.load("final_rmsd.npy")
end_to_end_distances=np.load("end_to_end_distances.npy")
Clustering_Algorithms={'KMeans':Clustering.KMeans, 'Agglomerative Clustering':Clustering.AgglomerativeClustering, 'Gaussian Mixture':Clustering.GaussianMixture, 'Birch Clustering':Clustering.BirchClustering}
n_clusters=[2,3,4,5,6]

rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
dist_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
dist_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}

final_protein_rep = np.load("representation_protein.npy")
plt.rcParams.update({'font.size': 30})
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

        label = f"{row_labels[x_ax]}{col_labels[y_ax]}"
        
        ax_actual_rmsd = axes_actual_rmsd[x_ax,y_ax]
        scatter_actual_rmsd = ax_actual_rmsd.scatter(times, rmsd_array, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_rmsd.text(200, 17.50, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_rmsd.grid(True)
        handles_actual_rmsd, labels_actual_rmsd = scatter_actual_rmsd.legend_elements()
        ax_actual_rmsd.legend(handles_actual_rmsd, labels_actual_rmsd, title="Clusters",loc='upper left')

        ax_rmsd = axes_rmsd[x_ax,y_ax]
        scatter_rmsd = ax_rmsd.scatter(times, rmsd_array, c=rmsd_clusters, cmap='viridis', marker='o')
        ax_rmsd.text(200, 17.50, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_rmsd.grid(True)
        handles_rmsd, labels_rmsd = scatter_rmsd.legend_elements()
        ax_rmsd.legend(handles_rmsd, labels_rmsd, title="Clusters",loc='upper left')

        ax_actual_dist = axes_actual_dist[x_ax,y_ax]
        scatter_actual_dist = ax_actual_dist.scatter(times, end_to_end_distances, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_dist.text(200, 40.0, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_dist.grid(True)
        handles_actual_dist, labels_actual_dist = scatter_actual_dist.legend_elements()
        ax_actual_dist.legend(handles_actual_dist, labels_actual_dist, title="Clusters",loc='upper left')

        ax_dist = axes_dist[x_ax,y_ax]
        scatter_dist = ax_dist.scatter(times, end_to_end_distances, c=dist_clusters, cmap='viridis', marker='o')
        ax_dist.text(200, 40.0, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_dist.grid(True)
        handles_dist, labels_dist = scatter_dist.legend_elements()
        ax_dist.legend(handles_dist, labels_dist, title="Clusters",loc='upper left')
        
        y_ax=y_ax+1
    x_ax=x_ax+1

xticks = [0, 100, 200, 300, 400]
xticklabels = ['0', '100', '200', '300', '400']

fig_actual_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_rmsd.text(0.5, 0.0005, "Time(in µs)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_rmsd.savefig("actual_rmsd.jpg", bbox_inches='tight')

fig_actual_dist.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_dist.text(0.5, 0.0005, "Time(in µs)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_dist.text(0.001, 0.52, "End to End Distance", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_dist.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_dist.savefig("actual_dist.jpg", bbox_inches='tight')

fig_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_rmsd.text(0.5, 0.0005, "Time(in µs)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_rmsd.savefig("rmsd.jpg", bbox_inches='tight')

fig_dist.tight_layout(h_pad=0.0, w_pad=0.0)
fig_dist.text(0.5, 0.0005, "Time(in µs)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_dist.text(0.001, 0.52, "End to End Distance", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_dist.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_dist.savefig("dist.jpg", bbox_inches='tight')

def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
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

plot_bars(axes[0], rmsd_property_scores, 'A')
plot_bars(axes[1], rmsd_label_scores, 'B')
plot_bars(axes[2], dist_property_scores, 'C')
plot_bars(axes[3], dist_label_scores, 'D')

plt.tight_layout()
plt.savefig("scores_rmsd_dist.jpg")

