import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoring import scoring 
from sklearn.decomposition import PCA
import MDAnalysis as mda
import matplotlib.colors
import matplotlib.ticker as ticker

def dimensionality_reduction(data):
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(data)
    pca_result = pca_result.flatten()
    return pca_result

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
distances_array_reduced=dimensionality_reduction(distances_array)

Clustering_Algorithms={'KMeans':Clustering.KMeans, 'AgglomerativeClustering':Clustering.AgglomerativeClustering, 'GaussianMixture':Clustering.GaussianMixture, 'BirchClustering':Clustering.BirchClustering}

n_clusters=[2,3,4,5,6]

backbone_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
backbone_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
ligand_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
ligand_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}

final_protein_rep = np.load("representation_protein.npy")
plt.rcParams.update({'font.size': 30})
fig_actual_backbone_rmsd, axes_actual_backbone_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_ligand_rmsd, axes_actual_ligand_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_distances, axes_actual_distances = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_backbone_rmsd, axes_backbone_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_ligand_rmsd, axes_ligand_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_distances, axes_distances = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')

subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
xticks=[100,300,500,700]
xticklabels=['100','300','500','700']
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

        label = f"{row_labels[x_ax]}{col_labels[y_ax]}"
        
        ax_actual_backbone_rmsd = axes_actual_backbone_rmsd[x_ax,y_ax]
        scatter_actual_backbone_rmsd=ax_actual_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_backbone_rmsd.text(8, 1.4, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_backbone_rmsd.grid(True)
        handles_actual_backbone_rmsd, labels_actual_backbone_rmsd = scatter_actual_backbone_rmsd.legend_elements()
        ax_actual_backbone_rmsd.legend(handles_actual_backbone_rmsd, labels_actual_backbone_rmsd, title="Clusters",loc='lower left')

        ax_backbone_rmsd = axes_backbone_rmsd[x_ax,y_ax]
        scatter_backbone_rmsd=ax_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=backbone_rmsd_clusters, cmap='viridis', marker='o')
        ax_backbone_rmsd.text(8, 1.4, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_backbone_rmsd.grid(True)
        handles_backbone_rmsd, labels_backbone_rmsd = scatter_backbone_rmsd.legend_elements()
        ax_backbone_rmsd.legend(handles_backbone_rmsd, labels_backbone_rmsd, title="Clusters",loc='lower left')

        ax_actual_ligand_rmsd = axes_actual_ligand_rmsd[x_ax,y_ax]
        scatter_actual_ligand_rmsd=ax_actual_ligand_rmsd.scatter(times, ligand_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_ligand_rmsd.text(7, 40, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_ligand_rmsd.grid(True)
        handles_actual_ligand_rmsd, labels_actual_ligand_rmsd = scatter_actual_ligand_rmsd.legend_elements()
        ax_actual_ligand_rmsd.legend(handles_actual_ligand_rmsd, labels_actual_ligand_rmsd, title="Clusters",loc='upper left')

        ax_ligand_rmsd = axes_ligand_rmsd[x_ax,y_ax]
        scatter_ligand_rmsd=ax_ligand_rmsd.scatter(times, ligand_rmsd_plot, c=ligand_rmsd_clusters, cmap='viridis', marker='o')
        ax_ligand_rmsd.text(7, 40, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_ligand_rmsd.grid(True)
        handles_ligand_rmsd, labels_ligand_rmsd = scatter_ligand_rmsd.legend_elements()
        ax_ligand_rmsd.legend(handles_ligand_rmsd, labels_ligand_rmsd, title="Clusters",loc='upper left')

        ax_actual_distances = axes_actual_distances[x_ax,y_ax]
        scatter_actual_distances=ax_actual_distances.scatter(times, distances_array_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_distances.text(7,80,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_distances.grid(True)
        handles_actual_distances, labels_actual_distances = scatter_actual_distances.legend_elements()
        ax_actual_distances.legend(handles_actual_distances, labels_actual_distances, title="Clusters",loc='upper left')

        ax_distances = axes_distances[x_ax,y_ax]
        scatter_distances=ax_distances.scatter(times, distances_array_reduced, c=dist_clusters, cmap='viridis', marker='o')
        ax_distances.text(7,80,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_distances.grid(True)
        handles_distances, labels_distances = scatter_distances.legend_elements()
        ax_distances.legend(handles_distances, labels_distances, title="Clusters",loc='upper left')

        y_ax=y_ax+1
    x_ax=x_ax+1

xticks=[0,7,15]
xticklabels=['0','7','15']

fig_actual_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_backbone_rmsd.savefig("backbone_rmsd_actual_total.jpg", bbox_inches='tight')

fig_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_backbone_rmsd.savefig("backbone_rmsd_total.jpg", bbox_inches='tight')

fig_actual_ligand_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_ligand_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_ligand_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_ligand_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_ligand_rmsd.savefig("ligand_rmsd_actual_total.jpg", bbox_inches='tight')

fig_ligand_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_ligand_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_ligand_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_ligand_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_ligand_rmsd.savefig("ligand_rmsd_total.jpg", bbox_inches='tight')

fig_actual_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_distances.savefig("distances_actual_total.jpg", bbox_inches='tight')

fig_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_distances.savefig("distances_total.jpg", bbox_inches='tight')


def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(3, 2, figsize=(10, 15), sharex=True)
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

plot_bars(axes[0], backbone_rmsd_property_scores, 'A')
plot_bars(axes[1], backbone_rmsd_label_scores, 'B')
plot_bars(axes[2], ligand_rmsd_property_scores, 'C')
plot_bars(axes[3], ligand_rmsd_label_scores, 'D')
plot_bars(axes[4], distances_property_scores, 'E')
plot_bars(axes[5], distances_label_scores, 'F')

plt.tight_layout()
plt.savefig("scores_property_label_total.jpg")

