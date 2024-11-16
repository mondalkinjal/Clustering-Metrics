import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoringDBSCAN import scoring 
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import seaborn as sns
import MDAnalysis as mda
from sklearn.decomposition import PCA

def dimensionality_reduction(data):
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(data)
    pca_result = pca_result.flatten()
    return pca_result

u=mda.Universe("protein_ligand.gro","final_production.xtc")
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.1
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
Clustering = Clustering()

backbone_rmsd=np.load("backbone_rmsd.npy")
backbone_rmsd=backbone_rmsd.reshape(-1,1)
backbone_rmsd_plot=backbone_rmsd.flatten()
print ("backbone_rmsd",backbone_rmsd.shape)
mana_rmsd=np.load("mana_rmsd.npy")
mana_rmsd=mana_rmsd.reshape(-1,1)
mana_rmsd_plot=mana_rmsd.flatten()
distances_array=np.load("distances_array.npy")
distances_array_reduced=dimensionality_reduction(distances_array)
h_bond=np.load("h_bond.npy")
h_bond=h_bond[1:,:]
h_bond_reduced=dimensionality_reduction(h_bond)
salt_bridge=np.load("salt_bridge.npy")
salt_bridge_reduced=dimensionality_reduction(salt_bridge)



algorithm_total=Clustering.DBSCAN_total
algorithm=Clustering.DBSCAN
backbone_rmsd_list=[2,10,20]
eps_backbone_rmsd_list=[0.01,0.1,1.00]
mana_rmsd_list=[2,10,20]
eps_mana_rmsd_list=[0.01,0.05,0.08,0.09]
distances_array_list=[20,40,80,160]
eps_distances_array_list=[0.01,0.1,0.3,0.5,0.7,0.9,1.00,2.00,3.00]
h_bond_list=[280]
eps_h_bond_list=[0.1,1.00]
salt_bridge_list=[280]
eps_salt_bridge_list=[0.1,1.00]


min_samples_list=[10,40,80,160]
eps_list=[0.04,0.05,0.06,0.07,0.09]
score_property_matrix = np.zeros((len(min_samples_list), len(eps_list)))
score_label_matrix = np.zeros((len(min_samples_list), len(eps_list)))


backbone_rmsd_ground_truth,backbone_rmsd_score=algorithm_total(backbone_rmsd,backbone_rmsd_list,eps_backbone_rmsd_list)
mana_rmsd_ground_truth,mana_rmsd_score=algorithm_total(mana_rmsd,mana_rmsd_list,eps_mana_rmsd_list)
distances_array_ground_truth,distances_array_score=algorithm_total(distances_array,distances_array_list,eps_distances_array_list)
h_bond_ground_truth,h_bond_score=algorithm_total(h_bond,h_bond_list,eps_h_bond_list)
salt_bridge_ground_truth,salt_bridge_score=algorithm_total(salt_bridge,salt_bridge_list,eps_salt_bridge_list)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

scatter1 = ax1.scatter(times, backbone_rmsd_plot, c=backbone_rmsd_ground_truth, cmap='viridis', marker='o')
ax1.set_ylabel("RMSD Ground Truth", fontsize=12, fontweight='bold')
ax1.grid(True)
legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
ax1.add_artist(legend1)

# Scatter plot for times vs dist_ground_truth with clusters as colors on the second subplot
scatter2 = ax2.scatter(times, mana_rmsd_plot, c=mana_rmsd_ground_truth, cmap='viridis', marker='o')
ax2.set_xlabel("Time (in µs)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Distance Ground Truth", fontsize=12, fontweight='bold')
ax2.grid(True)
legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
ax2.add_artist(legend2)


scatter3 = ax3.scatter(times, distances_array_reduced, c=distances_array_ground_truth, cmap='viridis', marker='o')
ax3.set_xlabel("Time (in µs)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Distance Ground Truth", fontsize=12, fontweight='bold')
ax3.grid(True)
legend3 = ax3.legend(*scatter3.legend_elements(), title="Clusters")
ax3.add_artist(legend3)

scatter4 = ax4.scatter(times, h_bond_reduced, c=h_bond_ground_truth, cmap='viridis', marker='o')
ax4.set_xlabel("Time (in µs)", fontsize=12, fontweight='bold')
ax4.set_ylabel("Distance Ground Truth", fontsize=12, fontweight='bold')
ax4.grid(True)
legend4 = ax4.legend(*scatter4.legend_elements(), title="Clusters")
ax4.add_artist(legend4)

scatter5 = ax5.scatter(times, salt_bridge_reduced, c=salt_bridge_ground_truth, cmap='viridis', marker='o')
ax5.set_xlabel("Time (in µs)", fontsize=12, fontweight='bold')
ax5.set_ylabel("Distance Ground Truth", fontsize=12, fontweight='bold')
ax5.grid(True)
legend5 = ax5.legend(*scatter5.legend_elements(), title="Clusters")
ax5.add_artist(legend5)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("ground_truth_5x1_plot.jpg", dpi=150)
plt.show()

final_protein_rep = np.load("representation_protein.npy")
plt.rcParams.update({'font.size': 30})

w1, w2, w3, w4, w5 = 0.0, 0.35, 0.35, 0.15, 0.15

fig_backbone_rmsd, axes_backbone_rmsd = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_mana_rmsd, axes_mana_rmsd = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_distances, axes_distances = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_hbond, axes_hbond = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_saltbridge, axes_saltbridge = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')

x_ax=0
for min_samples in min_samples_list:
    
    y_ax=0
    for eps in eps_list:
        cluster_labels=algorithm(final_protein_rep,eps,min_samples)
        
        ax_backbone_rmsd = axes_backbone_rmsd[x_ax,y_ax]
        scatter_backbone_rmsd=ax_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_backbone_rmsd.text(375, 4, min_samples, ha='center', va='center',fontweight='bold')
        ax_backbone_rmsd.grid(True)
        handles_backbone_rmsd, labels_backbone_rmsd = scatter_backbone_rmsd.legend_elements()
        labels_mapped = []
        for label in labels_backbone_rmsd:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_backbone_rmsd.legend(handles_backbone_rmsd, labels_mapped, title="Clusters",loc='lower right')
        
        ax_mana_rmsd = axes_mana_rmsd[x_ax,y_ax]
        scatter_mana_rmsd=ax_mana_rmsd.scatter(times, mana_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_mana_rmsd.text(375, 32, min_samples, ha='center', va='center',fontweight='bold')
        ax_mana_rmsd.grid(True)
        handles_mana_rmsd, labels_mana_rmsd = scatter_mana_rmsd.legend_elements()
        labels_mapped = []
        for label in labels_mana_rmsd:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_mana_rmsd.legend(handles_mana_rmsd, labels_mapped, title="Clusters",loc='lower right')

        ax_distances = axes_distances[x_ax,y_ax]
        scatter_distances=ax_distances.scatter(times, distances_array_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_distances.text(375,25, min_samples, ha='center', va='center',fontweight='bold')
        ax_distances.grid(True)
        handles_distances, labels_distances = scatter_distances.legend_elements()
        labels_mapped = []
        for label in labels_distances:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_distances.legend(handles_distances, labels_mapped, title="Clusters",loc='lower right')
        
        ax_hbond = axes_hbond[x_ax,y_ax]
        scatter_hbond=ax_hbond.scatter(times, h_bond_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_hbond.text(375,0.8, min_samples, ha='center', va='center',fontweight='bold')
        ax_hbond.grid(True)
        handles_hbond, labels_hbond = scatter_hbond.legend_elements()
        labels_mapped = []
        for label in labels_hbond:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_hbond.legend(handles_hbond, labels_mapped, title="Clusters",loc='lower right')
        
        ax_saltbridge = axes_saltbridge[x_ax,y_ax]
        scatter_saltbridge=ax_saltbridge.scatter(times, salt_bridge_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_saltbridge.text(375,0.8,min_samples, ha='center', va='center',fontweight='bold')
        ax_saltbridge.grid(True)
        handles_saltbridge, labels_saltbridge = scatter_saltbridge.legend_elements()
        labels_mapped = []
        for label in labels_saltbridge:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_saltbridge.legend(handles_saltbridge, labels_mapped, title="Clusters",loc='lower right')
        
        backbone_rmsd_property,backbone_rmsd_label,backbone_rmsd_clusters=scoring(backbone_rmsd,cluster_labels,backbone_rmsd_ground_truth,backbone_rmsd_score,backbone_rmsd_list,eps_backbone_rmsd_list,algorithm_total)
        mana_rmsd_property,mana_rmsd_label,mana_rmsd_clusters=scoring(mana_rmsd,cluster_labels,mana_rmsd_ground_truth,mana_rmsd_score,mana_rmsd_list,eps_mana_rmsd_list,algorithm_total)
        distances_property,distances_label,distances_clusters=scoring(distances_array,cluster_labels,distances_array_ground_truth,distances_array_score,distances_array_list,eps_distances_array_list,algorithm_total)
        h_bond_property,h_bond_label,h_bond_clusters=scoring(h_bond,cluster_labels,h_bond_ground_truth,h_bond_score,h_bond_list,eps_h_bond_list,algorithm_total)
        salt_bridge_property,salt_bridge_label,salt_bridge_clusters=scoring(salt_bridge,cluster_labels,salt_bridge_ground_truth,salt_bridge_score,salt_bridge_list,eps_salt_bridge_list,algorithm_total)
        score_property = w1 * backbone_rmsd_property + w2 * mana_rmsd_property + w3 * distances_property + w4 * h_bond_property + w5 * salt_bridge_property
        score_label = w1 * backbone_rmsd_label + w2 * mana_rmsd_label + w3 * distances_label + w4 * h_bond_label + w5 * salt_bridge_label

        # Store scores in the matrices
        score_property_matrix[x_ax, y_ax] = score_property
        score_label_matrix[x_ax, y_ax] = score_label 
        
        y_ax=y_ax+1
    x_ax=x_ax+1

xticks=[100,300,500,700]
xticklabels=['100','300','500','700']

fig_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_backbone_rmsd.savefig("backbone_rmsd_actual_dbscan.jpg", bbox_inches='tight')

fig_mana_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_mana_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_mana_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_mana_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_mana_rmsd.savefig("mana_rmsd_actual_dbscan.jpg", bbox_inches='tight')

fig_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_distances.savefig("distances_dbscan.jpg", bbox_inches='tight')

fig_hbond.tight_layout(h_pad=0.0, w_pad=0.0)
fig_hbond.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_hbond.text(0.001, 0.52, "hbond", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_hbond.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_hbond.savefig("hbond_dbscan.jpg", bbox_inches='tight')

fig_saltbridge.tight_layout(h_pad=0.0, w_pad=0.0)
fig_saltbridge.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_saltbridge.text(0.001, 0.52, "Salt-Bridge", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_saltbridge.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_saltbridge.savefig("saltbridge_dbscan.jpg", bbox_inches='tight')

fig_scores, axes_scores = plt.subplots(1, 2, figsize=(14, 10))

fontsize = 18  # Adjust this value to control font size of titles and axis labels
tick_fontsize = 18  # Adjust this value for tick labels
cbar_fontsize = 18
# Plot score_property heatmap
sns_heatmap1=sns.heatmap(score_property_matrix, ax=axes_scores[0], cmap='coolwarm', annot=False, fmt=".2f", xticklabels=eps_list, yticklabels=min_samples_list)
axes_scores[0].set_title("A", fontsize=fontsize, fontweight='bold')
axes_scores[0].set_xlabel(r'$\epsilon$', fontsize=fontsize, fontweight='bold')
axes_scores[0].set_ylabel("min-points", fontsize=fontsize, fontweight='bold')
axes_scores[0].set_xticklabels(eps_list, fontsize=tick_fontsize)
axes_scores[0].set_yticklabels(min_samples_list, fontsize=tick_fontsize)
cbar_prop = sns_heatmap1.collections[0].colorbar
cbar_prop.ax.tick_params(labelsize=cbar_fontsize)


sns_heatmap2=sns.heatmap(score_label_matrix, ax=axes_scores[1], cmap='coolwarm', annot=False, fmt=".2f", xticklabels=eps_list, yticklabels=min_samples_list)
axes_scores[1].set_title("B", fontsize=fontsize, fontweight='bold')
axes_scores[1].set_xlabel(r'$\epsilon$', fontsize=fontsize, fontweight='bold')
axes_scores[1].set_ylabel("min-points", fontsize=fontsize, fontweight='bold')

axes_scores[1].set_xticklabels(eps_list, fontsize=tick_fontsize)
axes_scores[1].set_yticklabels(min_samples_list, fontsize=tick_fontsize)
cbar_label = sns_heatmap2.collections[0].colorbar
cbar_label.ax.tick_params(labelsize=cbar_fontsize)
plt.tight_layout()
plt.savefig("heatmaps_smlt.jpg")

