import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
#from properties import IsingProperties
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


algorithm_total=Clustering.DBSCAN_total
algorithm=Clustering.DBSCAN
backbone_rmsd_list=[2,10,20]
eps_backbone_rmsd_list=[0.01,0.1,1.00]
ligand_rmsd_list=[20,40,80,160]
eps_ligand_rmsd_list=[0.7,1.00,5.00,10.00,15.00]
distances_array_list=[20,40,80,160]
eps_distances_array_list=[0.7,1.00,5.00,10.00,15.00]

min_samples_list=[1,10,20,40]
eps_list=[1.0,5.0,20.0,40.0]



w1, w2 = 0.5, 0.5  # Adjust these weights as needed

# Initialize 2D arrays to store scores
score_property_matrix = np.zeros((len(min_samples_list), len(eps_list)))
score_label_matrix = np.zeros((len(min_samples_list), len(eps_list)))


backbone_rmsd_ground_truth,backbone_rmsd_score=algorithm_total(backbone_rmsd,backbone_rmsd_list,eps_backbone_rmsd_list)
ligand_rmsd_ground_truth,ligand_rmsd_score=algorithm_total(ligand_rmsd,ligand_rmsd_list,eps_ligand_rmsd_list)
distances_array_ground_truth,distances_array_score=algorithm_total(distances_array,distances_array_list,eps_distances_array_list)



final_protein_rep = np.load("representation_protein.npy")
plt.rcParams.update({'font.size': 30})

w1, w2, w3 = 0.0, 0.50, 0.50

fig_backbone_rmsd, axes_backbone_rmsd = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_ligand_rmsd, axes_ligand_rmsd = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_distances, axes_distances = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')

x_ax=0
for min_samples in min_samples_list:
    
    y_ax=0
    for eps in eps_list:
        cluster_labels=algorithm(final_protein_rep,eps,min_samples)
        ax_backbone_rmsd = axes_backbone_rmsd[x_ax,y_ax]
        scatter_backbone_rmsd=ax_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_backbone_rmsd.text(8, 1.4, min_samples, ha='center', va='center',fontweight='bold')
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
        ax_backbone_rmsd.legend(handles_backbone_rmsd, labels_mapped, title="Clusters",loc='lower left')
        
        ax_ligand_rmsd = axes_ligand_rmsd[x_ax,y_ax]
        scatter_ligand_rmsd=ax_ligand_rmsd.scatter(times, ligand_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_ligand_rmsd.text(7, 40, min_samples, ha='center', va='center',fontweight='bold')
        ax_ligand_rmsd.grid(True)
        handles_ligand_rmsd, labels_ligand_rmsd = scatter_ligand_rmsd.legend_elements()
        labels_mapped = []
        for label in labels_ligand_rmsd:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_ligand_rmsd.legend(handles_ligand_rmsd, labels_mapped, title="Clusters",loc='upper left')

        ax_distances = axes_distances[x_ax,y_ax]
        scatter_distances=ax_distances.scatter(times, distances_array_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_distances.text(7,80, min_samples, ha='center', va='center',fontweight='bold')
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
        ax_distances.legend(handles_distances, labels_mapped, title="Clusters",loc='upper left')
        
        backbone_rmsd_property,backbone_rmsd_label,backbone_rmsd_clusters=scoring(backbone_rmsd,cluster_labels,backbone_rmsd_ground_truth,backbone_rmsd_score,backbone_rmsd_list,eps_backbone_rmsd_list,algorithm_total)
        ligand_rmsd_property,ligand_rmsd_label,ligand_rmsd_clusters=scoring(ligand_rmsd,cluster_labels,ligand_rmsd_ground_truth,ligand_rmsd_score,ligand_rmsd_list,eps_ligand_rmsd_list,algorithm_total)
        distances_property,distances_label,distances_clusters=scoring(distances_array,cluster_labels,distances_array_ground_truth,distances_array_score,distances_array_list,eps_distances_array_list,algorithm_total)
        
        print ("backbone_rmsd_property,backbone_rmsd_label,mana_rmsd_property,mana_rmsd_label,distances_property,distances_label,h_bond_property,h_bond_label,salt_bridge_property,salt_bridge_label,min_samples,eps",backbone_rmsd_property,backbone_rmsd_label,ligand_rmsd_property,ligand_rmsd_label,distances_property,distances_label,min_samples,eps,flush=True)
       
        score_property = w1 * backbone_rmsd_property + w2 * ligand_rmsd_property + w3 * distances_property 
        score_label = w1 * backbone_rmsd_label + w2 * ligand_rmsd_label + w3 * distances_label 

        # Store scores in the matrices
        score_property_matrix[x_ax, y_ax] = score_property
        score_label_matrix[x_ax, y_ax] = score_label 
        
        y_ax=y_ax+1
    x_ax=x_ax+1

xticks=[0,7,15]
xticklabels=['0','7','15']

fig_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_backbone_rmsd.savefig("backbone_rmsd_dbscan.jpg", bbox_inches='tight')

fig_ligand_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_ligand_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_ligand_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_ligand_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_ligand_rmsd.savefig("ligand_rmsd_dbscan.jpg", bbox_inches='tight')

fig_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_distances.savefig("distances_dbscan.jpg", bbox_inches='tight')


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
plt.savefig("heatmaps_dbscan.png")

