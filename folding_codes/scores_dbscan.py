import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoringDBSCAN import scoring 
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import seaborn as sns
import MDAnalysis as mda
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

Clustering = Clustering()
u=mda.Universe("protein_folding.pdb")
dcd_files = [f'pnas2012-1yrf-WT-345K-protein-{i:03d}.dcd' for i in range(0, 199)]
u.load_new(dcd_files)
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.000200135
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
times = times[99::100]
loaded_array = np.load("final_rmsd.npy")
rmsd_array_plot=np.load("final_rmsd.npy")
end_to_end_distances_plot=np.load("end_to_end_distances.npy")

rmsd_array = rmsd_array_plot.reshape(-1,1)
end_to_end_distances = end_to_end_distances_plot.reshape(-1,1)
loaded_array = loaded_array.reshape(-1,1)

algorithm_total=Clustering.DBSCAN_total
algorithm=Clustering.DBSCAN
min_samples_list_rmsd=[150,250,350,450,900,1000]
eps_list_rmsd=[0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,0.9]
min_samples_list_dist=[20,40,60,80,100,120,140,150,250,350,450,900,1000]
eps_list_dist=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]

min_samples_list=[150,400,650,900]
eps_list=[0.1,0.3,0.5,0.7]
row_labels = ['A', 'B', 'C', 'D' ,'E' ,'F','G']
col_labels = ['(i)', '(ii)','(iii)','(iv)','(v)','(vi)','7','8','9','10','11','12','13','14','15']

w1, w2 = 0.3, 0.7  # Adjust these weights as needed

# Initialize 2D arrays to store scores
score_property_matrix = np.zeros((len(min_samples_list), len(eps_list)))
score_label_matrix = np.zeros((len(min_samples_list), len(eps_list)))


rmsd_ground_truth,rmsd_score=algorithm_total(rmsd_array,min_samples_list_rmsd,eps_list_rmsd)
dist_ground_truth,dist_score=algorithm_total(end_to_end_distances,min_samples_list_dist,eps_list_dist)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

scatter1 = ax1.scatter(times, rmsd_array_plot, c=rmsd_ground_truth, cmap='viridis', marker='o')
ax1.set_ylabel("RMSD Ground Truth", fontsize=12, fontweight='bold')
ax1.grid(True)
legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
ax1.add_artist(legend1)

# Scatter plot for times vs dist_ground_truth with clusters as colors on the second subplot
scatter2 = ax2.scatter(times, end_to_end_distances_plot, c=dist_ground_truth, cmap='viridis', marker='o')
ax2.set_xlabel("Time (in µs)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Distance Ground Truth", fontsize=12, fontweight='bold')
ax2.grid(True)
legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
ax2.add_artist(legend2)

# Adjust layout and save the plot
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("rmsd_dist_ground_truth_2x1_plot.jpg", dpi=150)
plt.show()

plt.rcParams.update({'font.size': 30})
fig_actual_rmsd, axes_actual_rmsd = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_dist, axes_actual_dist = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')

x_ax=0
for min_samples in min_samples_list:
    
    y_ax=0
    for eps in eps_list:
        cluster_labels=algorithm(loaded_array,eps,min_samples)

        
        ax_actual_rmsd = axes_actual_rmsd[x_ax,y_ax]
        scatter_actual_rmsd = ax_actual_rmsd.scatter(times, rmsd_array_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_rmsd.text(200, 17.5, min_samples, ha='center', va='center',fontweight='bold')
        handles_actual_rmsd, labels_actual_rmsd = scatter_actual_rmsd.legend_elements()
        labels_mapped = []
        for label in labels_actual_rmsd:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(float(number_str))
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_rmsd.legend(handles_actual_rmsd, labels_mapped, title="Clusters", loc='upper left')

        ax_actual_dist = axes_actual_dist[x_ax,y_ax]
        scatter_actual_dist = ax_actual_dist.scatter(times, end_to_end_distances_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_dist.text(200, 40.0, min_samples, ha='center', va='center',fontweight='bold')
        handles_actual_dist, labels_actual_dist = scatter_actual_dist.legend_elements()
        labels_mapped = []
        for label in labels_actual_dist:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(float(number_str))
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_dist.legend(handles_actual_dist, labels_mapped, title="Clusters", loc='upper left')    

        rmsd_property,rmsd_label,rmsd_clusters=scoring(rmsd_array,cluster_labels,rmsd_ground_truth,rmsd_score,min_samples_list_rmsd,eps_list_rmsd,algorithm_total)
        dist_property,dist_label,dist_clusters=scoring(end_to_end_distances,cluster_labels,dist_ground_truth,dist_score,min_samples_list_dist,eps_list_dist,algorithm_total)
        
        print ("rmsd_property,rmsd_label,dist_property,dist_label,min_samples,eps,max_clusters",rmsd_property,rmsd_label,dist_property,dist_label,min_samples,eps,np.max(cluster_labels)+1,flush=True)
       
        score_property = w1 * rmsd_property + w2 * dist_property
        score_label = w1 * rmsd_label + w2 * dist_label

        # Store scores in the matrices
        score_property_matrix[x_ax, y_ax] = score_property
        score_label_matrix[x_ax, y_ax] = score_label 
        
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
fig_actual_rmsd.savefig("rmsd_actual.jpg", bbox_inches='tight')

fig_actual_dist.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_dist.text(0.5, 0.0005, "Time(in µs)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_dist.text(0.001, 0.52, "End to End Distance", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_dist.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_dist.savefig("dist_actual.jpg", bbox_inches='tight')

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
plt.savefig("heatmaps_folding.png")

