import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from properties import IsingProperties
from scoringDBSCAN import scoring 
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import seaborn as sns

Clustering = Clustering()
Ising_properties = IsingProperties(lattice_size=30)
loaded_array=np.load("representation_microstates.npy")
properties={'Energy':Ising_properties.Energy,'Magnetization':Ising_properties.Magnetization}
algorithm_total=Clustering.DBSCAN_total
algorithm=Clustering.DBSCAN
min_samples_list_total=[800,1800,2800]
eps_list_total=[0.05,0.15,0.25,0.35,0.45]


min_samples_list=[800,1800,2800,3800]
eps_list=[0.40,0.50,0.60,0.70]
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)','(iii)','(iv)']

final_lattice_arr=[]
temperatures = np.linspace(1.0, 5.0, 20)

for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_microstates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)

    final_lattice_arr.append(final_arr.copy())
final_comb_array = np.concatenate(final_lattice_arr, axis=0)
final_comb_array = final_comb_array[99::100, ...]

w1, w2 = 0.5, 0.5  # Adjust these weights as needed

# Initialize 2D arrays to store scores
score_property_matrix = np.zeros((len(min_samples_list), len(eps_list)))
score_label_matrix = np.zeros((len(min_samples_list), len(eps_list)))

temperature=np.arange(len(final_comb_array)) 
plt.rcParams.update({'font.size': 30})
energy=properties['Energy'](final_comb_array)
mag=properties['Magnetization'](final_comb_array)
energy_combined_data = np.array(energy).reshape(-1,1)
mag_combined_data = np.array(mag).reshape(-1,1)
energy_ground_truth,energy_score=algorithm_total(energy_combined_data,min_samples_list_total,eps_list_total)
mag_ground_truth,mag_score=algorithm_total(mag_combined_data,min_samples_list_total,eps_list_total)


fig_actual_energy, axes_actual_energy = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_mag, axes_actual_mag = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')



x_ax=0
for min_samples in min_samples_list:
    
    y_ax=0
    for eps in eps_list:
        cluster_labels=algorithm(loaded_array,eps,min_samples)
        
        ax_actual_mag = axes_actual_mag[x_ax,y_ax]
        scatter_actual_mag = ax_actual_mag.scatter(temperature, mag, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_mag.text(len(final_comb_array)/2, 1.00, min_samples, ha='center', va='center',fontweight='bold')
        ax_actual_mag.grid(True)
        handles_actual_mag, labels_actual_mag = scatter_actual_mag.legend_elements()
        labels_mapped = []
        for label in labels_actual_mag:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_mag.legend(handles_actual_mag, labels_mapped, title="Clusters")

        ax_actual_energy = axes_actual_energy[x_ax,y_ax]
        scatter_actual_energy = ax_actual_energy.scatter(temperature, energy, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_energy.text(len(final_comb_array)/2,-0.25, min_samples, ha='center', va='center',fontweight='bold')
        ax_actual_energy.grid(True)
        handles_actual_energy, labels_actual_energy = scatter_actual_energy.legend_elements()
        labels_mapped = []
        for label in labels_actual_energy:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_energy.legend(handles_actual_energy, labels_mapped, title="Clusters")    

        energy_property,energy_label,energy_clusters=scoring(energy_combined_data,cluster_labels,energy_ground_truth,energy_score,min_samples_list_total,eps_list_total,algorithm_total)
        mag_property,mag_label,mag_clusters=scoring(mag_combined_data,cluster_labels,mag_ground_truth,mag_score,min_samples_list_total,eps_list_total,algorithm_total)
        
        print ("energy_property,energy_label,mag_property,mag_label,min_samples,eps",energy_property,energy_label,mag_property,mag_label,min_samples,eps)
       
        score_property = w1 * energy_property + w2 * mag_property
        score_label = w1 * energy_label + w2 * mag_label

        # Store scores in the matrices
        score_property_matrix[x_ax, y_ax] = score_property
        score_label_matrix[x_ax, y_ax] = score_label 
        
        y_ax=y_ax+1
    x_ax=x_ax+1

xticks = np.linspace(0, len(final_comb_array), 5)
xticklabels = ['1.0', '2.0', '3.0', '4.0', '5.0']

fig_actual_energy.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_energy.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_energy.text(0.001, 0.52, "Energy", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_energy.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_energy.savefig("dbscan_energy_clusters.jpg", bbox_inches='tight')

fig_actual_mag.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_mag.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_mag.text(0.001, 0.52, "Magnetization", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_mag.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_mag.savefig("dbscan_mag_clusters.jpg", bbox_inches='tight')

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
plt.savefig("heatmaps_microstates.jpg")

