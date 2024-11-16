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
lattice_size=30
Ising_properties = IsingProperties(lattice_size=30)
loaded_array=np.load("representation_macrostates.npy")
properties={'specific_heat':Ising_properties.specific_heat,'sus':Ising_properties.sus}
algorithm_total=Clustering.DBSCAN_total
algorithm=Clustering.DBSCAN
min_samples_list_total=[2,3,4,5,6,7,8,9,10]
eps_list_total=[0.00001,0.00005,0.0001,0.0002,0.0004]


min_samples_list=[2,10,20,40]
eps_list=[0.40,0.45,0.50,0.55]
row_labels = ['A', 'B', 'C', 'D' ,'E' ,'F']
col_labels = ['(i)', '(ii)','(iii)','(iv)']

final_lattice_arr=[]
temperatures = np.linspace(1.0, 5.0, 200)

for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_macrostates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)

    final_lattice_arr.append(final_arr.copy())
final_comb_array = np.stack(final_lattice_arr, axis=0)
final_comb_array = final_comb_array[:,99::100,:,:]

w1, w2 = 0.5, 0.5  # Adjust these weights as needed

# Initialize 2D arrays to store scores
score_property_matrix = np.zeros((len(min_samples_list), len(eps_list)))
score_label_matrix = np.zeros((len(min_samples_list), len(eps_list)))

temperature=np.arange(len(final_comb_array)) 
plt.rcParams.update({'font.size': 30})
cv=properties['specific_heat'](final_comb_array,temperatures)
sus=properties['sus'](final_comb_array,temperatures)
cv_combined_data = np.array(cv).reshape(-1,1)
sus_combined_data = np.array(sus).reshape(-1,1)
cv_ground_truth,cv_score=algorithm_total(cv_combined_data,min_samples_list_total,eps_list_total)
sus_ground_truth,sus_score=algorithm_total(sus_combined_data,min_samples_list_total,eps_list_total)


fig_actual_cv, axes_actual_cv = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_sus, axes_actual_sus = plt.subplots(len(min_samples_list), len(eps_list), figsize=(45, 40), sharex='all', sharey='all')



x_ax=0
for min_samples in min_samples_list:
    
    y_ax=0
    for eps in eps_list:
        cluster_labels=algorithm(loaded_array,eps,min_samples)
        
        ax_actual_cv = axes_actual_cv[x_ax,y_ax]
        cvwhole=cv*lattice_size*lattice_size
        suswhole=sus*lattice_size*lattice_size
        scatter_actual_cv = ax_actual_cv.scatter(temperature, cvwhole, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_cv.text(len(final_comb_array)/2, 2.0, min_samples, ha='center', va='center',fontweight='bold')
        ax_actual_cv.grid(True)
        handles_actual_cv, labels_actual_cv = scatter_actual_cv.legend_elements()
        labels_mapped = []
        for label in labels_actual_cv:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_cv.legend(handles_actual_cv, labels_mapped, title="Clusters")

        ax_actual_sus = axes_actual_sus[x_ax,y_ax]
        scatter_actual_sus = ax_actual_sus.scatter(temperature, suswhole, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_sus.text(len(final_comb_array)/2,20.0, min_samples, ha='center', va='center',fontweight='bold')
        ax_actual_sus.grid(True)
        handles_actual_sus, labels_actual_sus = scatter_actual_sus.legend_elements()
        labels_mapped = []
        for label in labels_actual_sus:
            match = re.search(r'\$\\mathdefault\{(.+?)\}\$', label)
            if match:
                number_str = match.group(1)
                number_str = number_str.replace('−', '-')
                cluster_label = int(number_str)
                descriptive_label = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
                labels_mapped.append(descriptive_label)
            else:
                labels_mapped.append(label)
        ax_actual_sus.legend(handles_actual_sus, labels_mapped, title="Clusters")    

        cv_property,cv_label,cv_clusters=scoring(cv_combined_data,cluster_labels,cv_ground_truth,cv_score,min_samples_list_total,eps_list_total,algorithm_total)
        sus_property,sus_label,sus_clusters=scoring(sus_combined_data,cluster_labels,sus_ground_truth,sus_score,min_samples_list_total,eps_list_total,algorithm_total)
        
        print ("cv_property,cv_label,sus_property,sus_label,min_samples,eps",cv_property,cv_label,sus_property,sus_label,min_samples,eps)
       
        score_property = w1 * cv_property + w2 * sus_property
        score_label = w1 * cv_label + w2 * sus_label

        # Store scores in the matrices
        score_property_matrix[x_ax, y_ax] = score_property
        score_label_matrix[x_ax, y_ax] = score_label 
        
        y_ax=y_ax+1
    x_ax=x_ax+1

xticks = np.linspace(0, len(final_comb_array), 5)
xticklabels = ['1.0', '2.0', '3.0', '4.0', '5.0']

fig_actual_cv.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_cv.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_cv.text(0.001, 0.52, "Heat Capacity", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_cv.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_cv.savefig("dbscan_cv_clusters.jpg", bbox_inches='tight')

fig_actual_sus.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_sus.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_sus.text(0.001, 0.52, "Susceptibility", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_sus.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_sus.savefig("dbscan_sus_clusters.jpg", bbox_inches='tight')

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
plt.savefig("heatmaps_macrostates.jpg")

