import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from properties import IsingProperties
from scoring import scoring 
Clustering = Clustering()
lattice_size=30
Ising_properties = IsingProperties(lattice_size=30)
loaded_array=np.load("representation_macrostates.npy")
properties={'specific_heat':Ising_properties.specific_heat,'sus':Ising_properties.sus}
Clustering_Algorithms={'KMeans':Clustering.KMeans, 'Agglomerative Clustering':Clustering.AgglomerativeClustering,'Gaussian Mixture':Clustering.GaussianMixture, 'Birch Clustering':Clustering.BirchClustering}
n_clusters=[2,3,4,5,6]
final_lattice_arr=[]
temperatures = np.linspace(1.0, 5.0, 200)
for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_macrostates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)
    final_lattice_arr.append(final_arr)

final_comb_array = np.stack(final_lattice_arr, axis=0)
final_comb_array = final_comb_array[:,99::100,:,:]
cv=properties['specific_heat'](final_comb_array,temperatures)
sus=properties['sus'](final_comb_array,temperatures)
xticks=np.linspace(0,len(final_comb_array),10)
xticklabels=np.linspace(1.0,5.0,10)
cv_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
cv_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
sus_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
sus_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(cv, marker='o', linestyle='',color='b')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Specific Heat')
ax1.set_xticks(xticks)
ax1.set_xticklabels(np.round(xticklabels,2))
ax1.grid(True, axis='x')
ax2.plot(sus,marker='o', linestyle='',color='b')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Susceptibility')
ax2.set_xticks(xticks)
ax2.set_xticklabels(np.round(xticklabels,2))
ax2.grid(True, axis='x')
plt.tight_layout()
plt.savefig("combined_cv_sus.jpg")
plt.show()
fig, axes = plt.subplots(4, len(n_clusters), figsize=(40, 40))
temperature=np.arange(len(final_comb_array))

plt.rcParams.update({'font.size': 30})
fig_actual_cv, axes_actual_cv = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_sus, axes_actual_sus = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_cv, axes_cv = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_sus, axes_sus = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
plt.subplots_adjust(wspace=0, hspace=0)
subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    y_ax=0
    for n in n_clusters:
        cluster_labels=algorithm(loaded_array,n)
        
        cv=properties['specific_heat'](final_comb_array,temperatures)
        sus=properties['sus'](final_comb_array,temperatures)
        cv=cv*lattice_size*lattice_size
        sus=sus*lattice_size*lattice_size
        cv_property,cv_label,cv_clusters=scoring(cv,cluster_labels,name,n,0.40,algorithm)
        sus_property,sus_label,sus_clusters=scoring(sus,cluster_labels,name,n,0.40,algorithm)
        cv_property_scores[name].append(cv_property)
        cv_label_scores[name].append(cv_label)
        sus_property_scores[name].append(sus_property)
        sus_label_scores[name].append(sus_label)  
        label = f"{row_labels[x_ax]}{col_labels[y_ax]}"
        
        ax_actual_cv = axes_actual_cv[x_ax,y_ax]
        scatter_actual_cv = ax_actual_cv.scatter(temperature, cv, c=cluster_labels, cmap='viridis', marker='o')
        
        ax_actual_cv.text(len(final_comb_array)/2, 2.20, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_cv.grid(True)
        handles_actual_cv, labels_actual_cv = scatter_actual_cv.legend_elements()
        ax_actual_cv.legend(handles_actual_cv, labels_actual_cv, title="Clusters")
        
        ax_cv = axes_cv[x_ax,y_ax]
        scatter_cv = ax_cv.scatter(temperature, cv, c=cv_clusters, cmap='viridis', marker='o')
        ax_cv.text(len(final_comb_array)/2, 2.20, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_cv.grid(True)
        handles_cv, labels_cv = scatter_cv.legend_elements()
        ax_cv.legend(handles_cv, labels_cv, title="Clusters")

        ax_actual_sus = axes_actual_sus[x_ax,y_ax]
        scatter_actual_sus = ax_actual_sus.scatter(temperature, sus, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_sus.text(len(final_comb_array)/2, 20.0, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_sus.grid(True)
        handles_actual_sus, labels_actual_sus = scatter_actual_sus.legend_elements()
        ax_actual_sus.legend(handles_actual_sus, labels_actual_sus, title="Clusters")

        ax_sus = axes_sus[x_ax,y_ax]
        scatter_sus = ax_sus.scatter(temperature, sus, c=sus_clusters, cmap='viridis', marker='o')
        ax_sus.text(len(final_comb_array)/2, 20.0, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_sus.grid(True)
        handles_sus, labels_sus = scatter_sus.legend_elements()
        ax_sus.legend(handles_sus, labels_sus, title="Clusters")
        
        y_ax=y_ax+1
    x_ax=x_ax+1
plt.savefig("clusters_cv_sus.jpg")

xticks = np.linspace(0, len(final_comb_array), 5)
xticklabels = ['1.0', '2.0', '3.0', '4.0', '5.0']

fig_actual_cv.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_cv.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_cv.text(0.001, 0.52, "Heat Capacity", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_cv.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_cv.savefig("actual_cv_clusters.jpg", bbox_inches='tight')

fig_actual_sus.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_sus.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_sus.text(0.001, 0.52, "Susceptibility", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_sus.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_sus.savefig("actual_sus_clusters.jpg", bbox_inches='tight')

fig_cv.tight_layout(h_pad=0.0, w_pad=0.0)
fig_cv.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_cv.text(0.001, 0.52, "Heat Capacity", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_cv.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_cv.savefig("actual_cv.jpg", bbox_inches='tight')

fig_sus.tight_layout(h_pad=0.0, w_pad=0.0)
fig_sus.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_sus.text(0.001, 0.52, "Susceptibility", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_sus.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_sus.savefig("actual_sus.jpg", bbox_inches='tight')

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

plot_bars(axes[0], cv_property_scores, 'A')
plot_bars(axes[1], cv_label_scores, 'B')
plot_bars(axes[2], sus_property_scores, 'C')
plot_bars(axes[3], sus_label_scores, 'D')


plt.tight_layout()
plt.savefig("scores_cv_sus.jpg")

