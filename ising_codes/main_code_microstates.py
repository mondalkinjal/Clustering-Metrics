import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from properties import IsingProperties
from scoring import scoring 
Clustering = Clustering()
Ising_properties = IsingProperties(lattice_size=30)
loaded_array=np.load("representation_microstates.npy")
properties={'Energy':Ising_properties.Energy,'Magnetization':Ising_properties.Magnetization}
Clustering_Algorithms={'KMeans':Clustering.KMeans, 'Agglomerative Clustering':Clustering.AgglomerativeClustering, 'Gaussian Mixture':Clustering.GaussianMixture, 'Birch Clustering':Clustering.BirchClustering}
n_clusters=[2,3,4,5,6]
final_lattice_arr=[]
temperatures = np.linspace(1.0, 5.0, 20)
energy_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
energy_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
mag_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
mag_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_microstates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)

    final_lattice_arr.append(final_arr.copy())
final_comb_array = np.concatenate(final_lattice_arr, axis=0)
final_comb_array = final_comb_array[99::100, ...]
energy=properties['Energy'](final_comb_array)
mag=properties['Magnetization'](final_comb_array)

xticks=np.linspace(0,len(final_comb_array),10)
xticklabels=np.linspace(1.0,5.0,10)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(mag, marker='o', linestyle='',color='b')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Magnetization')
ax1.set_xticks(xticks)
ax1.set_xticklabels(np.round(xticklabels,2))
ax1.grid(True, axis='x')
ax2.plot(energy,marker='o', linestyle='',color='b')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Energy')
ax2.set_xticks(xticks)

temperature=np.arange(len(final_comb_array)) 
plt.rcParams.update({'font.size': 30})
fig_actual_energy, axes_actual_energy = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_mag, axes_actual_mag = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_energy, axes_energy = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_mag, axes_mag = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
plt.subplots_adjust(wspace=0, hspace=0)

subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    
    y_ax=0
    for n in n_clusters:

        cluster_labels=algorithm(loaded_array,n)
        energy=properties['Energy'](final_comb_array)
        mag=properties['Magnetization'](final_comb_array)
        
        parameter = cluster_labels
        energy_property,energy_label,energy_clusters=scoring(np.array(energy),cluster_labels,name,parameter,0.20,algorithm)
        mag_property,mag_label,mag_clusters=scoring(np.array(mag),cluster_labels,name,parameter,0.20,algorithm)
        
        energy_property_scores[name].append(energy_property)
        energy_label_scores[name].append(energy_label)
        mag_property_scores[name].append(mag_property)
        mag_label_scores[name].append(mag_label)
        
        label = f"{row_labels[x_ax]}{col_labels[y_ax]}"
        
        
        ax_actual_mag = axes_actual_mag[x_ax,y_ax]
        scatter_actual_mag = ax_actual_mag.scatter(temperature, mag, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_mag.text(len(final_comb_array)/2, 1.00, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_mag.grid(True)
        handles_actual_mag, labels_actual_mag = scatter_actual_mag.legend_elements()
        ax_actual_mag.legend(handles_actual_mag, labels_actual_mag, title="Clusters")
        

        ax_mag = axes_mag[x_ax,y_ax]
        scatter_mag = ax_mag.scatter(temperature, mag, c=mag_clusters, cmap='viridis', marker='o')
        ax_mag.text(len(final_comb_array)/2, 1.00, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_mag.grid(True)
        handles_mag, labels_mag = scatter_mag.legend_elements()
        ax_mag.legend(handles_mag, labels_mag, title="Clusters")
        
        ax_actual_energy = axes_actual_energy[x_ax,y_ax]
        scatter_actual_energy = ax_actual_energy.scatter(temperature, energy, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_energy.text(len(final_comb_array)/2,-0.25, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_energy.grid(True)
        handles_actual_energy, labels_actual_energy = scatter_actual_energy.legend_elements()
        ax_actual_energy.legend(handles_actual_energy, labels_actual_energy, title="Clusters")
    
        ax_energy = axes_energy[x_ax,y_ax]
        scatter_energy = ax_energy.scatter(temperature, energy, c=energy_clusters, cmap='viridis', marker='o')
        ax_energy.text(len(final_comb_array)/2,-0.25, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_energy.grid(True)
        handles_energy, labels_energy = scatter_energy.legend_elements()
        ax_energy.legend(handles_energy, labels_energy, title="Clusters")
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
fig_actual_energy.savefig("actual_energy_clusters.jpg", bbox_inches='tight')

fig_actual_mag.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_mag.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_mag.text(0.001, 0.52, "Magnetization", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_mag.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_mag.savefig("actual_mag_clusters.jpg", bbox_inches='tight')

fig_energy.tight_layout(h_pad=0.0, w_pad=0.0)
fig_energy.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_energy.text(0.001, 0.52, "Energy", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_energy.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_energy.savefig("actual_energy.jpg", bbox_inches='tight')

fig_mag.tight_layout(h_pad=0.0, w_pad=0.0)
fig_mag.text(0.5, 0.0005, "Temperature", ha='center', va='center', fontweight='bold', fontsize=40)
fig_mag.text(0.001, 0.52, "Magnetization", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_mag.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_mag.savefig("actual_mag.jpg", bbox_inches='tight')

def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)

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

plot_bars(axes[0], energy_property_scores, 'A')
plot_bars(axes[1], energy_label_scores, 'B')
plot_bars(axes[2], mag_property_scores, 'C')
plot_bars(axes[3], mag_label_scores, 'D')

plt.tight_layout()
plt.savefig("scores_mag_energy.jpg")
        
