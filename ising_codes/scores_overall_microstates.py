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
eps=[0.30,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
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

plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_energy, axes_actual_energy = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_mag, axes_actual_mag = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_energy, axes_energy = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_mag, axes_mag = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')


subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    
    iterator = n_clusters
    y_ax=0
    for n in iterator:

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
        
        y_ax=y_ax+1
    x_ax=x_ax+1
fig_actual_energy.tight_layout()

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

# Example usage
w1 = 0.3
w2 = 0.7

result_average_property = weighted_average_dicts(energy_property_scores, mag_property_scores, w1, w2)
result_average_label = weighted_average_dicts(energy_label_scores, mag_label_scores, w1, w2)

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
plt.savefig("overall_scores_mag_energy.jpg")
        
        
